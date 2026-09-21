#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train configured project stage-1 spatial block CV models.

This stage creates regular-grid spatial blocks from sample x/y coordinates,
builds StratifiedGroupKFold train/test folds within each site, trains the
configured stage-1 models, and writes fold-level test predictions. It does not
compute evaluation metrics, run cross-site validation, predict maps, or create
figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    configured_feature_sets,
    configured_model_persistence,
    ensure_output_dirs,
    get_site_config,
    load_all_model_configs,
    load_experiment_matrix,
    load_project_config,
    load_sites_config,
)
from utils.modeling import (
    build_model,
    extract_feature_importance,
    model_parts,
    predict_positive_scores,
    save_model,
)
from utils.splitters import (
    assign_spatial_blocks,
    make_spatial_block_folds,
    summarize_spatial_blocks,
    validate_spatial_fold_classes,
)
from utils.training_io import (
    json_string,
    read_samples,
    write_feature_importance,
    write_spatial_block_predictions,
    write_spatial_block_train_summary,
)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Validate that ``value`` is a mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    """Validate that ``value`` is a list."""

    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list, got {type(value).__name__}.")
    return value


def _filter_requested_value(
    values: Sequence[Any],
    requested: Any | None,
    *,
    option_name: str,
    matrix_name: str,
) -> list[Any]:
    """Return all configured values or the one requested CLI filter value."""

    values_list = list(values)
    if requested is None:
        return values_list
    if requested not in values_list:
        available = ", ".join(str(value) for value in values_list)
        raise ValueError(
            f"{option_name}={requested!r} is not listed in {matrix_name}. "
            f"Available values: {available}"
        )
    return [requested]


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    """Return positive/negative sample counts for binary labels."""

    return {
        "positive": int(np.count_nonzero(labels == 1)),
        "negative": int(np.count_nonzero(labels == 0)),
    }


def _read_spatial_block_cv_config(experiment_matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated spatial block CV stage-1 settings."""

    validation = _require_mapping(experiment_matrix.get("validation"), "validation")
    spatial = _require_mapping(
        validation.get("spatial_block_cv"),
        "validation.spatial_block_cv",
    )
    if not bool(spatial.get("enable")):
        raise ValueError(
            "validation.spatial_block_cv.enable is false; "
            "09_train_spatial_block_cv.py requires it."
        )

    required = [
        "split_method",
        "n_splits",
        "block_size_m",
        "origin",
        "shuffle_blocks",
        "min_positive_per_fold",
        "min_negative_per_fold",
        "stage1",
    ]
    missing = [key for key in required if key not in spatial]
    if missing:
        raise ValueError(
            "validation.spatial_block_cv is missing required key(s): "
            + ", ".join(missing)
        )

    split_method = spatial["split_method"]
    if split_method != "stratified_group_kfold":
        raise ValueError(
            f"Unsupported spatial_block_cv.split_method={split_method!r}; "
            "only 'stratified_group_kfold' is supported."
        )

    n_splits = spatial["n_splits"]
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("validation.spatial_block_cv.n_splits must be an integer >= 2.")

    block_size_m = spatial["block_size_m"]
    if isinstance(block_size_m, bool) or not isinstance(block_size_m, (int, float)):
        raise ValueError("validation.spatial_block_cv.block_size_m must be numeric.")
    if float(block_size_m) <= 0:
        raise ValueError("validation.spatial_block_cv.block_size_m must be > 0.")

    origin = spatial["origin"]
    if origin != "global_projected_grid":
        raise ValueError(
            f"Unsupported spatial_block_cv.origin={origin!r}; "
            "only 'global_projected_grid' is supported."
        )

    shuffle_blocks = spatial["shuffle_blocks"]
    if not isinstance(shuffle_blocks, bool):
        raise ValueError("validation.spatial_block_cv.shuffle_blocks must be a bool.")

    min_positive_per_fold = spatial["min_positive_per_fold"]
    if (
        isinstance(min_positive_per_fold, bool)
        or not isinstance(min_positive_per_fold, int)
        or min_positive_per_fold < 1
    ):
        raise ValueError(
            "validation.spatial_block_cv.min_positive_per_fold must be an integer >= 1."
        )

    min_negative_per_fold = spatial["min_negative_per_fold"]
    if (
        isinstance(min_negative_per_fold, bool)
        or not isinstance(min_negative_per_fold, int)
        or min_negative_per_fold < 1
    ):
        raise ValueError(
            "validation.spatial_block_cv.min_negative_per_fold must be an integer >= 1."
        )

    stage1 = _require_mapping(spatial["stage1"], "validation.spatial_block_cv.stage1")
    model_ids = _require_list(stage1.get("models"), "validation.spatial_block_cv.stage1.models")
    if not model_ids:
        raise ValueError("validation.spatial_block_cv.stage1.models must not be empty.")
    if any(not isinstance(model_id, str) for model_id in model_ids):
        raise ValueError("validation.spatial_block_cv.stage1.models must contain only strings.")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError(
            f"validation.spatial_block_cv.stage1.models contains duplicate values: {model_ids}"
        )
    feature_set_ids = configured_feature_sets(experiment_matrix)

    return {
        "split_method": str(split_method),
        "n_splits": int(n_splits),
        "block_size_m": float(block_size_m),
        "origin": str(origin),
        "shuffle_blocks": bool(shuffle_blocks),
        "min_positive_per_fold": int(min_positive_per_fold),
        "min_negative_per_fold": int(min_negative_per_fold),
        "models": list(model_ids),
        "feature_sets": list(feature_set_ids),
    }


def _validate_fold_filter(fold_id_filter: int | None, n_splits: int) -> None:
    """Validate an optional fold id CLI filter."""

    if fold_id_filter is None:
        return
    if isinstance(fold_id_filter, bool) or not isinstance(fold_id_filter, int):
        raise ValueError("--fold-id must be an integer.")
    if fold_id_filter < 0 or fold_id_filter >= n_splits:
        raise ValueError(
            f"--fold-id must be between 0 and {n_splits - 1}; got {fold_id_filter}."
        )


def _save_spatial_fold_outputs(
    *,
    site_id: str,
    site_name: str,
    feature_set_id: str,
    model_id: str,
    model_config: Mapping[str, Any],
    seed: int,
    fold_id: int,
    samples: Any,
    labels: np.ndarray,
    block_ids: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    feature_columns: Sequence[str],
    samples_path: Path,
    model_dir: Path,
    cv_config: Mapping[str, Any],
    n_blocks_total: int,
    fold_counts: Mapping[str, int],
    save_validation_model: bool = True,
) -> dict[str, Any]:
    """Train one spatial-block fold and save all fold-level outputs."""

    model_meta, _ = model_parts(model_id, model_config)
    model_type = str(model_meta["type"])

    train_samples = samples.iloc[train_positions]
    test_samples = samples.iloc[test_positions]
    x_train = train_samples[list(feature_columns)]
    y_train = labels[train_positions]
    x_test = test_samples[list(feature_columns)]
    y_test = labels[test_positions]

    model, effective_params = build_model(model_config, seed)
    model.fit(x_train, y_train)

    y_score = predict_positive_scores(model, x_test, model_id=model_id)
    importance, importance_type = extract_feature_importance(
        model,
        model_id=model_id,
        model_type=model_type,
        feature_columns=feature_columns,
    )

    fold_dir = model_dir / f"seed_{seed}" / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "model.joblib"
    predictions_path = fold_dir / "predictions.csv"
    feature_importance_path = fold_dir / "feature_importance.csv"

    test_block_ids = block_ids[test_positions]
    if save_validation_model:
        save_model(model, model_path)
    write_spatial_block_predictions(
        predictions_path,
        site_id=site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
        fold_id=fold_id,
        test_indices=samples.index.to_numpy()[test_positions],
        test_samples=test_samples,
        test_block_ids=test_block_ids,
        y_test=y_test,
        y_score=y_score,
    )
    write_feature_importance(
        feature_importance_path,
        site_id=site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
        feature_columns=feature_columns,
        importance=importance,
        importance_type=importance_type,
    )

    return {
        "validation_type": "spatial_block_cv",
        "site_id": site_id,
        "site_name": site_name,
        "feature_set": feature_set_id,
        "model_id": model_id,
        "model_type": model_type,
        "seed": seed,
        "fold_id": fold_id,
        "samples_path": str(samples_path),
        "output_dir": str(fold_dir),
        "block_size_m": float(cv_config["block_size_m"]),
        "n_splits": int(cv_config["n_splits"]),
        "split_method": cv_config["split_method"],
        "n_blocks_total": int(n_blocks_total),
        "train_block_count": int(fold_counts["train_block_count"]),
        "test_block_count": int(fold_counts["test_block_count"]),
        "test_block_ids": json_string(sorted(str(block_id) for block_id in set(test_block_ids))),
        "train_sample_count": int(train_samples.shape[0]),
        "test_sample_count": int(test_samples.shape[0]),
        "train_positive_count": int(fold_counts["train_positive"]),
        "train_negative_count": int(fold_counts["train_negative"]),
        "test_positive_count": int(fold_counts["test_positive"]),
        "test_negative_count": int(fold_counts["test_negative"]),
        "model_path": str(model_path) if save_validation_model else "",
        "predictions_path": str(predictions_path),
        "feature_importance_path": str(feature_importance_path),
        "n_features": len(feature_columns),
        "feature_columns": json_string(list(feature_columns)),
        "params": json_string(effective_params),
        "status": "ok",
        "message": "",
    }


def process_spatial_block_model_runs(
    *,
    site_id: str,
    site_name: str,
    feature_set_id: str,
    model_id: str,
    model_config: Mapping[str, Any],
    seeds: Sequence[int],
    samples: Any,
    feature_columns: Sequence[str],
    labels: np.ndarray,
    block_ids: np.ndarray,
    samples_path: Path,
    spatial_block_root: Path,
    cv_config: Mapping[str, Any],
    fold_id_filter: int | None = None,
    save_validation_model: bool = True,
) -> Path:
    """Train all selected spatial-block CV seed/fold runs for one model."""

    block_summary = summarize_spatial_blocks(samples, block_ids, labels)
    n_blocks_total = int(block_summary["n_blocks_total"])
    model_dir = spatial_block_root / site_id / feature_set_id / model_id
    summary_rows = []

    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"experiment_matrix.seeds must contain only integers, got {seed!r}.")

        folds = make_spatial_block_folds(
            labels=labels,
            block_ids=block_ids,
            n_splits=int(cv_config["n_splits"]),
            seed=int(seed),
            shuffle=bool(cv_config["shuffle_blocks"]),
        )
        for fold_id, (train_positions, test_positions) in enumerate(folds):
            if fold_id_filter is not None and fold_id != fold_id_filter:
                continue

            fold_counts = validate_spatial_fold_classes(
                labels=labels,
                block_ids=block_ids,
                train_positions=train_positions,
                test_positions=test_positions,
                site_id=site_id,
                feature_set_id=feature_set_id,
                seed=int(seed),
                fold_id=fold_id,
                block_size_m=float(cv_config["block_size_m"]),
                min_positive_per_fold=int(cv_config["min_positive_per_fold"]),
                min_negative_per_fold=int(cv_config["min_negative_per_fold"]),
            )

            print(
                f"Training spatial-block CV: {site_id} / {feature_set_id} / "
                f"{model_id} / seed_{seed} / fold_{fold_id}",
                flush=True,
            )
            summary_rows.append(
                _save_spatial_fold_outputs(
                    site_id=site_id,
                    site_name=site_name,
                    feature_set_id=feature_set_id,
                    model_id=model_id,
                    model_config=model_config,
                    seed=int(seed),
                    fold_id=fold_id,
                    samples=samples,
                    labels=labels,
                    block_ids=block_ids,
                    train_positions=train_positions,
                    test_positions=test_positions,
                    feature_columns=feature_columns,
                    samples_path=samples_path,
                    model_dir=model_dir,
                    cv_config=cv_config,
                    n_blocks_total=n_blocks_total,
                    fold_counts=fold_counts,
                    save_validation_model=save_validation_model,
                )
            )
            print(
                f"Trained spatial-block CV: {site_id} / {feature_set_id} / "
                f"{model_id} / seed_{seed} / fold_{fold_id}",
                flush=True,
            )

    if not summary_rows:
        raise ValueError(
            f"No spatial block CV folds were trained: site_id={site_id}, "
            f"feature_set={feature_set_id}, model_id={model_id}, fold_id_filter={fold_id_filter}"
        )

    summary_path = model_dir / "train_summary.csv"
    write_spatial_block_train_summary(summary_path, summary_rows)
    print(f"Spatial-block train summary written to: {summary_path}", flush=True)
    return summary_path


def run_spatial_block_cv_training(
    project_id: str = DEFAULT_PROJECT_ID,
    *,
    site_filter: str | None = None,
    feature_set_filter: str | None = None,
    model_id_filter: str | None = None,
    seed_filter: int | None = None,
    fold_id_filter: int | None = None,
) -> list[Path]:
    """Train all configured project stage-1 spatial block CV runs."""

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    experiment_matrix = load_experiment_matrix(project_id=project_id)
    model_configs = load_all_model_configs(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="09")
    cv_config = _read_spatial_block_cv_config(experiment_matrix)
    _validate_fold_filter(fold_id_filter, int(cv_config["n_splits"]))
    persistence = configured_model_persistence(project_config)

    site_ids = _require_list(experiment_matrix.get("sites"), "experiment_matrix.sites")
    matrix_model_ids = _require_list(experiment_matrix.get("models"), "experiment_matrix.models")
    matrix_seeds = _require_list(experiment_matrix.get("seeds"), "experiment_matrix.seeds")

    site_ids = _filter_requested_value(
        site_ids,
        site_filter,
        option_name="--site",
        matrix_name="experiment_matrix.sites",
    )
    feature_set_ids = _filter_requested_value(
        cv_config["feature_sets"],
        feature_set_filter,
        option_name="--feature-set",
        matrix_name="experiment_matrix.feature_sets",
    )
    model_ids = _filter_requested_value(
        cv_config["models"],
        model_id_filter,
        option_name="--model-id",
        matrix_name="validation.spatial_block_cv.stage1.models",
    )
    seeds = _filter_requested_value(
        matrix_seeds,
        seed_filter,
        option_name="--seed",
        matrix_name="experiment_matrix.seeds",
    )

    for model_id in model_ids:
        if model_id not in matrix_model_ids:
            raise ValueError(
                f"Spatial block model_id={model_id!r} is not listed in experiment_matrix.models."
            )
        if model_id not in model_configs:
            raise KeyError(f"Model config was not loaded: {model_id}")

    samples_root = output_dirs["samples"]
    spatial_block_root = output_dirs["spatial_block_cv"]
    summary_paths: list[Path] = []

    for site_id in site_ids:
        if not isinstance(site_id, str):
            raise ValueError("experiment_matrix.sites must contain only strings.")
        site_config = get_site_config(sites_config, site_id)
        site_name = str(site_config.get("site_name", site_id))

        for feature_set_id in feature_set_ids:
            if not isinstance(feature_set_id, str):
                raise ValueError("experiment_matrix.feature_sets must contain only strings.")

            samples_path = samples_root / site_id / f"{feature_set_id}_samples.csv"
            samples, feature_columns, labels = read_samples(
                samples_path,
                site_id=site_id,
                feature_set_id=feature_set_id,
            )
            label_counts = _class_counts(labels)
            if label_counts["positive"] == 0 or label_counts["negative"] == 0:
                raise ValueError(
                    f"Spatial block CV requires both classes: site_id={site_id}, "
                    f"feature_set={feature_set_id}, class_counts={label_counts}"
                )

            block_ids = assign_spatial_blocks(
                samples,
                block_size_m=float(cv_config["block_size_m"]),
            )

            for model_id in model_ids:
                if not isinstance(model_id, str):
                    raise ValueError("spatial_block_cv.stage1.models must contain only strings.")

                summary_paths.append(
                    process_spatial_block_model_runs(
                        site_id=site_id,
                        site_name=site_name,
                        feature_set_id=feature_set_id,
                        model_id=model_id,
                        model_config=model_configs[model_id],
                        seeds=seeds,
                        samples=samples,
                        feature_columns=feature_columns,
                        labels=labels,
                        block_ids=block_ids,
                        samples_path=samples_path,
                        spatial_block_root=spatial_block_root,
                        cv_config=cv_config,
                        fold_id_filter=fold_id_filter,
                        save_validation_model=persistence["save_validation_models"],
                    )
                )

    return summary_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train configured project stage-1 spatial block CV models.",
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=f"Project id to process. Default: {DEFAULT_PROJECT_ID}",
    )
    parser.add_argument(
        "--site",
        default=None,
        help="Optional site id filter for smoke tests.",
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        help="Optional stage-1 feature-set id filter for smoke tests.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional stage-1 model id filter for smoke tests.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional integer seed filter for smoke tests.",
    )
    parser.add_argument(
        "--fold-id",
        default=None,
        type=int,
        help="Optional zero-based fold id filter for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    """Run spatial block CV training from the command line."""

    args = parse_args()
    run_spatial_block_cv_training(
        project_id=args.project_id,
        site_filter=args.site,
        feature_set_filter=args.feature_set,
        model_id_filter=args.model_id,
        seed_filter=args.seed,
        fold_id_filter=args.fold_id,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train cross-site transfer models from configured project sample CSV files.

This stage trains on all samples from one site and predicts all samples from a
different site. It does not evaluate thresholds or metrics, run spatial block
CV, predict maps, or create figures.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    configured_cross_site_pairs,
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
from utils.training_io import (
    json_string,
    read_samples,
    write_cross_site_predictions,
    write_feature_importance,
)


CROSS_SITE_TRAIN_SUMMARY_COLUMNS = [
    "validation_type",
    "train_site_id",
    "test_site_id",
    "feature_set",
    "model_id",
    "model_type",
    "seed",
    "train_samples_path",
    "test_samples_path",
    "output_dir",
    "model_path",
    "predictions_path",
    "feature_importance_path",
    "n_features",
    "feature_columns",
    "train_sample_count",
    "test_sample_count",
    "train_positive_count",
    "train_negative_count",
    "test_positive_count",
    "test_negative_count",
    "params",
    "status",
    "message",
]


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


def _pair_id(train_site_id: str, test_site_id: str) -> str:
    """Return the cross-site output directory id for one ordered pair."""

    return f"train_{train_site_id}_test_{test_site_id}"


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    """Return positive/negative sample counts for binary labels."""

    return {
        "positive": int(np.count_nonzero(labels == 1)),
        "negative": int(np.count_nonzero(labels == 0)),
    }


def _validate_has_both_classes(labels: np.ndarray, *, label: str) -> None:
    """Ensure a train/test sample set contains both binary classes."""

    counts = _class_counts(labels)
    if counts["positive"] == 0 or counts["negative"] == 0:
        raise ValueError(f"{label} samples must contain both 0/1 classes: {counts}")


def _validate_feature_columns(
    *,
    train_columns: Sequence[str],
    test_columns: Sequence[str],
    train_site_id: str,
    test_site_id: str,
    feature_set_id: str,
) -> list[str]:
    """Return shared feature columns after validating identical order."""

    train_list = list(train_columns)
    test_list = list(test_columns)
    if train_list != test_list:
        raise ValueError(
            "Train/test feature columns must match exactly and in the same order: "
            f"train_site_id={train_site_id}, test_site_id={test_site_id}, "
            f"feature_set={feature_set_id}"
        )
    return train_list


def _write_cross_site_train_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write model-level cross-site train summary CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CROSS_SITE_TRAIN_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _save_cross_site_seed_outputs(
    *,
    train_site_id: str,
    test_site_id: str,
    feature_set_id: str,
    model_id: str,
    model_config: Mapping[str, Any],
    seed: int,
    train_samples: Any,
    test_samples: Any,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    feature_columns: Sequence[str],
    train_samples_path: Path,
    test_samples_path: Path,
    model_dir: Path,
    save_validation_model: bool = True,
) -> dict[str, Any]:
    """Train one cross-site seed run and save seed-level outputs."""

    _validate_has_both_classes(train_labels, label=f"train_site_id={train_site_id}")
    _validate_has_both_classes(test_labels, label=f"test_site_id={test_site_id}")

    model_meta, _ = model_parts(model_id, model_config)
    model_type = str(model_meta["type"])

    x_train = train_samples[list(feature_columns)]
    y_train = train_samples["label"].to_numpy(dtype=int, copy=False)
    x_test = test_samples[list(feature_columns)]
    y_test = test_samples["label"].to_numpy(dtype=int, copy=False)

    model, effective_params = build_model(model_config, seed)
    model.fit(x_train, y_train)

    y_score = predict_positive_scores(model, x_test, model_id=model_id)
    importance, importance_type = extract_feature_importance(
        model,
        model_id=model_id,
        model_type=model_type,
        feature_columns=feature_columns,
    )

    seed_dir = model_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_path = seed_dir / "model.joblib"
    predictions_path = seed_dir / "predictions.csv"
    feature_importance_path = seed_dir / "feature_importance.csv"

    if save_validation_model:
        save_model(model, model_path)
    write_cross_site_predictions(
        predictions_path,
        train_site_id=train_site_id,
        test_site_id=test_site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
        test_indices=test_samples.index.to_numpy(),
        test_samples=test_samples,
        y_test=y_test,
        y_score=y_score,
    )
    write_feature_importance(
        feature_importance_path,
        site_id=train_site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
        feature_columns=feature_columns,
        importance=importance,
        importance_type=importance_type,
    )

    train_counts = _class_counts(y_train)
    test_counts = _class_counts(y_test)
    return {
        "validation_type": "cross_site",
        "train_site_id": train_site_id,
        "test_site_id": test_site_id,
        "feature_set": feature_set_id,
        "model_id": model_id,
        "model_type": model_type,
        "seed": seed,
        "train_samples_path": str(train_samples_path),
        "test_samples_path": str(test_samples_path),
        "output_dir": str(seed_dir),
        "model_path": str(model_path) if save_validation_model else "",
        "predictions_path": str(predictions_path),
        "feature_importance_path": str(feature_importance_path),
        "n_features": len(feature_columns),
        "feature_columns": json_string(list(feature_columns)),
        "train_sample_count": int(train_samples.shape[0]),
        "test_sample_count": int(test_samples.shape[0]),
        "train_positive_count": train_counts["positive"],
        "train_negative_count": train_counts["negative"],
        "test_positive_count": test_counts["positive"],
        "test_negative_count": test_counts["negative"],
        "params": json_string(effective_params),
        "status": "ok",
        "message": "",
    }


def process_cross_site_model_runs(
    *,
    train_site_id: str,
    test_site_id: str,
    feature_set_id: str,
    model_id: str,
    model_config: Mapping[str, Any],
    seeds: Sequence[int],
    train_samples: Any,
    test_samples: Any,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    feature_columns: Sequence[str],
    train_samples_path: Path,
    test_samples_path: Path,
    cross_site_root: Path,
    save_validation_model: bool = True,
) -> Path:
    """Train all seed runs for one cross-site pair/feature-set/model."""

    model_dir = (
        cross_site_root
        / _pair_id(train_site_id, test_site_id)
        / feature_set_id
        / model_id
    )
    summary_rows = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"experiment_matrix.seeds must contain only integers, got {seed!r}.")
        print(
            f"Training cross-site: {train_site_id} -> {test_site_id} / "
            f"{feature_set_id} / {model_id} / seed_{seed}",
            flush=True,
        )
        summary_rows.append(
            _save_cross_site_seed_outputs(
                train_site_id=train_site_id,
                test_site_id=test_site_id,
                feature_set_id=feature_set_id,
                model_id=model_id,
                model_config=model_config,
                seed=int(seed),
                train_samples=train_samples,
                test_samples=test_samples,
                train_labels=train_labels,
                test_labels=test_labels,
                feature_columns=feature_columns,
                train_samples_path=train_samples_path,
                test_samples_path=test_samples_path,
                model_dir=model_dir,
                save_validation_model=save_validation_model,
            )
        )
        print(
            f"Trained cross-site: {train_site_id} -> {test_site_id} / "
            f"{feature_set_id} / {model_id} / seed_{seed}",
            flush=True,
        )

    summary_path = model_dir / "train_summary.csv"
    _write_cross_site_train_summary(summary_path, summary_rows)
    print(f"Cross-site train summary written to: {summary_path}", flush=True)
    return summary_path


def run_cross_site_training(
    project_id: str = DEFAULT_PROJECT_ID,
    *,
    train_site: str | None = None,
    test_site: str | None = None,
    feature_set: str | None = None,
    model_id_filter: str | None = None,
    seed_filter: int | None = None,
) -> list[Path]:
    """Train the directional cross-site pairs configured for the project."""

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    experiment_matrix = load_experiment_matrix(project_id=project_id)
    model_configs = load_all_model_configs(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="07")
    persistence = configured_model_persistence(project_config)

    feature_set_ids = _require_list(
        experiment_matrix.get("feature_sets"),
        "experiment_matrix.feature_sets",
    )
    model_ids = _require_list(experiment_matrix.get("models"), "experiment_matrix.models")
    seeds = _require_list(experiment_matrix.get("seeds"), "experiment_matrix.seeds")

    cross_site_pairs = configured_cross_site_pairs(
        experiment_matrix,
        train_site=train_site,
        test_site=test_site,
    )
    feature_set_ids = _filter_requested_value(
        feature_set_ids,
        feature_set,
        option_name="--feature-set",
        matrix_name="experiment_matrix.feature_sets",
    )
    model_ids = _filter_requested_value(
        model_ids,
        model_id_filter,
        option_name="--model-id",
        matrix_name="experiment_matrix.models",
    )
    seeds = _filter_requested_value(
        seeds,
        seed_filter,
        option_name="--seed",
        matrix_name="experiment_matrix.seeds",
    )

    samples_root = output_dirs["samples"]
    cross_site_root = output_dirs["cross_site"]
    summary_paths: list[Path] = []

    for train_site_id, test_site_id in cross_site_pairs:
        get_site_config(sites_config, train_site_id)
        get_site_config(sites_config, test_site_id)

        for feature_set_id in feature_set_ids:
            if not isinstance(feature_set_id, str):
                raise ValueError("experiment_matrix.feature_sets must contain only strings.")

            train_samples_path = samples_root / train_site_id / f"{feature_set_id}_samples.csv"
            test_samples_path = samples_root / test_site_id / f"{feature_set_id}_samples.csv"
            train_samples, train_feature_columns, train_labels = read_samples(
                train_samples_path,
                site_id=train_site_id,
                feature_set_id=feature_set_id,
                min_class_count=1,
            )
            test_samples, test_feature_columns, test_labels = read_samples(
                test_samples_path,
                site_id=test_site_id,
                feature_set_id=feature_set_id,
                min_class_count=1,
            )
            feature_columns = _validate_feature_columns(
                train_columns=train_feature_columns,
                test_columns=test_feature_columns,
                train_site_id=train_site_id,
                test_site_id=test_site_id,
                feature_set_id=feature_set_id,
            )

            for model_id in model_ids:
                if not isinstance(model_id, str):
                    raise ValueError("experiment_matrix.models must contain only strings.")
                if model_id not in model_configs:
                    raise KeyError(f"Model config was not loaded: {model_id}")

                summary_paths.append(
                    process_cross_site_model_runs(
                        train_site_id=train_site_id,
                        test_site_id=test_site_id,
                        feature_set_id=feature_set_id,
                        model_id=model_id,
                        model_config=model_configs[model_id],
                        seeds=seeds,
                        train_samples=train_samples,
                        test_samples=test_samples,
                        train_labels=train_labels,
                        test_labels=test_labels,
                        feature_columns=feature_columns,
                        train_samples_path=train_samples_path,
                        test_samples_path=test_samples_path,
                        cross_site_root=cross_site_root,
                        save_validation_model=persistence["save_validation_models"],
                    )
                )

    return summary_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train configured project cross-site transfer binary classification models.",
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=f"Project id to process. Default: {DEFAULT_PROJECT_ID}",
    )
    parser.add_argument(
        "--train-site",
        default=None,
        help="Optional train site id filter for smoke tests.",
    )
    parser.add_argument(
        "--test-site",
        default=None,
        help="Optional test site id filter for smoke tests.",
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        help="Optional feature-set id filter for smoke tests.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional model id filter for smoke tests.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional integer seed filter for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    """Run cross-site training from the command line."""

    args = parse_args()
    run_cross_site_training(
        project_id=args.project_id,
        train_site=args.train_site,
        test_site=args.test_site,
        feature_set=args.feature_set,
        model_id_filter=args.model_id,
        seed_filter=args.seed,
    )


if __name__ == "__main__":
    main()

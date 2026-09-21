#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train within-site binary classifiers from configured project sample CSV files.

This stage performs only within-site train/test splitting, model fitting, test
set probability prediction, model serialization, and feature-importance export.
Threshold selection, cross-site evaluation, map prediction, and figures belong
to later pipeline stages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.config_io import (
    DEFAULT_PROJECT_ID,
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
from utils.splitters import within_site_stratified_split
from utils.training_io import (
    json_string,
    read_samples,
    write_feature_importance,
    write_predictions,
    write_train_summary,
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


def _read_within_site_config(experiment_matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated within-site split settings."""

    validation = _require_mapping(experiment_matrix.get("validation"), "validation")
    within_site = _require_mapping(validation.get("within_site"), "validation.within_site")

    if not bool(within_site.get("enable")):
        raise ValueError("validation.within_site.enable is false; 05_train_within_site.py requires it.")

    split_method = within_site.get("split_method")
    if split_method != "random_stratified":
        raise ValueError(
            f"Unsupported within-site split_method={split_method!r}; "
            "only 'random_stratified' is supported."
        )

    train_fraction = within_site.get("train_fraction")
    test_fraction = within_site.get("test_fraction")
    if isinstance(train_fraction, bool) or not isinstance(train_fraction, (int, float)):
        raise ValueError("validation.within_site.train_fraction must be numeric.")
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, (int, float)):
        raise ValueError("validation.within_site.test_fraction must be numeric.")
    if abs(float(train_fraction) + float(test_fraction) - 1.0) > 1e-6:
        raise ValueError("within-site train_fraction + test_fraction must equal 1.0.")

    stratify_by_label = within_site.get("stratify_by_label")
    if not isinstance(stratify_by_label, bool):
        raise ValueError("validation.within_site.stratify_by_label must be a bool.")

    return {
        "split_method": split_method,
        "train_fraction": float(train_fraction),
        "test_fraction": float(test_fraction),
        "stratify_by_label": stratify_by_label,
    }


def _save_seed_outputs(
    *,
    samples: Any,
    labels: np.ndarray,
    feature_columns: Sequence[str],
    model_config: Mapping[str, Any],
    split_config: Mapping[str, Any],
    site_id: str,
    site_name: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    samples_path: Path,
    model_dir: Path,
    save_validation_model: bool = True,
) -> dict[str, Any]:
    """Train one seed run and save all seed-level outputs."""

    model_meta, _ = model_parts(model_id, model_config)
    model_type = str(model_meta["type"])

    sample_indices = samples.index.to_numpy()
    train_indices, test_indices = within_site_stratified_split(
        sample_indices=sample_indices,
        labels=labels,
        split_config=split_config,
        seed=seed,
    )

    train_samples = samples.loc[train_indices]
    test_samples = samples.loc[test_indices]
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
    train_samples_path = seed_dir / "train_samples.csv"
    test_samples_path = seed_dir / "test_samples.csv"
    predictions_path = seed_dir / "predictions.csv"
    feature_importance_path = seed_dir / "feature_importance.csv"

    if save_validation_model:
        save_model(model, model_path)
    train_samples.to_csv(train_samples_path, index=False)
    test_samples.to_csv(test_samples_path, index=False)
    write_predictions(
        predictions_path,
        site_id=site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
        test_indices=test_indices,
        test_samples=test_samples,
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
        "site_id": site_id,
        "site_name": site_name,
        "feature_set": feature_set_id,
        "model_id": model_id,
        "model_type": model_type,
        "seed": seed,
        "samples_path": str(samples_path),
        "output_dir": str(seed_dir),
        "model_path": str(model_path) if save_validation_model else "",
        "train_samples_path": str(train_samples_path),
        "test_samples_path": str(test_samples_path),
        "predictions_path": str(predictions_path),
        "feature_importance_path": str(feature_importance_path),
        "n_features": len(feature_columns),
        "feature_columns": json_string(list(feature_columns)),
        "total_samples": int(samples.shape[0]),
        "train_samples": int(train_samples.shape[0]),
        "test_samples": int(test_samples.shape[0]),
        "train_positive": int(np.count_nonzero(y_train == 1)),
        "train_negative": int(np.count_nonzero(y_train == 0)),
        "test_positive": int(np.count_nonzero(y_test == 1)),
        "test_negative": int(np.count_nonzero(y_test == 0)),
        "train_fraction": float(split_config["train_fraction"]),
        "test_fraction": float(split_config["test_fraction"]),
        "split_method": split_config["split_method"],
        "stratify_by_label": bool(split_config["stratify_by_label"]),
        "params": json_string(effective_params),
        "status": "ok",
        "message": "",
    }


def process_model_runs(
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
    samples_path: Path,
    within_site_root: Path,
    split_config: Mapping[str, Any],
    save_validation_model: bool = True,
) -> Path:
    """Train all seed runs for one site/feature-set/model combination."""

    model_dir = within_site_root / site_id / feature_set_id / model_id
    summary_rows = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"experiment_matrix.seeds must contain only integers, got {seed!r}.")
        print(f"Training: {site_id} / {feature_set_id} / {model_id} / seed_{seed}", flush=True)
        summary_rows.append(
            _save_seed_outputs(
                samples=samples,
                labels=labels,
                feature_columns=feature_columns,
                model_config=model_config,
                split_config=split_config,
                site_id=site_id,
                site_name=site_name,
                feature_set_id=feature_set_id,
                model_id=model_id,
                seed=int(seed),
                samples_path=samples_path,
                model_dir=model_dir,
                save_validation_model=save_validation_model,
            )
        )
        print(f"Trained: {site_id} / {feature_set_id} / {model_id} / seed_{seed}", flush=True)

    summary_path = model_dir / "train_summary.csv"
    write_train_summary(summary_path, summary_rows)
    return summary_path


def run_training(project_id: str = DEFAULT_PROJECT_ID) -> list[Path]:
    """Train all configured within-site model runs."""

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    experiment_matrix = load_experiment_matrix(project_id=project_id)
    model_configs = load_all_model_configs(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="05")
    split_config = _read_within_site_config(experiment_matrix)
    persistence = configured_model_persistence(project_config)

    site_ids = _require_list(experiment_matrix.get("sites"), "experiment_matrix.sites")
    feature_set_ids = _require_list(
        experiment_matrix.get("feature_sets"),
        "experiment_matrix.feature_sets",
    )
    model_ids = _require_list(experiment_matrix.get("models"), "experiment_matrix.models")
    seeds = _require_list(experiment_matrix.get("seeds"), "experiment_matrix.seeds")

    summary_paths: list[Path] = []
    for site_id in site_ids:
        if not isinstance(site_id, str):
            raise ValueError("experiment_matrix.sites must contain only strings.")
        site_config = get_site_config(sites_config, site_id)
        site_name = str(site_config.get("site_name", site_id))

        for feature_set_id in feature_set_ids:
            if not isinstance(feature_set_id, str):
                raise ValueError("experiment_matrix.feature_sets must contain only strings.")

            samples_path = output_dirs["samples"] / site_id / f"{feature_set_id}_samples.csv"
            samples, feature_columns, labels = read_samples(
                samples_path,
                site_id=site_id,
                feature_set_id=feature_set_id,
            )

            for model_id in model_ids:
                if not isinstance(model_id, str):
                    raise ValueError("experiment_matrix.models must contain only strings.")
                if model_id not in model_configs:
                    raise KeyError(f"Model config was not loaded: {model_id}")

                summary_paths.append(
                    process_model_runs(
                        site_id=site_id,
                        site_name=site_name,
                        feature_set_id=feature_set_id,
                        model_id=model_id,
                        model_config=model_configs[model_id],
                        seeds=seeds,
                        samples=samples,
                        feature_columns=feature_columns,
                        labels=labels,
                        samples_path=samples_path,
                        within_site_root=output_dirs["within_site"],
                        split_config=split_config,
                        save_validation_model=persistence["save_validation_models"],
                    )
                )

    return summary_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train configured project within-site binary classification models.",
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=f"Project id to process. Default: {DEFAULT_PROJECT_ID}",
    )
    return parser.parse_args()


def main() -> None:
    """Run within-site training from the command line."""

    args = parse_args()
    summary_paths = run_training(project_id=args.project_id)
    for summary_path in summary_paths:
        print(f"Train summary written to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

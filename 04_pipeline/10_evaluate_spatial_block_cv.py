#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate configured project spatial block CV prediction files.

This stage reads predictions produced by ``09_train_spatial_block_cv.py`` and
computes fold-level threshold selection, probability metrics,
threshold-dependent metrics, optional curves, and model/root summaries. It does
not retrain models, alter spatial folds, predict maps, or create figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    configured_feature_sets,
    ensure_output_dirs,
    get_site_config,
    load_experiment_matrix,
    load_project_config,
    load_sites_config,
)
from utils.evaluation_core import (
    aligned_pr_thresholds,
    evaluate_scores,
    roc_curve_arrays,
    spatial_block_curve_rows,
)
from utils.evaluation_io import (
    SPATIAL_BLOCK_SEED_METRIC_COLUMNS,
    read_spatial_block_predictions,
    summarize_spatial_block_evaluation_rows,
    validate_spatial_block_train_summary,
    write_csv_rows,
    write_spatial_block_confusion_matrix,
    write_spatial_block_curves,
    write_spatial_block_evaluation_metrics,
    write_spatial_block_summary,
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


def _read_evaluation_config(project_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated threshold and curve-output settings from project config."""

    analysis = _require_mapping(project_config.get("analysis"), "project_config.analysis")
    threshold_method = analysis.get("threshold_selection_method")
    if threshold_method != "precision_target":
        raise ValueError(
            f"Unsupported threshold_selection_method={threshold_method!r}; "
            "only 'precision_target' is supported."
        )

    precision_target = analysis.get("precision_target")
    if isinstance(precision_target, bool) or not isinstance(precision_target, (int, float)):
        raise ValueError("analysis.precision_target must be numeric.")
    if not 0 <= float(precision_target) <= 1:
        raise ValueError("analysis.precision_target must be between 0 and 1.")

    default_threshold = analysis.get("probability_threshold_default")
    if isinstance(default_threshold, bool) or not isinstance(default_threshold, (int, float)):
        raise ValueError("analysis.probability_threshold_default must be numeric.")

    save_curve_csv = analysis.get("save_curve_csv", True)
    if not isinstance(save_curve_csv, bool):
        raise ValueError("analysis.save_curve_csv must be a bool when provided.")

    max_curve_points = analysis.get("max_curve_points", 2000)
    if isinstance(max_curve_points, bool) or not isinstance(max_curve_points, int):
        raise ValueError("analysis.max_curve_points must be an integer when provided.")
    if int(max_curve_points) < 2:
        raise ValueError("analysis.max_curve_points must be >= 2.")

    curve_downsample_method = analysis.get("curve_downsample_method", "even")
    if curve_downsample_method != "even":
        raise ValueError(
            f"Unsupported analysis.curve_downsample_method={curve_downsample_method!r}; "
            "only 'even' is supported."
        )

    return {
        "threshold_method": str(threshold_method),
        "precision_target": float(precision_target),
        "probability_threshold_default": float(default_threshold),
        "save_curve_csv": save_curve_csv,
        "max_curve_points": int(max_curve_points),
        "curve_downsample_method": str(curve_downsample_method),
    }


def _read_spatial_block_cv_config(experiment_matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated spatial block CV evaluation settings."""

    validation = _require_mapping(experiment_matrix.get("validation"), "validation")
    spatial = _require_mapping(
        validation.get("spatial_block_cv"),
        "validation.spatial_block_cv",
    )
    if not bool(spatial.get("enable")):
        raise ValueError(
            "validation.spatial_block_cv.enable is false; "
            "10_evaluate_spatial_block_cv.py requires it."
        )

    required = ["n_splits", "block_size_m", "stage1"]
    missing = [key for key in required if key not in spatial]
    if missing:
        raise ValueError(
            "validation.spatial_block_cv is missing required key(s): "
            + ", ".join(missing)
        )

    n_splits = spatial["n_splits"]
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("validation.spatial_block_cv.n_splits must be an integer >= 2.")

    block_size_m = spatial["block_size_m"]
    if isinstance(block_size_m, bool) or not isinstance(block_size_m, (int, float)):
        raise ValueError("validation.spatial_block_cv.block_size_m must be numeric.")
    if float(block_size_m) <= 0:
        raise ValueError("validation.spatial_block_cv.block_size_m must be > 0.")

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
        "n_splits": int(n_splits),
        "block_size_m": float(block_size_m),
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


def _fold_ids(n_splits: int, fold_id_filter: int | None) -> list[int]:
    """Return configured fold ids or one requested fold id."""

    _validate_fold_filter(fold_id_filter, n_splits)
    if fold_id_filter is not None:
        return [int(fold_id_filter)]
    return list(range(int(n_splits)))


def _evaluate_spatial_block_fold(
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    fold_id: int,
    predictions_path: Path,
    train_summary_path: Path,
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one spatial-block CV fold prediction file and write outputs."""

    print(
        f"Evaluating spatial-block CV: {site_id} / {feature_set_id} / "
        f"{model_id} / seed_{seed} / fold_{fold_id}",
        flush=True,
    )
    train_summary_row = validate_spatial_block_train_summary(
        train_summary_path,
        site_id=site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
        fold_id=fold_id,
    )
    y_true, y_score, _ = read_spatial_block_predictions(
        predictions_path,
        site_id=site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
        fold_id=fold_id,
    )

    evaluation = evaluate_scores(y_true, y_score, evaluation_config)
    threshold_info = evaluation["threshold_info"]
    threshold = float(evaluation["threshold"])
    counts = evaluation["counts"]
    binary_metrics = evaluation["binary_metrics"]

    metric_row = {
        "validation_type": "spatial_block_cv",
        "site_id": site_id,
        "feature_set": feature_set_id,
        "model_id": model_id,
        "seed": seed,
        "fold_id": fold_id,
        "threshold_method": evaluation_config["threshold_method"],
        "threshold": threshold,
        "threshold_target": float(evaluation_config["precision_target"]),
        "threshold_target_met": bool(threshold_info.get("target_met", False)),
        "AUC": evaluation["AUC"],
        "AP": evaluation["AP"],
        "Accuracy": binary_metrics["Accuracy"],
        "Precision": binary_metrics["Precision"],
        "Recall": binary_metrics["Recall"],
        "F1": binary_metrics["F1"],
        "Specificity": binary_metrics["Specificity"],
        "Balanced_Accuracy": binary_metrics["Balanced_Accuracy"],
        "MCC": binary_metrics["MCC"],
        "TP": counts["TP"],
        "TN": counts["TN"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "Positive_Count": evaluation["Positive_Count"],
        "Negative_Count": evaluation["Negative_Count"],
        "block_size_m": float(train_summary_row["block_size_m"]),
        "n_splits": int(train_summary_row["n_splits"]),
        "train_block_count": int(train_summary_row["train_block_count"]),
        "test_block_count": int(train_summary_row["test_block_count"]),
        "predictions_path": str(predictions_path),
        "train_summary_path": str(train_summary_path),
        "status": "ok",
        "message": "",
    }

    fold_dir = predictions_path.parent
    write_spatial_block_evaluation_metrics(fold_dir / "evaluation_metrics.csv", metric_row)
    write_spatial_block_confusion_matrix(
        fold_dir / "confusion_matrix.csv",
        site_id=site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
        fold_id=fold_id,
        threshold=threshold,
        counts=counts,
    )

    if bool(evaluation_config["save_curve_csv"]):
        fpr, tpr, roc_thresholds = roc_curve_arrays(y_true, y_score)
        roc_rows = spatial_block_curve_rows(
            fpr,
            tpr,
            roc_thresholds,
            site_id=site_id,
            feature_set_id=feature_set_id,
            model_id=model_id,
            seed=seed,
            fold_id=fold_id,
            x_key="fpr",
            y_key="tpr",
            max_curve_points=int(evaluation_config["max_curve_points"]),
            downsample_method=str(evaluation_config["curve_downsample_method"]),
        )

        pr_thresholds_for_rows = aligned_pr_thresholds(
            evaluation["pr_thresholds"],
            row_count=len(evaluation["pr_precision"]),
        )
        pr_rows = spatial_block_curve_rows(
            evaluation["pr_precision"],
            evaluation["pr_recall"],
            pr_thresholds_for_rows,
            site_id=site_id,
            feature_set_id=feature_set_id,
            model_id=model_id,
            seed=seed,
            fold_id=fold_id,
            x_key="precision",
            y_key="recall",
            max_curve_points=int(evaluation_config["max_curve_points"]),
            downsample_method=str(evaluation_config["curve_downsample_method"]),
        )
        write_spatial_block_curves(fold_dir, roc_rows=roc_rows, pr_rows=pr_rows)

    print(
        f"Evaluated spatial-block CV: {site_id} / {feature_set_id} / "
        f"{model_id} / seed_{seed} / fold_{fold_id}",
        flush=True,
    )
    return metric_row


def process_spatial_block_model_evaluation(
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seeds: Sequence[int],
    fold_ids: Sequence[int],
    spatial_block_root: Path,
    evaluation_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate all selected fold predictions for one spatial-block model."""

    model_dir = spatial_block_root / site_id / feature_set_id / model_id
    train_summary_path = model_dir / "train_summary.csv"
    fold_rows = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"experiment_matrix.seeds must contain only integers, got {seed!r}.")
        for fold_id in fold_ids:
            if isinstance(fold_id, bool) or not isinstance(fold_id, int):
                raise ValueError(f"fold_id values must be integers, got {fold_id!r}.")
            predictions_path = model_dir / f"seed_{seed}" / f"fold_{fold_id}" / "predictions.csv"
            fold_rows.append(
                _evaluate_spatial_block_fold(
                    site_id=site_id,
                    feature_set_id=feature_set_id,
                    model_id=model_id,
                    seed=int(seed),
                    fold_id=int(fold_id),
                    predictions_path=predictions_path,
                    train_summary_path=train_summary_path,
                    evaluation_config=evaluation_config,
                )
            )

    model_summary = summarize_spatial_block_evaluation_rows(fold_rows)
    summary_path = model_dir / "evaluation_summary.csv"
    write_spatial_block_summary(summary_path, model_summary)
    print(f"Wrote spatial-block model evaluation summary: {summary_path}", flush=True)
    return fold_rows


def run_spatial_block_evaluation(
    project_id: str = DEFAULT_PROJECT_ID,
    *,
    site_filter: str | None = None,
    feature_set_filter: str | None = None,
    model_id_filter: str | None = None,
    seed_filter: int | None = None,
    fold_id_filter: int | None = None,
) -> list[Path]:
    """Evaluate all configured spatial block CV prediction files."""

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    experiment_matrix = load_experiment_matrix(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="10")
    evaluation_config = _read_evaluation_config(project_config)
    spatial_config = _read_spatial_block_cv_config(experiment_matrix)

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
        spatial_config["feature_sets"],
        feature_set_filter,
        option_name="--feature-set",
        matrix_name="experiment_matrix.feature_sets",
    )
    model_ids = _filter_requested_value(
        spatial_config["models"],
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
    selected_fold_ids = _fold_ids(int(spatial_config["n_splits"]), fold_id_filter)

    for model_id in model_ids:
        if model_id not in matrix_model_ids:
            raise ValueError(
                f"Spatial block model_id={model_id!r} is not listed in experiment_matrix.models."
            )

    spatial_block_root = output_dirs["spatial_block_cv"]
    all_fold_rows: list[dict[str, Any]] = []
    for site_id in site_ids:
        if not isinstance(site_id, str):
            raise ValueError("experiment_matrix.sites must contain only strings.")
        get_site_config(sites_config, site_id)

        for feature_set_id in feature_set_ids:
            if not isinstance(feature_set_id, str):
                raise ValueError("experiment_matrix.feature_sets must contain only strings.")

            for model_id in model_ids:
                if not isinstance(model_id, str):
                    raise ValueError("spatial_block_cv.stage1.models must contain only strings.")
                all_fold_rows.extend(
                    process_spatial_block_model_evaluation(
                        site_id=site_id,
                        feature_set_id=feature_set_id,
                        model_id=model_id,
                        seeds=seeds,
                        fold_ids=selected_fold_ids,
                        spatial_block_root=spatial_block_root,
                        evaluation_config=evaluation_config,
                    )
                )

    if not all_fold_rows:
        raise ValueError("No spatial-block CV fold metrics were evaluated.")

    fold_metrics_path = spatial_block_root / "spatial_block_cv_fold_metrics.csv"
    summary_path = spatial_block_root / "spatial_block_cv_summary.csv"
    write_csv_rows(fold_metrics_path, all_fold_rows, SPATIAL_BLOCK_SEED_METRIC_COLUMNS)
    write_spatial_block_summary(
        summary_path,
        summarize_spatial_block_evaluation_rows(all_fold_rows),
    )
    return [fold_metrics_path, summary_path]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate configured project spatial block CV predictions.",
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
        help="Optional spatial-block feature-set id filter for smoke tests.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional spatial-block model id filter for smoke tests.",
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
    """Run spatial block CV evaluation from the command line."""

    args = parse_args()
    output_paths = run_spatial_block_evaluation(
        project_id=args.project_id,
        site_filter=args.site,
        feature_set_filter=args.feature_set,
        model_id_filter=args.model_id,
        seed_filter=args.seed,
        fold_id_filter=args.fold_id,
    )
    for output_path in output_paths:
        print(f"Spatial-block evaluation output written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()

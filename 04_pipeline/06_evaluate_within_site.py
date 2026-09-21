#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate within-site random-stratified validation predictions.

This stage reads predictions produced by ``05_train_within_site.py`` and computes
threshold selection, probability metrics, threshold-dependent metrics, curves,
and seed summaries. It does not retrain models, perform cross-site evaluation,
run spatial block CV, predict maps, or create figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    ensure_output_dirs,
    get_site_config,
    load_experiment_matrix,
    load_project_config,
    load_sites_config,
)
from utils.evaluation_core import (
    aligned_pr_thresholds,
    curve_rows,
    evaluate_scores,
    roc_curve_arrays,
)
from utils.evaluation_io import (
    SEED_METRIC_COLUMNS,
    read_predictions,
    summarize_evaluation_rows,
    validate_train_summary,
    write_confusion_matrix,
    write_curves,
    write_csv_rows,
    write_evaluation_metrics,
    write_summary,
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


def _validate_within_site_enabled(experiment_matrix: Mapping[str, Any]) -> None:
    """Ensure the experiment matrix enables within-site validation."""

    validation = _require_mapping(experiment_matrix.get("validation"), "validation")
    within_site = _require_mapping(validation.get("within_site"), "validation.within_site")
    if not bool(within_site.get("enable")):
        raise ValueError("validation.within_site.enable is false; 06_evaluate_within_site.py requires it.")


def _evaluate_seed(
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    predictions_path: Path,
    train_summary_path: Path,
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one seed prediction file and write seed-level outputs."""

    print(f"Evaluating: {site_id} / {feature_set_id} / {model_id} / seed_{seed}", flush=True)
    y_true, y_score, _ = read_predictions(
        predictions_path,
        site_id=site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
    )

    evaluation = evaluate_scores(y_true, y_score, evaluation_config)
    threshold_info = evaluation["threshold_info"]
    threshold = float(evaluation["threshold"])
    counts = evaluation["counts"]
    binary_metrics = evaluation["binary_metrics"]

    metric_row = {
        "validation_type": "within_site",
        "site_id": site_id,
        "feature_set": feature_set_id,
        "model_id": model_id,
        "seed": seed,
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
        "predictions_path": str(predictions_path),
        "train_summary_path": str(train_summary_path),
        "status": "ok",
        "message": "",
    }

    seed_dir = predictions_path.parent
    write_evaluation_metrics(seed_dir / "evaluation_metrics.csv", metric_row)
    write_confusion_matrix(
        seed_dir / "confusion_matrix.csv",
        site_id=site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
        threshold=threshold,
        counts=counts,
    )

    if bool(evaluation_config["save_curve_csv"]):
        fpr, tpr, roc_thresholds = roc_curve_arrays(y_true, y_score)
        roc_rows = curve_rows(
            fpr,
            tpr,
            roc_thresholds,
            site_id=site_id,
            feature_set_id=feature_set_id,
            model_id=model_id,
            seed=seed,
            x_key="fpr",
            y_key="tpr",
            max_curve_points=int(evaluation_config["max_curve_points"]),
            downsample_method=str(evaluation_config["curve_downsample_method"]),
        )

        pr_thresholds_for_rows = aligned_pr_thresholds(
            evaluation["pr_thresholds"],
            row_count=len(evaluation["pr_precision"]),
        )
        pr_rows = curve_rows(
            evaluation["pr_precision"],
            evaluation["pr_recall"],
            pr_thresholds_for_rows,
            site_id=site_id,
            feature_set_id=feature_set_id,
            model_id=model_id,
            seed=seed,
            x_key="precision",
            y_key="recall",
            max_curve_points=int(evaluation_config["max_curve_points"]),
            downsample_method=str(evaluation_config["curve_downsample_method"]),
        )
        write_curves(seed_dir, roc_rows=roc_rows, pr_rows=pr_rows)

    print(f"Evaluated: {site_id} / {feature_set_id} / {model_id} / seed_{seed}", flush=True)
    return metric_row


def process_model_evaluation(
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seeds: Sequence[int],
    within_site_root: Path,
    evaluation_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate all seed predictions for one model and write model summary."""

    model_dir = within_site_root / site_id / feature_set_id / model_id
    train_summary_path = model_dir / "train_summary.csv"
    validate_train_summary(
        train_summary_path,
        site_id=site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
    )

    seed_rows = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"experiment_matrix.seeds must contain only integers, got {seed!r}.")
        predictions_path = model_dir / f"seed_{seed}" / "predictions.csv"
        seed_rows.append(
            _evaluate_seed(
                site_id=site_id,
                feature_set_id=feature_set_id,
                model_id=model_id,
                seed=int(seed),
                predictions_path=predictions_path,
                train_summary_path=train_summary_path,
                evaluation_config=evaluation_config,
            )
        )

    model_summary = summarize_evaluation_rows(seed_rows)
    summary_path = model_dir / "evaluation_summary.csv"
    write_summary(summary_path, model_summary)
    print(f"Wrote model evaluation summary: {summary_path}", flush=True)
    return seed_rows


def run_evaluation(project_id: str = DEFAULT_PROJECT_ID) -> list[Path]:
    """Evaluate all configured within-site prediction files."""

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    experiment_matrix = load_experiment_matrix(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="06")
    _validate_within_site_enabled(experiment_matrix)
    evaluation_config = _read_evaluation_config(project_config)

    site_ids = _require_list(experiment_matrix.get("sites"), "experiment_matrix.sites")
    feature_set_ids = _require_list(
        experiment_matrix.get("feature_sets"),
        "experiment_matrix.feature_sets",
    )
    model_ids = _require_list(experiment_matrix.get("models"), "experiment_matrix.models")
    seeds = _require_list(experiment_matrix.get("seeds"), "experiment_matrix.seeds")

    all_seed_rows: list[dict[str, Any]] = []
    for site_id in site_ids:
        if not isinstance(site_id, str):
            raise ValueError("experiment_matrix.sites must contain only strings.")
        get_site_config(sites_config, site_id)

        for feature_set_id in feature_set_ids:
            if not isinstance(feature_set_id, str):
                raise ValueError("experiment_matrix.feature_sets must contain only strings.")
            for model_id in model_ids:
                if not isinstance(model_id, str):
                    raise ValueError("experiment_matrix.models must contain only strings.")
                all_seed_rows.extend(
                    process_model_evaluation(
                        site_id=site_id,
                        feature_set_id=feature_set_id,
                        model_id=model_id,
                        seeds=seeds,
                        within_site_root=output_dirs["within_site"],
                        evaluation_config=evaluation_config,
                    )
                )

    if not all_seed_rows:
        raise ValueError("No seed metrics were evaluated.")

    within_site_root = output_dirs["within_site"]
    seed_metrics_path = within_site_root / "within_site_seed_metrics.csv"
    summary_path = within_site_root / "within_site_summary.csv"
    write_csv_rows(seed_metrics_path, all_seed_rows, SEED_METRIC_COLUMNS)
    write_summary(summary_path, summarize_evaluation_rows(all_seed_rows))
    return [seed_metrics_path, summary_path]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate configured project within-site validation predictions.",
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=f"Project id to process. Default: {DEFAULT_PROJECT_ID}",
    )
    return parser.parse_args()


def main() -> None:
    """Run within-site evaluation from the command line."""

    args = parse_args()
    output_paths = run_evaluation(project_id=args.project_id)
    for output_path in output_paths:
        print(f"Evaluation output written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()

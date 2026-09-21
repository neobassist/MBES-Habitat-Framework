#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate configured project cross-site transfer validation predictions.

This stage reads predictions produced by ``07_train_cross_site.py`` and
computes threshold selection, probability metrics, threshold-dependent metrics,
optional curves, and seed/model/root summaries. It does not retrain models,
run spatial block CV, predict maps, or create figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    configured_cross_site_pairs,
    ensure_output_dirs,
    get_site_config,
    load_experiment_matrix,
    load_project_config,
    load_sites_config,
)
from utils.evaluation_core import (
    aligned_pr_thresholds,
    cross_site_curve_rows,
    evaluate_scores,
    roc_curve_arrays,
)
from utils.evaluation_io import (
    CROSS_SITE_SEED_METRIC_COLUMNS,
    read_cross_site_predictions,
    summarize_cross_site_evaluation_rows,
    validate_cross_site_train_summary,
    write_cross_site_confusion_matrix,
    write_cross_site_curves,
    write_cross_site_evaluation_metrics,
    write_cross_site_summary,
    write_csv_rows,
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


def _pair_id(train_site_id: str, test_site_id: str) -> str:
    """Return the cross-site output directory id for one ordered pair."""

    return f"train_{train_site_id}_test_{test_site_id}"


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


def _validate_cross_site_enabled(experiment_matrix: Mapping[str, Any]) -> None:
    """Ensure the experiment matrix enables cross-site validation."""

    validation = _require_mapping(experiment_matrix.get("validation"), "validation")
    cross_site = _require_mapping(validation.get("cross_site"), "validation.cross_site")
    if not bool(cross_site.get("enable")):
        raise ValueError("validation.cross_site.enable is false; 08_evaluate_cross_site.py requires it.")


def _evaluate_cross_site_seed(
    *,
    train_site_id: str,
    test_site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    predictions_path: Path,
    train_summary_path: Path,
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one cross-site seed prediction file and write seed outputs."""

    print(
        f"Evaluating cross-site: {train_site_id} -> {test_site_id} / "
        f"{feature_set_id} / {model_id} / seed_{seed}",
        flush=True,
    )
    y_true, y_score, _ = read_cross_site_predictions(
        predictions_path,
        train_site_id=train_site_id,
        test_site_id=test_site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
    )

    evaluation = evaluate_scores(y_true, y_score, evaluation_config)
    threshold_info = evaluation["threshold_info"]
    threshold = float(evaluation["threshold"])
    counts = evaluation["counts"]
    binary_metrics = evaluation["binary_metrics"]
    site_id = test_site_id

    metric_row = {
        "validation_type": "cross_site",
        "train_site_id": train_site_id,
        "test_site_id": test_site_id,
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
    write_cross_site_evaluation_metrics(seed_dir / "evaluation_metrics.csv", metric_row)
    write_cross_site_confusion_matrix(
        seed_dir / "confusion_matrix.csv",
        train_site_id=train_site_id,
        test_site_id=test_site_id,
        site_id=site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
        seed=seed,
        threshold=threshold,
        counts=counts,
    )

    if bool(evaluation_config["save_curve_csv"]):
        fpr, tpr, roc_thresholds = roc_curve_arrays(y_true, y_score)
        roc_rows = cross_site_curve_rows(
            fpr,
            tpr,
            roc_thresholds,
            train_site_id=train_site_id,
            test_site_id=test_site_id,
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
        pr_rows = cross_site_curve_rows(
            evaluation["pr_precision"],
            evaluation["pr_recall"],
            pr_thresholds_for_rows,
            train_site_id=train_site_id,
            test_site_id=test_site_id,
            site_id=site_id,
            feature_set_id=feature_set_id,
            model_id=model_id,
            seed=seed,
            x_key="precision",
            y_key="recall",
            max_curve_points=int(evaluation_config["max_curve_points"]),
            downsample_method=str(evaluation_config["curve_downsample_method"]),
        )
        write_cross_site_curves(seed_dir, roc_rows=roc_rows, pr_rows=pr_rows)

    print(
        f"Evaluated cross-site: {train_site_id} -> {test_site_id} / "
        f"{feature_set_id} / {model_id} / seed_{seed}",
        flush=True,
    )
    return metric_row


def process_cross_site_model_evaluation(
    *,
    train_site_id: str,
    test_site_id: str,
    feature_set_id: str,
    model_id: str,
    seeds: Sequence[int],
    cross_site_root: Path,
    evaluation_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate all seed predictions for one cross-site model run."""

    model_dir = cross_site_root / _pair_id(train_site_id, test_site_id) / feature_set_id / model_id
    train_summary_path = model_dir / "train_summary.csv"
    validate_cross_site_train_summary(
        train_summary_path,
        train_site_id=train_site_id,
        test_site_id=test_site_id,
        feature_set_id=feature_set_id,
        model_id=model_id,
    )

    seed_rows = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"experiment_matrix.seeds must contain only integers, got {seed!r}.")
        predictions_path = model_dir / f"seed_{seed}" / "predictions.csv"
        seed_rows.append(
            _evaluate_cross_site_seed(
                train_site_id=train_site_id,
                test_site_id=test_site_id,
                feature_set_id=feature_set_id,
                model_id=model_id,
                seed=int(seed),
                predictions_path=predictions_path,
                train_summary_path=train_summary_path,
                evaluation_config=evaluation_config,
            )
        )

    model_summary = summarize_cross_site_evaluation_rows(seed_rows)
    summary_path = model_dir / "evaluation_summary.csv"
    write_cross_site_summary(summary_path, model_summary)
    print(f"Wrote cross-site model evaluation summary: {summary_path}", flush=True)
    return seed_rows


def run_cross_site_evaluation(
    project_id: str = DEFAULT_PROJECT_ID,
    *,
    train_site: str | None = None,
    test_site: str | None = None,
    feature_set: str | None = None,
    model_id_filter: str | None = None,
    seed_filter: int | None = None,
) -> list[Path]:
    """Evaluate all configured cross-site prediction files."""

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    experiment_matrix = load_experiment_matrix(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="08")
    _validate_cross_site_enabled(experiment_matrix)
    evaluation_config = _read_evaluation_config(project_config)

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

    cross_site_root = output_dirs["cross_site"]
    all_seed_rows: list[dict[str, Any]] = []
    for train_site_id, test_site_id in cross_site_pairs:
        get_site_config(sites_config, train_site_id)
        get_site_config(sites_config, test_site_id)

        for feature_set_id in feature_set_ids:
            if not isinstance(feature_set_id, str):
                raise ValueError("experiment_matrix.feature_sets must contain only strings.")

            for model_id in model_ids:
                if not isinstance(model_id, str):
                    raise ValueError("experiment_matrix.models must contain only strings.")
                all_seed_rows.extend(
                    process_cross_site_model_evaluation(
                        train_site_id=train_site_id,
                        test_site_id=test_site_id,
                        feature_set_id=feature_set_id,
                        model_id=model_id,
                        seeds=seeds,
                        cross_site_root=cross_site_root,
                        evaluation_config=evaluation_config,
                    )
                )

    if not all_seed_rows:
        raise ValueError("No cross-site seed metrics were evaluated.")

    seed_metrics_path = cross_site_root / "cross_site_seed_metrics.csv"
    summary_path = cross_site_root / "cross_site_summary.csv"
    write_csv_rows(seed_metrics_path, all_seed_rows, CROSS_SITE_SEED_METRIC_COLUMNS)
    write_cross_site_summary(summary_path, summarize_cross_site_evaluation_rows(all_seed_rows))
    return [seed_metrics_path, summary_path]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate configured project cross-site transfer validation predictions.",
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
    """Run cross-site evaluation from the command line."""

    args = parse_args()
    output_paths = run_cross_site_evaluation(
        project_id=args.project_id,
        train_site=args.train_site,
        test_site=args.test_site,
        feature_set=args.feature_set,
        model_id_filter=args.model_id,
        seed_filter=args.seed,
    )
    for output_path in output_paths:
        print(f"Cross-site evaluation output written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()

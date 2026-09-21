#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation CSV I/O helpers shared by validation stages."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.metrics import summarize_group_metrics


REQUIRED_PREDICTION_COLUMNS = [
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "split",
    "sample_index",
    "x",
    "y",
    "y_true",
    "y_score",
]
CROSS_SITE_REQUIRED_PREDICTION_COLUMNS = [
    "validation_type",
    "train_site_id",
    "test_site_id",
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "split",
    "sample_index",
    "x",
    "y",
    "y_true",
    "y_score",
]
SPATIAL_BLOCK_REQUIRED_PREDICTION_COLUMNS = [
    "validation_type",
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "fold_id",
    "block_id",
    "split",
    "sample_index",
    "x",
    "y",
    "y_true",
    "y_score",
]
SEED_METRIC_COLUMNS = [
    "validation_type",
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "threshold_method",
    "threshold",
    "threshold_target",
    "threshold_target_met",
    "AUC",
    "AP",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Specificity",
    "Balanced_Accuracy",
    "MCC",
    "TP",
    "TN",
    "FP",
    "FN",
    "Positive_Count",
    "Negative_Count",
    "predictions_path",
    "train_summary_path",
    "status",
    "message",
]
CROSS_SITE_SEED_METRIC_COLUMNS = [
    "validation_type",
    "train_site_id",
    "test_site_id",
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "threshold_method",
    "threshold",
    "threshold_target",
    "threshold_target_met",
    "AUC",
    "AP",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Specificity",
    "Balanced_Accuracy",
    "MCC",
    "TP",
    "TN",
    "FP",
    "FN",
    "Positive_Count",
    "Negative_Count",
    "predictions_path",
    "train_summary_path",
    "status",
    "message",
]
SPATIAL_BLOCK_SEED_METRIC_COLUMNS = [
    "validation_type",
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "fold_id",
    "threshold_method",
    "threshold",
    "threshold_target",
    "threshold_target_met",
    "AUC",
    "AP",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Specificity",
    "Balanced_Accuracy",
    "MCC",
    "TP",
    "TN",
    "FP",
    "FN",
    "Positive_Count",
    "Negative_Count",
    "block_size_m",
    "n_splits",
    "train_block_count",
    "test_block_count",
    "predictions_path",
    "train_summary_path",
    "status",
    "message",
]
CONFUSION_COLUMNS = [
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "threshold",
    "TP",
    "TN",
    "FP",
    "FN",
]
CROSS_SITE_CONFUSION_COLUMNS = [
    "validation_type",
    "train_site_id",
    "test_site_id",
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "threshold",
    "TP",
    "TN",
    "FP",
    "FN",
]
SPATIAL_BLOCK_CONFUSION_COLUMNS = [
    "validation_type",
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "fold_id",
    "threshold",
    "TP",
    "TN",
    "FP",
    "FN",
]
ROC_CURVE_COLUMNS = ["site_id", "feature_set", "model_id", "seed", "fpr", "tpr", "threshold"]
CROSS_SITE_ROC_CURVE_COLUMNS = [
    "validation_type",
    "train_site_id",
    "test_site_id",
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "fpr",
    "tpr",
    "threshold",
]
SPATIAL_BLOCK_ROC_CURVE_COLUMNS = [
    "validation_type",
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "fold_id",
    "fpr",
    "tpr",
    "threshold",
]
PR_CURVE_COLUMNS = [
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "precision",
    "recall",
    "threshold",
]
CROSS_SITE_PR_CURVE_COLUMNS = [
    "validation_type",
    "train_site_id",
    "test_site_id",
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "precision",
    "recall",
    "threshold",
]
SPATIAL_BLOCK_PR_CURVE_COLUMNS = [
    "validation_type",
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "fold_id",
    "precision",
    "recall",
    "threshold",
]
SUMMARY_GROUP_COLUMNS = ["validation_type", "site_id", "feature_set", "model_id"]
CROSS_SITE_SUMMARY_GROUP_COLUMNS = [
    "validation_type",
    "train_site_id",
    "test_site_id",
    "site_id",
    "feature_set",
    "model_id",
]
SPATIAL_BLOCK_SUMMARY_GROUP_COLUMNS = [
    "validation_type",
    "site_id",
    "feature_set",
    "model_id",
]
SUMMARY_METRIC_COLUMNS = [
    "threshold",
    "AUC",
    "AP",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Specificity",
    "Balanced_Accuracy",
    "MCC",
    "TP",
    "TN",
    "FP",
    "FN",
    "Positive_Count",
    "Negative_Count",
]
SUMMARY_OUTPUT_COLUMNS = SUMMARY_GROUP_COLUMNS + ["n"] + [
    f"{metric}_{suffix}"
    for metric in [
        "Threshold",
        "AUC",
        "AP",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "Balanced_Accuracy",
        "MCC",
        "TP",
        "TN",
        "FP",
        "FN",
        "Positive_Count",
        "Negative_Count",
    ]
    for suffix in ["mean", "sd"]
]
SPATIAL_BLOCK_SUMMARY_COLUMNS = SPATIAL_BLOCK_SUMMARY_GROUP_COLUMNS + ["n"] + [
    f"{metric}_{suffix}"
    for metric in [
        "Threshold",
        "AUC",
        "AP",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "Balanced_Accuracy",
        "MCC",
        "TP",
        "TN",
        "FP",
        "FN",
        "Positive_Count",
        "Negative_Count",
    ]
    for suffix in ["mean", "sd"]
]
CROSS_SITE_SUMMARY_OUTPUT_COLUMNS = CROSS_SITE_SUMMARY_GROUP_COLUMNS + ["n"] + [
    f"{metric}_{suffix}"
    for metric in [
        "Threshold",
        "AUC",
        "AP",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "Balanced_Accuracy",
        "MCC",
        "TP",
        "TN",
        "FP",
        "FN",
        "Positive_Count",
        "Negative_Count",
    ]
    for suffix in ["mean", "sd"]
]


def _require_pandas() -> Any:
    """Import pandas with a clear dependency error."""

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "pandas is required for evaluation CSV I/O. Install pandas in the "
            "mbes_seaweed environment before running this stage."
        ) from exc
    return pd


def read_nonempty_csv(path: Path, label: str) -> Any:
    """Read a non-empty CSV with pandas."""

    pd = _require_pandas()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")

    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} is empty: {path}") from exc
    if data.empty:
        raise ValueError(f"{label} contains zero rows: {path}")
    return data


def validate_train_summary(
    train_summary_path: Path,
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
) -> None:
    """Read and validate the presence of a train summary."""

    summary = read_nonempty_csv(train_summary_path, "train_summary.csv")
    for column, expected in [
        ("site_id", site_id),
        ("feature_set", feature_set_id),
        ("model_id", model_id),
    ]:
        if column in summary.columns and not (summary[column].astype(str) == str(expected)).all():
            raise ValueError(
                f"train_summary.csv has {column} values that do not match {expected!r}: "
                f"{train_summary_path}"
            )


def validate_cross_site_train_summary(
    train_summary_path: Path,
    *,
    train_site_id: str,
    test_site_id: str,
    feature_set_id: str,
    model_id: str,
) -> None:
    """Read and validate the presence of a cross-site train summary."""

    summary = read_nonempty_csv(train_summary_path, "train_summary.csv")
    for column, expected in [
        ("validation_type", "cross_site"),
        ("train_site_id", train_site_id),
        ("test_site_id", test_site_id),
        ("feature_set", feature_set_id),
        ("model_id", model_id),
    ]:
        if column in summary.columns and not (summary[column].astype(str) == str(expected)).all():
            raise ValueError(
                f"train_summary.csv has {column} values that do not match {expected!r}: "
                f"{train_summary_path}"
            )


def validate_spatial_block_train_summary(
    train_summary_path: Path,
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    fold_id: int,
) -> dict[str, Any]:
    """Read and validate one spatial-block train summary row."""

    summary = read_nonempty_csv(train_summary_path, "train_summary.csv")
    required_columns = [
        "validation_type",
        "site_id",
        "feature_set",
        "model_id",
        "seed",
        "fold_id",
        "block_size_m",
        "n_splits",
        "train_block_count",
        "test_block_count",
    ]
    missing = [column for column in required_columns if column not in summary.columns]
    if missing:
        raise ValueError(
            f"train_summary.csv is missing required column(s): {', '.join(missing)}; "
            f"path={train_summary_path}"
        )

    for column, expected in [
        ("validation_type", "spatial_block_cv"),
        ("site_id", site_id),
        ("feature_set", feature_set_id),
        ("model_id", model_id),
    ]:
        if not (summary[column].astype(str) == str(expected)).all():
            raise ValueError(
                f"train_summary.csv has {column} values that do not match {expected!r}: "
                f"{train_summary_path}"
            )

    matching = summary[
        (summary["seed"].astype(int) == int(seed))
        & (summary["fold_id"].astype(int) == int(fold_id))
    ]
    if matching.empty:
        raise ValueError(
            f"train_summary.csv has no row for seed={seed}, fold_id={fold_id}: "
            f"{train_summary_path}"
        )
    if matching.shape[0] != 1:
        raise ValueError(
            f"train_summary.csv has multiple rows for seed={seed}, fold_id={fold_id}: "
            f"{train_summary_path}"
        )
    return dict(matching.iloc[0].to_dict())


def read_predictions(
    predictions_path: Path,
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Read and validate one seed predictions CSV."""

    predictions = read_nonempty_csv(predictions_path, "predictions.csv")
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(
            f"predictions.csv is missing required column(s): {', '.join(missing)}; "
            f"path={predictions_path}"
        )

    if not (predictions["split"].astype(str) == "test").all():
        raise ValueError(f"predictions.csv contains split values other than 'test': {predictions_path}")
    for column, expected in [
        ("site_id", site_id),
        ("feature_set", feature_set_id),
        ("model_id", model_id),
    ]:
        if not (predictions[column].astype(str) == str(expected)).all():
            raise ValueError(
                f"predictions.csv has {column} values that do not match {expected!r}: "
                f"{predictions_path}"
            )

    seeds = predictions["seed"].to_numpy()
    if not np.all(seeds == seed):
        raise ValueError(
            f"predictions.csv seed values do not match seed={seed}: {predictions_path}"
        )

    y_true_float = predictions["y_true"].to_numpy(dtype="float64", copy=False)
    if not np.all(np.isfinite(y_true_float)):
        raise ValueError(f"y_true contains NaN or infinite values: {predictions_path}")
    if not np.all(np.isin(y_true_float, [0.0, 1.0])):
        unique_values = sorted(set(y_true_float.tolist()))
        raise ValueError(
            f"y_true must contain only 0/1 values in {predictions_path}; found {unique_values}"
        )
    y_true = y_true_float.astype(int)

    if not {0, 1}.issubset(set(y_true.tolist())):
        counts = {
            "negative": int(np.count_nonzero(y_true == 0)),
            "positive": int(np.count_nonzero(y_true == 1)),
        }
        raise ValueError(f"test set is missing a class in {predictions_path}: {counts}")

    y_score = predictions["y_score"].to_numpy(dtype="float64", copy=False)
    if not np.all(np.isfinite(y_score)):
        raise ValueError(f"y_score contains NaN or infinite values: {predictions_path}")
    if not np.all((0.0 <= y_score) & (y_score <= 1.0)):
        raise ValueError(f"y_score contains values outside [0, 1]: {predictions_path}")

    return y_true, y_score, predictions


def read_cross_site_predictions(
    predictions_path: Path,
    *,
    train_site_id: str,
    test_site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Read and validate one cross-site seed predictions CSV."""

    predictions = read_nonempty_csv(predictions_path, "predictions.csv")
    missing = [
        column
        for column in CROSS_SITE_REQUIRED_PREDICTION_COLUMNS
        if column not in predictions.columns
    ]
    if missing:
        raise ValueError(
            f"predictions.csv is missing required column(s): {', '.join(missing)}; "
            f"path={predictions_path}"
        )

    for column, expected in [
        ("validation_type", "cross_site"),
        ("train_site_id", train_site_id),
        ("test_site_id", test_site_id),
        ("site_id", test_site_id),
        ("feature_set", feature_set_id),
        ("model_id", model_id),
        ("split", "test"),
    ]:
        if not (predictions[column].astype(str) == str(expected)).all():
            raise ValueError(
                f"predictions.csv has {column} values that do not match {expected!r}: "
                f"{predictions_path}"
            )

    seeds = predictions["seed"].to_numpy()
    if not np.all(seeds == seed):
        raise ValueError(
            f"predictions.csv seed values do not match seed={seed}: {predictions_path}"
        )

    y_true_float = predictions["y_true"].to_numpy(dtype="float64", copy=False)
    if not np.all(np.isfinite(y_true_float)):
        raise ValueError(f"y_true contains NaN or infinite values: {predictions_path}")
    if not np.all(np.isin(y_true_float, [0.0, 1.0])):
        unique_values = sorted(set(y_true_float.tolist()))
        raise ValueError(
            f"y_true must contain only 0/1 values in {predictions_path}; found {unique_values}"
        )
    y_true = y_true_float.astype(int)

    if not {0, 1}.issubset(set(y_true.tolist())):
        counts = {
            "negative": int(np.count_nonzero(y_true == 0)),
            "positive": int(np.count_nonzero(y_true == 1)),
        }
        raise ValueError(f"test set is missing a class in {predictions_path}: {counts}")

    y_score = predictions["y_score"].to_numpy(dtype="float64", copy=False)
    if not np.all(np.isfinite(y_score)):
        raise ValueError(f"y_score contains NaN or infinite values: {predictions_path}")
    if not np.all((0.0 <= y_score) & (y_score <= 1.0)):
        raise ValueError(f"y_score contains values outside [0, 1]: {predictions_path}")

    return y_true, y_score, predictions


def read_spatial_block_predictions(
    predictions_path: Path,
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Read and validate one spatial-block CV fold predictions CSV."""

    predictions = read_nonempty_csv(predictions_path, "predictions.csv")
    missing = [
        column
        for column in SPATIAL_BLOCK_REQUIRED_PREDICTION_COLUMNS
        if column not in predictions.columns
    ]
    if missing:
        raise ValueError(
            f"predictions.csv is missing required column(s): {', '.join(missing)}; "
            f"path={predictions_path}"
        )

    for column, expected in [
        ("validation_type", "spatial_block_cv"),
        ("site_id", site_id),
        ("feature_set", feature_set_id),
        ("model_id", model_id),
        ("split", "test"),
    ]:
        if not (predictions[column].astype(str) == str(expected)).all():
            raise ValueError(
                f"predictions.csv has {column} values that do not match {expected!r}: "
                f"{predictions_path}"
            )

    seeds = predictions["seed"].to_numpy()
    if not np.all(seeds == int(seed)):
        raise ValueError(
            f"predictions.csv seed values do not match seed={seed}: {predictions_path}"
        )

    fold_ids = predictions["fold_id"].to_numpy()
    if not np.all(fold_ids == int(fold_id)):
        raise ValueError(
            f"predictions.csv fold_id values do not match fold_id={fold_id}: "
            f"{predictions_path}"
        )
    if predictions["block_id"].isna().any():
        raise ValueError(f"predictions.csv contains missing block_id values: {predictions_path}")

    y_true_float = predictions["y_true"].to_numpy(dtype="float64", copy=False)
    if not np.all(np.isfinite(y_true_float)):
        raise ValueError(f"y_true contains NaN or infinite values: {predictions_path}")
    if not np.all(np.isin(y_true_float, [0.0, 1.0])):
        unique_values = sorted(set(y_true_float.tolist()))
        raise ValueError(
            f"y_true must contain only 0/1 values in {predictions_path}; found {unique_values}"
        )
    y_true = y_true_float.astype(int)

    if not {0, 1}.issubset(set(y_true.tolist())):
        counts = {
            "negative": int(np.count_nonzero(y_true == 0)),
            "positive": int(np.count_nonzero(y_true == 1)),
        }
        raise ValueError(f"test fold is missing a class in {predictions_path}: {counts}")

    y_score = predictions["y_score"].to_numpy(dtype="float64", copy=False)
    if not np.all(np.isfinite(y_score)):
        raise ValueError(f"y_score contains NaN or infinite values: {predictions_path}")
    if not np.all((0.0 <= y_score) & (y_score <= 1.0)):
        raise ValueError(f"y_score contains values outside [0, 1]: {predictions_path}")

    return y_true, y_score, predictions


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Write rows to CSV with explicit columns."""

    if not rows:
        raise ValueError(f"Cannot write empty CSV rows: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_metrics(path: Path, metric_row: Mapping[str, Any]) -> None:
    """Write one seed evaluation metrics CSV."""

    write_csv_rows(path, [metric_row], SEED_METRIC_COLUMNS)


def write_cross_site_evaluation_metrics(path: Path, metric_row: Mapping[str, Any]) -> None:
    """Write one cross-site seed evaluation metrics CSV."""

    write_csv_rows(path, [metric_row], CROSS_SITE_SEED_METRIC_COLUMNS)


def write_spatial_block_evaluation_metrics(path: Path, metric_row: Mapping[str, Any]) -> None:
    """Write one spatial-block CV fold evaluation metrics CSV."""

    write_csv_rows(path, [metric_row], SPATIAL_BLOCK_SEED_METRIC_COLUMNS)


def write_confusion_matrix(
    path: Path,
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    threshold: float,
    counts: Mapping[str, int],
) -> None:
    """Write one compact confusion matrix CSV."""

    write_csv_rows(
        path,
        [
            {
                "site_id": site_id,
                "feature_set": feature_set_id,
                "model_id": model_id,
                "seed": seed,
                "threshold": threshold,
                "TP": counts["TP"],
                "TN": counts["TN"],
                "FP": counts["FP"],
                "FN": counts["FN"],
            }
        ],
        CONFUSION_COLUMNS,
    )


def write_cross_site_confusion_matrix(
    path: Path,
    *,
    train_site_id: str,
    test_site_id: str,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    threshold: float,
    counts: Mapping[str, int],
) -> None:
    """Write one compact cross-site confusion matrix CSV."""

    write_csv_rows(
        path,
        [
            {
                "validation_type": "cross_site",
                "train_site_id": train_site_id,
                "test_site_id": test_site_id,
                "site_id": site_id,
                "feature_set": feature_set_id,
                "model_id": model_id,
                "seed": seed,
                "threshold": threshold,
                "TP": counts["TP"],
                "TN": counts["TN"],
                "FP": counts["FP"],
                "FN": counts["FN"],
            }
        ],
        CROSS_SITE_CONFUSION_COLUMNS,
    )


def write_spatial_block_confusion_matrix(
    path: Path,
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    fold_id: int,
    threshold: float,
    counts: Mapping[str, int],
) -> None:
    """Write one compact spatial-block CV confusion matrix CSV."""

    write_csv_rows(
        path,
        [
            {
                "validation_type": "spatial_block_cv",
                "site_id": site_id,
                "feature_set": feature_set_id,
                "model_id": model_id,
                "seed": seed,
                "fold_id": fold_id,
                "threshold": threshold,
                "TP": counts["TP"],
                "TN": counts["TN"],
                "FP": counts["FP"],
                "FN": counts["FN"],
            }
        ],
        SPATIAL_BLOCK_CONFUSION_COLUMNS,
    )


def write_curves(
    seed_dir: Path,
    *,
    roc_rows: Sequence[Mapping[str, Any]],
    pr_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write ROC and PR curve CSV files."""

    write_csv_rows(seed_dir / "roc_curve.csv", roc_rows, ROC_CURVE_COLUMNS)
    write_csv_rows(seed_dir / "pr_curve.csv", pr_rows, PR_CURVE_COLUMNS)


def write_cross_site_curves(
    seed_dir: Path,
    *,
    roc_rows: Sequence[Mapping[str, Any]],
    pr_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write cross-site ROC and PR curve CSV files."""

    write_csv_rows(seed_dir / "roc_curve.csv", roc_rows, CROSS_SITE_ROC_CURVE_COLUMNS)
    write_csv_rows(seed_dir / "pr_curve.csv", pr_rows, CROSS_SITE_PR_CURVE_COLUMNS)


def write_spatial_block_curves(
    fold_dir: Path,
    *,
    roc_rows: Sequence[Mapping[str, Any]],
    pr_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write spatial-block CV ROC and PR curve CSV files."""

    write_csv_rows(fold_dir / "roc_curve.csv", roc_rows, SPATIAL_BLOCK_ROC_CURVE_COLUMNS)
    write_csv_rows(fold_dir / "pr_curve.csv", pr_rows, SPATIAL_BLOCK_PR_CURVE_COLUMNS)


def summarize_evaluation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize seed metrics and normalize Threshold column casing."""

    if not rows:
        raise ValueError("summary target rows must not be empty.")
    summaries = summarize_group_metrics(
        rows,
        group_cols=SUMMARY_GROUP_COLUMNS,
        metric_cols=SUMMARY_METRIC_COLUMNS,
    )
    normalized = []
    for summary in summaries:
        row = dict(summary)
        if "threshold_mean" in row:
            row["Threshold_mean"] = row.pop("threshold_mean")
        if "threshold_sd" in row:
            row["Threshold_sd"] = row.pop("threshold_sd")
        normalized.append(row)
    return normalized


def summarize_cross_site_evaluation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize cross-site seed metrics and normalize Threshold column casing."""

    if not rows:
        raise ValueError("summary target rows must not be empty.")
    summaries = summarize_group_metrics(
        rows,
        group_cols=CROSS_SITE_SUMMARY_GROUP_COLUMNS,
        metric_cols=SUMMARY_METRIC_COLUMNS,
    )
    normalized = []
    for summary in summaries:
        row = dict(summary)
        if "threshold_mean" in row:
            row["Threshold_mean"] = row.pop("threshold_mean")
        if "threshold_sd" in row:
            row["Threshold_sd"] = row.pop("threshold_sd")
        normalized.append(row)
    return normalized


def summarize_spatial_block_evaluation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize spatial-block CV fold metrics and normalize Threshold casing."""

    if not rows:
        raise ValueError("summary target rows must not be empty.")
    summaries = summarize_group_metrics(
        rows,
        group_cols=SPATIAL_BLOCK_SUMMARY_GROUP_COLUMNS,
        metric_cols=SUMMARY_METRIC_COLUMNS,
    )
    normalized = []
    for summary in summaries:
        row = dict(summary)
        if "threshold_mean" in row:
            row["Threshold_mean"] = row.pop("threshold_mean")
        if "threshold_sd" in row:
            row["Threshold_sd"] = row.pop("threshold_sd")
        normalized.append(row)
    return normalized


def write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write summary rows with a stable column order."""

    write_csv_rows(path, rows, SUMMARY_OUTPUT_COLUMNS)


def write_cross_site_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write cross-site summary rows with a stable column order."""

    write_csv_rows(path, rows, CROSS_SITE_SUMMARY_OUTPUT_COLUMNS)


def write_spatial_block_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write spatial-block CV summary rows with a stable column order."""

    write_csv_rows(path, rows, SPATIAL_BLOCK_SUMMARY_COLUMNS)

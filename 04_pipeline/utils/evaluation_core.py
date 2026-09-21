#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vectorized evaluation calculations shared by evaluation stages."""

from __future__ import annotations

from math import sqrt
from typing import Any, Mapping, Sequence

import numpy as np


def _require_sklearn_metrics() -> dict[str, Any]:
    """Import sklearn metric functions with a clear dependency error."""

    try:
        from sklearn.metrics import (  # type: ignore[import-not-found]
            average_precision_score,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )
    except ModuleNotFoundError as exc:
        raise ImportError(
            "scikit-learn is required for evaluation metric calculation. "
            "Install scikit-learn in the mbes_seaweed environment before "
            "running this stage."
        ) from exc
    return {
        "average_precision_score": average_precision_score,
        "precision_recall_curve": precision_recall_curve,
        "roc_auc_score": roc_auc_score,
        "roc_curve": roc_curve,
    }


def mcc_from_counts(counts: Mapping[str, int]) -> float:
    """Compute Matthews correlation coefficient from confusion counts."""

    tp = int(counts["TP"])
    tn = int(counts["TN"])
    fp = int(counts["FP"])
    fn = int(counts["FN"])
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denominator)


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Divide and return 0.0 when denominator is zero."""

    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def confusion_counts_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Return confusion counts using ``y_score >= threshold``."""

    y_pred = y_score >= float(threshold)
    labels = y_true.astype(bool, copy=False)
    tp = int(np.count_nonzero(labels & y_pred))
    tn = int(np.count_nonzero((~labels) & (~y_pred)))
    fp = int(np.count_nonzero((~labels) & y_pred))
    fn = int(np.count_nonzero(labels & (~y_pred)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def binary_metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    """Compute threshold-dependent metrics from one confusion matrix."""

    tp = int(counts["TP"])
    tn = int(counts["TN"])
    fp = int(counts["FP"])
    fn = int(counts["FN"])

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "Balanced_Accuracy": balanced_accuracy,
        "MCC": mcc_from_counts(counts),
    }


def threshold_from_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    *,
    target_precision: float,
    default_threshold: float,
) -> dict[str, Any]:
    """Select threshold from sklearn PR curve arrays."""

    if precision.size != recall.size or precision.size != thresholds.size + 1:
        raise ValueError(
            "precision_recall_curve arrays must satisfy "
            "len(precision) == len(recall) == len(thresholds) + 1."
        )
    if thresholds.size == 0:
        return {"threshold": float(default_threshold), "target_met": False}

    threshold_precision = precision[:-1]
    threshold_recall = recall[:-1]
    candidate_indices = np.flatnonzero(threshold_precision >= float(target_precision))
    if candidate_indices.size == 0:
        return {"threshold": float(default_threshold), "target_met": False}

    candidate_recalls = threshold_recall[candidate_indices]
    best_local_index = int(np.argmax(candidate_recalls))
    best_index = int(candidate_indices[best_local_index])
    return {"threshold": float(thresholds[best_index]), "target_met": True}


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, evaluation_config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute probability metrics, PR threshold selection, and binary metrics."""

    sklearn_metrics = _require_sklearn_metrics()
    pr_precision, pr_recall, pr_thresholds = sklearn_metrics["precision_recall_curve"](
        y_true,
        y_score,
    )
    threshold_info = threshold_from_pr_curve(
        precision=np.asarray(pr_precision, dtype="float64"),
        recall=np.asarray(pr_recall, dtype="float64"),
        thresholds=np.asarray(pr_thresholds, dtype="float64"),
        target_precision=float(evaluation_config["precision_target"]),
        default_threshold=float(evaluation_config["probability_threshold_default"]),
    )
    threshold = float(threshold_info["threshold"])
    counts = confusion_counts_from_scores(y_true, y_score, threshold)
    binary_metrics = binary_metrics_from_counts(counts)

    return {
        "threshold_info": threshold_info,
        "threshold": threshold,
        "counts": counts,
        "binary_metrics": binary_metrics,
        "Positive_Count": int(np.count_nonzero(y_true == 1)),
        "Negative_Count": int(np.count_nonzero(y_true == 0)),
        "AUC": float(sklearn_metrics["roc_auc_score"](y_true, y_score)),
        "AP": float(sklearn_metrics["average_precision_score"](y_true, y_score)),
        "pr_precision": pr_precision,
        "pr_recall": pr_recall,
        "pr_thresholds": pr_thresholds,
    }


def roc_curve_arrays(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sklearn ROC curve arrays."""

    fpr, tpr, thresholds = _require_sklearn_metrics()["roc_curve"](y_true, y_score)
    return (
        np.asarray(fpr, dtype="float64"),
        np.asarray(tpr, dtype="float64"),
        np.asarray(thresholds, dtype="float64"),
    )


def downsample_curve_indices(
    row_count: int,
    max_points: int,
    method: str,
) -> np.ndarray:
    """Return curve row indices to save, preserving first and last rows."""

    if row_count <= 0:
        raise ValueError("curve row_count must be positive.")
    if method != "even":
        raise ValueError(f"Unsupported curve downsample method: {method!r}")
    if row_count <= max_points:
        return np.arange(row_count, dtype=np.int64)
    if max_points < 2:
        raise ValueError("max_points must be >= 2.")

    indices = np.rint(np.linspace(0, row_count - 1, num=max_points)).astype(np.int64)
    indices[0] = 0
    indices[-1] = row_count - 1
    return np.unique(indices)


def aligned_pr_thresholds(thresholds: np.ndarray, row_count: int) -> np.ndarray:
    """Pad sklearn PR thresholds so each precision/recall row has one threshold cell."""

    if row_count != thresholds.size + 1:
        raise ValueError(
            "PR curve arrays must satisfy row_count == len(thresholds) + 1."
        )
    padded = np.empty(row_count, dtype="float64")
    padded[:-1] = thresholds
    padded[-1] = np.nan
    return padded


def curve_rows(
    x_values: Sequence[float],
    y_values: Sequence[float],
    thresholds: Sequence[float],
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    x_key: str,
    y_key: str,
    max_curve_points: int,
    downsample_method: str,
) -> list[dict[str, Any]]:
    """Return downsampled curve CSV rows with common run metadata."""

    x_array = np.asarray(x_values, dtype="float64")
    y_array = np.asarray(y_values, dtype="float64")
    threshold_array = np.asarray(thresholds, dtype="float64")
    if not (x_array.size == y_array.size == threshold_array.size):
        raise ValueError(f"Curve arrays have inconsistent lengths for {model_id} seed {seed}.")

    selected_indices = downsample_curve_indices(
        row_count=int(x_array.size),
        max_points=max_curve_points,
        method=downsample_method,
    )
    return [
        {
            "site_id": site_id,
            "feature_set": feature_set_id,
            "model_id": model_id,
            "seed": seed,
            x_key: float(x_array[int(index)]),
            y_key: float(y_array[int(index)]),
            "threshold": float(threshold_array[int(index)]),
        }
        for index in selected_indices
    ]


def cross_site_curve_rows(
    x_values: Sequence[float],
    y_values: Sequence[float],
    thresholds: Sequence[float],
    *,
    train_site_id: str,
    test_site_id: str,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    x_key: str,
    y_key: str,
    max_curve_points: int,
    downsample_method: str,
) -> list[dict[str, Any]]:
    """Return downsampled cross-site curve CSV rows with run metadata."""

    x_array = np.asarray(x_values, dtype="float64")
    y_array = np.asarray(y_values, dtype="float64")
    threshold_array = np.asarray(thresholds, dtype="float64")
    if not (x_array.size == y_array.size == threshold_array.size):
        raise ValueError(f"Curve arrays have inconsistent lengths for {model_id} seed {seed}.")

    selected_indices = downsample_curve_indices(
        row_count=int(x_array.size),
        max_points=max_curve_points,
        method=downsample_method,
    )
    return [
        {
            "validation_type": "cross_site",
            "train_site_id": train_site_id,
            "test_site_id": test_site_id,
            "site_id": site_id,
            "feature_set": feature_set_id,
            "model_id": model_id,
            "seed": seed,
            x_key: float(x_array[int(index)]),
            y_key: float(y_array[int(index)]),
            "threshold": float(threshold_array[int(index)]),
        }
        for index in selected_indices
    ]


def spatial_block_curve_rows(
    x_values: Sequence[float],
    y_values: Sequence[float],
    thresholds: Sequence[float],
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    fold_id: int,
    x_key: str,
    y_key: str,
    max_curve_points: int,
    downsample_method: str,
) -> list[dict[str, Any]]:
    """Return downsampled spatial-block CV curve CSV rows with run metadata."""

    x_array = np.asarray(x_values, dtype="float64")
    y_array = np.asarray(y_values, dtype="float64")
    threshold_array = np.asarray(thresholds, dtype="float64")
    if not (x_array.size == y_array.size == threshold_array.size):
        raise ValueError(
            f"Curve arrays have inconsistent lengths for {model_id} "
            f"seed {seed} fold {fold_id}."
        )

    selected_indices = downsample_curve_indices(
        row_count=int(x_array.size),
        max_points=max_curve_points,
        method=downsample_method,
    )
    return [
        {
            "validation_type": "spatial_block_cv",
            "site_id": site_id,
            "feature_set": feature_set_id,
            "model_id": model_id,
            "seed": seed,
            "fold_id": fold_id,
            x_key: float(x_array[int(index)]),
            y_key: float(y_array[int(index)]),
            "threshold": float(threshold_array[int(index)]),
        }
        for index in selected_indices
    ]

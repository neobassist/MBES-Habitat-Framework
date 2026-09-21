#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Metric helpers for binary MBES habitat classification experiments.

The functions avoid a hard dependency on scikit-learn so they can be used in a
minimal pipeline environment. NumPy is required for numeric operations.
"""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any, Mapping, Sequence


def _require_numpy() -> Any:
    """Import NumPy with a clear dependency error."""

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "NumPy is required for metric calculations. Install it with "
            "`pip install numpy` or a project environment file."
        ) from exc
    return np


def _as_1d_array(values: Any, name: str, dtype: str | None = None) -> Any:
    """Convert values to a 1D NumPy array."""

    np = _require_numpy()
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _validate_y_true(y_true: Any) -> Any:
    """Validate and return binary ground-truth labels as an integer array."""

    np = _require_numpy()
    labels = _as_1d_array(y_true, "y_true")
    if not np.all(np.isfinite(labels)):
        raise ValueError("y_true contains NaN or infinite values.")

    unique_values = set(np.unique(labels).tolist())
    if not unique_values.issubset({0, 1, False, True}):
        raise ValueError(f"y_true must contain only 0/1 labels, got {sorted(unique_values)}.")
    return labels.astype(int)


def _validate_scores(y_score: Any, expected_length: int) -> Any:
    """Validate and return probability-like scores as a float array."""

    np = _require_numpy()
    scores = _as_1d_array(y_score, "y_score", dtype="float64")
    if scores.size != expected_length:
        raise ValueError(
            f"y_score length ({scores.size}) must match y_true length ({expected_length})."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score contains NaN or infinite values.")
    return scores


def _validate_predictions(y_pred: Any, expected_length: int) -> Any:
    """Validate and return binary predictions as an integer array."""

    np = _require_numpy()
    predictions = _as_1d_array(y_pred, "y_pred")
    if predictions.size != expected_length:
        raise ValueError(
            f"y_pred length ({predictions.size}) must match y_true length ({expected_length})."
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("y_pred contains NaN or infinite values.")

    unique_values = set(np.unique(predictions).tolist())
    if not unique_values.issubset({0, 1, False, True}):
        raise ValueError(f"y_pred must contain only 0/1 labels, got {sorted(unique_values)}.")
    return predictions.astype(int)


def safe_divide(
    numerator: int | float,
    denominator: int | float,
    default: float = 0.0,
) -> float:
    """Divide two numbers and return ``default`` when the denominator is zero."""

    return float(default) if denominator == 0 else float(numerator) / float(denominator)


def apply_threshold(y_score: Any, threshold: float = 0.5) -> Any:
    """Convert scores to binary predictions using ``score >= threshold``."""

    np = _require_numpy()
    if not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be numeric.")
    scores = _as_1d_array(y_score, "y_score", dtype="float64")
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score contains NaN or infinite values.")
    return (scores >= float(threshold)).astype(int)


def find_threshold_at_precision(
    y_true: Any,
    y_score: Any,
    target_precision: float = 0.90,
    return_details: bool = False,
) -> float | dict[str, float | bool]:
    """Find the threshold with highest recall while meeting target precision.

    If no threshold reaches ``target_precision``, the threshold with the highest
    observed precision is returned and ``target_met`` is ``False`` when details
    are requested.
    """

    np = _require_numpy()
    labels = _validate_y_true(y_true)
    scores = _validate_scores(y_score, labels.size)

    if not 0 <= target_precision <= 1:
        raise ValueError("target_precision must be between 0 and 1.")

    thresholds = np.unique(scores)[::-1]
    best: dict[str, float | bool] | None = None
    fallback: dict[str, float | bool] | None = None

    for threshold in thresholds:
        predictions = apply_threshold(scores, float(threshold))
        counts = confusion_counts(labels, predictions)
        precision = safe_divide(counts["TP"], counts["TP"] + counts["FP"])
        recall = safe_divide(counts["TP"], counts["TP"] + counts["FN"])
        candidate = {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "target_met": precision >= target_precision,
        }

        if candidate["target_met"]:
            if best is None or recall > float(best["recall"]):
                best = candidate
        if fallback is None:
            fallback = candidate
        else:
            better_precision = precision > float(fallback["precision"])
            same_precision_better_recall = (
                precision == float(fallback["precision"])
                and recall > float(fallback["recall"])
            )
            if better_precision or same_precision_better_recall:
                fallback = candidate

    selected = best or fallback
    if selected is None:
        raise ValueError("Cannot select a threshold from empty scores.")

    return selected if return_details else float(selected["threshold"])


def compute_threshold_for_precision_target(
    y_true: Any,
    y_score: Any,
    target_precision: float = 0.90,
    return_details: bool = False,
) -> float | dict[str, float | bool]:
    """Compatibility wrapper for precision-target threshold selection."""

    return find_threshold_at_precision(
        y_true=y_true,
        y_score=y_score,
        target_precision=target_precision,
        return_details=return_details,
    )


def confusion_counts(y_true: Any, y_pred: Any) -> dict[str, int]:
    """Return binary confusion counts as ``TP``, ``TN``, ``FP``, and ``FN``."""

    labels = _validate_y_true(y_true)
    predictions = _validate_predictions(y_pred, labels.size)

    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def specificity_score(y_true: Any, y_pred: Any) -> float:
    """Return specificity, defined as ``TN / (TN + FP)``."""

    counts = confusion_counts(y_true, y_pred)
    return safe_divide(counts["TN"], counts["TN"] + counts["FP"])


def compute_binary_metrics(
    y_true: Any,
    y_pred: Any,
    threshold: float | None = None,
) -> dict[str, float | int]:
    """Compute threshold-dependent binary classification metrics.

    If ``threshold`` is provided, ``y_pred`` is treated as scores and thresholded
    before metric calculation.
    """

    labels = _validate_y_true(y_true)
    predictions = apply_threshold(y_pred, threshold) if threshold is not None else _validate_predictions(y_pred, labels.size)
    counts = confusion_counts(labels, predictions)

    precision = safe_divide(counts["TP"], counts["TP"] + counts["FP"])
    recall = safe_divide(counts["TP"], counts["TP"] + counts["FN"])
    specificity = safe_divide(counts["TN"], counts["TN"] + counts["FP"])
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(counts["TP"] + counts["TN"], labels.size)
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Specificity": specificity,
        **counts,
    }


def compute_probability_metrics(y_true: Any, y_score: Any) -> dict[str, float | int]:
    """Compute threshold-independent probability metrics AP and AUC."""

    labels = _validate_y_true(y_true)
    scores = _validate_scores(y_score, labels.size)

    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())

    return {
        "AP": _average_precision(labels, scores),
        "AUC": _roc_auc(labels, scores),
        "Positive_Count": positives,
        "Negative_Count": negatives,
    }


def compute_all_metrics(
    y_true: Any,
    y_score: Any,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute AP, AUC, and threshold-dependent binary metrics together."""

    labels = _validate_y_true(y_true)
    scores = _validate_scores(y_score, labels.size)

    probability_metrics = compute_probability_metrics(labels, scores)
    binary_metrics = compute_binary_metrics(labels, scores, threshold=threshold)
    return {"Threshold": float(threshold), **probability_metrics, **binary_metrics}


def roc_curve_data(y_true: Any, y_score: Any) -> dict[str, list[float]]:
    """Return ROC curve coordinates as serializable lists."""

    np = _require_numpy()
    labels = _validate_y_true(y_true)
    scores = _validate_scores(y_score, labels.size)

    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    thresholds = np.unique(scores)[::-1]

    fpr = [0.0]
    tpr = [0.0]
    out_thresholds = [float("inf")]

    for threshold in thresholds:
        predictions = apply_threshold(scores, float(threshold))
        counts = confusion_counts(labels, predictions)
        tpr.append(safe_divide(counts["TP"], positives))
        fpr.append(safe_divide(counts["FP"], negatives))
        out_thresholds.append(float(threshold))

    return {"fpr": fpr, "tpr": tpr, "thresholds": out_thresholds}


def precision_recall_curve_data(y_true: Any, y_score: Any) -> dict[str, list[float]]:
    """Return precision-recall curve coordinates as serializable lists."""

    np = _require_numpy()
    labels = _validate_y_true(y_true)
    scores = _validate_scores(y_score, labels.size)
    thresholds = np.unique(scores)[::-1]

    precision = []
    recall = []
    out_thresholds = []

    for threshold in thresholds:
        predictions = apply_threshold(scores, float(threshold))
        counts = confusion_counts(labels, predictions)
        precision.append(safe_divide(counts["TP"], counts["TP"] + counts["FP"]))
        recall.append(safe_divide(counts["TP"], counts["TP"] + counts["FN"]))
        out_thresholds.append(float(threshold))

    return {"precision": precision, "recall": recall, "thresholds": out_thresholds}


def summarize_seed_metrics(
    results: Sequence[Mapping[str, Any]] | Any,
    metric_cols: Sequence[str] | None = None,
    group_cols: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Summarize repeated seed results with mean and sample standard deviation."""

    records = _records_from_results(results)
    if not records:
        raise ValueError("results must contain at least one record.")

    if group_cols is None:
        preferred = ["validation_type", "train_site", "test_site", "feature_set", "model"]
        group_cols = [col for col in preferred if col in records[0]]

    return summarize_group_metrics(records, group_cols=group_cols, metric_cols=metric_cols)


def summarize_group_metrics(
    results: Sequence[Mapping[str, Any]] | Any,
    group_cols: Sequence[str],
    metric_cols: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Summarize numeric metrics by group columns."""

    records = _records_from_results(results)
    if not records:
        raise ValueError("results must contain at least one record.")
    if not group_cols:
        raise ValueError("group_cols must contain at least one column.")

    for col in group_cols:
        if col not in records[0]:
            raise KeyError(f"group column {col!r} is missing from result records.")

    if metric_cols is None:
        excluded = set(group_cols) | {"seed", "Threshold"}
        metric_cols = [
            key
            for key in records[0].keys()
            if key not in excluded and _all_numeric(records, key)
        ]

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(record[col] for col in group_cols)
        grouped[key].append(record)

    summaries: list[dict[str, Any]] = []
    for group_key, group_records in grouped.items():
        summary = {col: value for col, value in zip(group_cols, group_key)}
        summary["n"] = len(group_records)
        for metric in metric_cols:
            values = [float(record[metric]) for record in group_records if _is_number(record.get(metric))]
            if not values:
                continue
            summary[f"{metric}_mean"] = sum(values) / len(values)
            summary[f"{metric}_sd"] = _sample_sd(values)
        summaries.append(summary)

    return summaries


def rank_experiments(
    results: Sequence[Mapping[str, Any]] | Any,
    primary_metric: str = "AP",
    descending: bool = True,
    group_cols: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank experiment records or grouped summaries by a primary metric."""

    records = _records_from_results(results)
    if not records:
        raise ValueError("results must contain at least one record.")

    ranking_records = records
    metric_key = primary_metric

    if group_cols is not None:
        ranking_records = summarize_group_metrics(records, group_cols=group_cols)
        metric_key = f"{primary_metric}_mean"
    elif metric_key not in ranking_records[0] and f"{primary_metric}_mean" in ranking_records[0]:
        metric_key = f"{primary_metric}_mean"

    if metric_key not in ranking_records[0]:
        raise KeyError(f"primary metric {primary_metric!r} is missing from results.")

    ranked = sorted(
        [dict(record) for record in ranking_records],
        key=lambda record: float(record[metric_key]),
        reverse=descending,
    )
    for index, record in enumerate(ranked, start=1):
        record["rank"] = index

    return ranked


def _average_precision(labels: Any, scores: Any) -> float:
    """Compute average precision from ranked scores."""

    np = _require_numpy()
    positives = int((labels == 1).sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positive_cumsum = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, sorted_labels.size + 1)
    precision_at_k = true_positive_cumsum / ranks
    return float(precision_at_k[sorted_labels == 1].sum() / positives)


def _roc_auc(labels: Any, scores: Any) -> float:
    """Compute ROC AUC using average ranks, including tied scores."""

    np = _require_numpy()
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    ranks = _average_ranks(scores)
    positive_rank_sum = float(ranks[labels == 1].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _average_ranks(values: Any) -> Any:
    """Return 1-based average ranks for tied values."""

    np = _require_numpy()
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype="float64")

    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _records_from_results(results: Sequence[Mapping[str, Any]] | Any) -> list[dict[str, Any]]:
    """Convert a list of mappings or a pandas-like DataFrame to records."""

    if hasattr(results, "to_dict"):
        records = results.to_dict(orient="records")
    else:
        records = list(results)

    converted = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"results[{index}] must be a mapping, got {type(record).__name__}.")
        converted.append(dict(record))
    return converted


def _is_number(value: Any) -> bool:
    """Return ``True`` when value can be treated as a numeric metric."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _all_numeric(records: Sequence[Mapping[str, Any]], key: str) -> bool:
    """Return ``True`` when every non-missing value for ``key`` is numeric."""

    values = [record.get(key) for record in records if key in record]
    return bool(values) and all(_is_number(value) for value in values)


def _sample_sd(values: Sequence[float]) -> float:
    """Return sample standard deviation, or 0.0 for one value."""

    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return sqrt(variance)

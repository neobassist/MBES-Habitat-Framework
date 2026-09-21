#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Training CSV I/O helpers shared by supervised model stages."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REQUIRED_SAMPLE_COLUMNS = ["site_id", "feature_set", "x", "y", "label"]
OPTIONAL_SAMPLE_METADATA_COLUMNS = [
    "sample_id",
    "raster_row",
    "raster_col",
    "is_positive",
    "sampling_seed",
    "common_domain_policy",
]
PREDICTION_COLUMNS = [
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
CROSS_SITE_PREDICTION_COLUMNS = [
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
SPATIAL_BLOCK_PREDICTION_COLUMNS = [
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
FEATURE_IMPORTANCE_COLUMNS = [
    "site_id",
    "feature_set",
    "model_id",
    "seed",
    "feature",
    "importance",
    "importance_type",
    "rank",
]
TRAIN_SUMMARY_COLUMNS = [
    "site_id",
    "site_name",
    "feature_set",
    "model_id",
    "model_type",
    "seed",
    "samples_path",
    "output_dir",
    "model_path",
    "train_samples_path",
    "test_samples_path",
    "predictions_path",
    "feature_importance_path",
    "n_features",
    "feature_columns",
    "total_samples",
    "train_samples",
    "test_samples",
    "train_positive",
    "train_negative",
    "test_positive",
    "test_negative",
    "train_fraction",
    "test_fraction",
    "split_method",
    "stratify_by_label",
    "params",
    "status",
    "message",
]
SPATIAL_BLOCK_TRAIN_SUMMARY_COLUMNS = [
    "validation_type",
    "site_id",
    "site_name",
    "feature_set",
    "model_id",
    "model_type",
    "seed",
    "fold_id",
    "samples_path",
    "output_dir",
    "block_size_m",
    "n_splits",
    "split_method",
    "n_blocks_total",
    "train_block_count",
    "test_block_count",
    "test_block_ids",
    "train_sample_count",
    "test_sample_count",
    "train_positive_count",
    "train_negative_count",
    "test_positive_count",
    "test_negative_count",
    "model_path",
    "predictions_path",
    "feature_importance_path",
    "n_features",
    "feature_columns",
    "params",
    "status",
    "message",
]


def _require_pandas() -> Any:
    """Import pandas with a clear dependency error."""

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "pandas is required for training CSV I/O. Install pandas in the "
            "mbes_seaweed environment before running this stage."
        ) from exc
    return pd


def json_string(value: Any) -> str:
    """Serialize metadata for CSV summary cells."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def validate_sample_columns(samples: Any, samples_path: Path) -> list[str]:
    """Validate required sample columns and return feature columns."""

    missing = [column for column in REQUIRED_SAMPLE_COLUMNS if column not in samples.columns]
    if missing:
        raise ValueError(
            f"samples.csv is missing required column(s): {', '.join(missing)}; "
            f"path={samples_path}"
        )

    metadata_columns = set(REQUIRED_SAMPLE_COLUMNS) | set(OPTIONAL_SAMPLE_METADATA_COLUMNS)
    feature_columns = [column for column in samples.columns if column not in metadata_columns]
    if not feature_columns:
        raise ValueError(f"samples.csv has zero feature columns: {samples_path}")
    return feature_columns


def read_samples(
    samples_path: Path,
    *,
    site_id: str,
    feature_set_id: str,
    min_class_count: int = 2,
) -> tuple[Any, list[str], np.ndarray]:
    """Read and validate one sample CSV."""

    pd = _require_pandas()
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.csv not found: {samples_path}")
    if not samples_path.is_file():
        raise ValueError(f"samples path is not a file: {samples_path}")

    try:
        samples = pd.read_csv(samples_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"samples.csv is empty: {samples_path}") from exc

    if samples.empty:
        raise ValueError(f"samples.csv contains zero rows: {samples_path}")

    feature_columns = validate_sample_columns(samples, samples_path)

    if not (samples["site_id"].astype(str) == site_id).all():
        raise ValueError(f"samples.csv contains rows for a different site_id: {samples_path}")
    if not (samples["feature_set"].astype(str) == feature_set_id).all():
        raise ValueError(f"samples.csv contains rows for a different feature_set: {samples_path}")

    labels_float = samples["label"].to_numpy(dtype="float64", copy=False)
    if not np.all(np.isfinite(labels_float)):
        raise ValueError(f"label column contains NaN or infinite values: {samples_path}")
    if not np.all(np.isin(labels_float, [0.0, 1.0])):
        unique_values = sorted(set(labels_float.tolist()))
        raise ValueError(
            f"label must contain only 0/1 values in {samples_path}; found {unique_values}"
        )
    labels = labels_float.astype(int)

    feature_values = samples[feature_columns].to_numpy(dtype="float64", copy=False)
    if not np.all(np.isfinite(feature_values)):
        raise ValueError(f"Feature columns contain NaN or infinite values: {samples_path}")

    class_counts = {label: int(np.count_nonzero(labels == label)) for label in [0, 1]}
    if isinstance(min_class_count, bool) or int(min_class_count) < 1:
        raise ValueError("min_class_count must be a positive integer.")
    if min(class_counts.values()) < int(min_class_count):
        if int(min_class_count) == 2:
            raise ValueError(
                f"Each class must contain at least two samples for stratified split: "
                f"path={samples_path}, class_counts={class_counts}"
            )
        raise ValueError(
            f"Each class must contain at least {int(min_class_count)} sample(s): "
            f"path={samples_path}, class_counts={class_counts}"
        )

    return samples, feature_columns, labels


def write_predictions(
    path: Path,
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    test_indices: np.ndarray,
    test_samples: Any,
    y_test: np.ndarray,
    y_score: np.ndarray,
) -> None:
    """Write one seed predictions CSV using the stable schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_rows = {
        "site_id": np.full(test_samples.shape[0], site_id, dtype=object),
        "feature_set": np.full(test_samples.shape[0], feature_set_id, dtype=object),
        "model_id": np.full(test_samples.shape[0], model_id, dtype=object),
        "seed": np.full(test_samples.shape[0], seed, dtype=int),
        "split": np.full(test_samples.shape[0], "test", dtype=object),
        "sample_index": test_indices,
        "x": test_samples["x"].to_numpy(copy=False),
        "y": test_samples["y"].to_numpy(copy=False),
        "y_true": y_test,
        "y_score": y_score,
    }
    _require_pandas().DataFrame(prediction_rows, columns=PREDICTION_COLUMNS).to_csv(
        path,
        index=False,
    )


def write_cross_site_predictions(
    path: Path,
    *,
    train_site_id: str,
    test_site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    test_indices: np.ndarray,
    test_samples: Any,
    y_test: np.ndarray,
    y_score: np.ndarray,
) -> None:
    """Write one cross-site seed predictions CSV using the stable schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_rows = {
        "validation_type": np.full(test_samples.shape[0], "cross_site", dtype=object),
        "train_site_id": np.full(test_samples.shape[0], train_site_id, dtype=object),
        "test_site_id": np.full(test_samples.shape[0], test_site_id, dtype=object),
        "site_id": np.full(test_samples.shape[0], test_site_id, dtype=object),
        "feature_set": np.full(test_samples.shape[0], feature_set_id, dtype=object),
        "model_id": np.full(test_samples.shape[0], model_id, dtype=object),
        "seed": np.full(test_samples.shape[0], seed, dtype=int),
        "split": np.full(test_samples.shape[0], "test", dtype=object),
        "sample_index": test_indices,
        "x": test_samples["x"].to_numpy(copy=False),
        "y": test_samples["y"].to_numpy(copy=False),
        "y_true": y_test,
        "y_score": y_score,
    }
    _require_pandas().DataFrame(
        prediction_rows,
        columns=CROSS_SITE_PREDICTION_COLUMNS,
    ).to_csv(path, index=False)


def write_spatial_block_predictions(
    path: Path,
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    fold_id: int,
    test_indices: np.ndarray,
    test_samples: Any,
    test_block_ids: np.ndarray,
    y_test: np.ndarray,
    y_score: np.ndarray,
) -> None:
    """Write one spatial-block CV fold predictions CSV using the stable schema."""

    if len(test_block_ids) != test_samples.shape[0]:
        raise ValueError(
            f"test_block_ids length ({len(test_block_ids)}) does not match "
            f"test sample count ({test_samples.shape[0]})."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_rows = {
        "validation_type": np.full(test_samples.shape[0], "spatial_block_cv", dtype=object),
        "site_id": np.full(test_samples.shape[0], site_id, dtype=object),
        "feature_set": np.full(test_samples.shape[0], feature_set_id, dtype=object),
        "model_id": np.full(test_samples.shape[0], model_id, dtype=object),
        "seed": np.full(test_samples.shape[0], seed, dtype=int),
        "fold_id": np.full(test_samples.shape[0], fold_id, dtype=int),
        "block_id": np.asarray(test_block_ids, dtype=object),
        "split": np.full(test_samples.shape[0], "test", dtype=object),
        "sample_index": test_indices,
        "x": test_samples["x"].to_numpy(copy=False),
        "y": test_samples["y"].to_numpy(copy=False),
        "y_true": y_test,
        "y_score": y_score,
    }
    _require_pandas().DataFrame(
        prediction_rows,
        columns=SPATIAL_BLOCK_PREDICTION_COLUMNS,
    ).to_csv(path, index=False)


def write_feature_importance(
    path: Path,
    *,
    site_id: str,
    feature_set_id: str,
    model_id: str,
    seed: int,
    feature_columns: Sequence[str],
    importance: np.ndarray,
    importance_type: str,
) -> None:
    """Write feature importance CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-importance, kind="mergesort")
    rows = []
    for rank, feature_index in enumerate(order, start=1):
        rows.append(
            {
                "site_id": site_id,
                "feature_set": feature_set_id,
                "model_id": model_id,
                "seed": seed,
                "feature": feature_columns[int(feature_index)],
                "importance": float(importance[int(feature_index)]),
                "importance_type": importance_type,
                "rank": rank,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FEATURE_IMPORTANCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_train_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write model-level train summary CSV after the seed loop."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRAIN_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_spatial_block_train_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write model-level spatial-block CV train summary CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SPATIAL_BLOCK_TRAIN_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

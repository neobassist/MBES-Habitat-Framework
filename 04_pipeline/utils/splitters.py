#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validation split helpers shared by model-training stages."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _require_train_test_split() -> Any:
    """Import sklearn train_test_split with a clear dependency error."""

    try:
        from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "scikit-learn is required for within-site train/test splitting. "
            "Install scikit-learn in the mbes_seaweed environment before "
            "running this stage."
        ) from exc
    return train_test_split


def _require_stratified_group_kfold() -> Any:
    """Import sklearn StratifiedGroupKFold with a clear dependency error."""

    try:
        from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "scikit-learn is required for spatial block cross-validation. "
            "Install scikit-learn in the mbes_seaweed environment before "
            "running this stage."
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "sklearn.model_selection.StratifiedGroupKFold is required for "
            "spatial block cross-validation. Upgrade scikit-learn in the "
            "mbes_seaweed environment before running this stage."
        ) from exc
    return StratifiedGroupKFold


def validate_split_classes(labels: np.ndarray, split_indices: np.ndarray, split_name: str) -> None:
    """Ensure one split contains both binary classes."""

    split_labels = labels[split_indices]
    if not {0, 1}.issubset(set(split_labels.tolist())):
        counts = {
            "negative": int(np.count_nonzero(split_labels == 0)),
            "positive": int(np.count_nonzero(split_labels == 1)),
        }
        raise ValueError(f"{split_name} split is missing a class: {counts}")


def within_site_stratified_split(
    *,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    split_config: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices into within-site train/test arrays."""

    train_test_split = _require_train_test_split()
    stratify = labels if bool(split_config["stratify_by_label"]) else None

    try:
        train_indices, test_indices = train_test_split(
            sample_indices,
            test_size=float(split_config["test_fraction"]),
            random_state=int(seed),
            stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(f"Unable to perform within-site train/test split: {exc}") from exc

    validate_split_classes(labels, train_indices, "train")
    validate_split_classes(labels, test_indices, "test")
    return train_indices, test_indices


def assign_spatial_blocks(samples: Any, block_size_m: float) -> np.ndarray:
    """Assign regular-grid spatial block ids from sample x/y coordinates."""

    if isinstance(block_size_m, bool) or not isinstance(block_size_m, (int, float)):
        raise ValueError("block_size_m must be numeric.")
    if float(block_size_m) <= 0:
        raise ValueError("block_size_m must be > 0.")
    for column in ["x", "y"]:
        if column not in samples.columns:
            raise ValueError(f"samples is missing required coordinate column: {column}")

    x = samples["x"].to_numpy(dtype="float64", copy=False)
    y = samples["y"].to_numpy(dtype="float64", copy=False)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Sample x/y coordinates contain NaN or infinite values.")

    block_cols = np.floor(x / float(block_size_m)).astype("int64")
    block_rows = np.floor(y / float(block_size_m)).astype("int64")
    return np.fromiter(
        (f"{row}_{col}" for row, col in zip(block_rows, block_cols)),
        dtype=object,
        count=x.size,
    )


def summarize_spatial_blocks(samples: Any, block_ids: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Summarize occupied spatial blocks and class occupancy."""

    del samples
    block_ids = np.asarray(block_ids, dtype=object)
    labels = np.asarray(labels, dtype=int)
    if block_ids.shape[0] != labels.shape[0]:
        raise ValueError(
            f"block_ids length ({block_ids.shape[0]}) does not match labels length "
            f"({labels.shape[0]})."
        )

    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    positive_blocks = set(inverse[labels == 1].tolist())
    negative_blocks = set(inverse[labels == 0].tolist())
    mixed_blocks = positive_blocks.intersection(negative_blocks)
    return {
        "n_blocks_total": int(unique_blocks.size),
        "positive_block_count": int(len(positive_blocks)),
        "negative_block_count": int(len(negative_blocks)),
        "mixed_block_count": int(len(mixed_blocks)),
    }


def make_spatial_block_folds(
    labels: np.ndarray,
    block_ids: np.ndarray,
    n_splits: int,
    seed: int,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create StratifiedGroupKFold train/test position arrays."""

    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2.")
    if not isinstance(shuffle, bool):
        raise ValueError("shuffle must be a bool.")

    labels = np.asarray(labels, dtype=int)
    block_ids = np.asarray(block_ids, dtype=object)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D array.")
    if block_ids.ndim != 1:
        raise ValueError("block_ids must be a 1-D array.")
    if labels.shape[0] != block_ids.shape[0]:
        raise ValueError(
            f"labels length ({labels.shape[0]}) does not match block_ids length "
            f"({block_ids.shape[0]})."
        )

    unique_blocks = np.unique(block_ids)
    if unique_blocks.size < n_splits:
        raise ValueError(
            f"Spatial block CV requires at least n_splits occupied blocks: "
            f"n_blocks={unique_blocks.size}, n_splits={n_splits}"
        )

    StratifiedGroupKFold = _require_stratified_group_kfold()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=int(seed) if shuffle else None,
    )
    positions = np.arange(labels.shape[0])
    try:
        return [
            (train_positions, test_positions)
            for train_positions, test_positions in splitter.split(
                positions,
                labels,
                groups=block_ids,
            )
        ]
    except ValueError as exc:
        raise ValueError(f"Unable to create spatial block CV folds: {exc}") from exc


def validate_spatial_fold_classes(
    *,
    labels: np.ndarray,
    block_ids: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    site_id: str,
    feature_set_id: str,
    seed: int,
    fold_id: int,
    block_size_m: float,
    min_positive_per_fold: int,
    min_negative_per_fold: int,
) -> dict[str, int]:
    """Validate one spatial fold and return class/block counts."""

    labels = np.asarray(labels, dtype=int)
    block_ids = np.asarray(block_ids, dtype=object)
    train_positions = np.asarray(train_positions, dtype=int)
    test_positions = np.asarray(test_positions, dtype=int)

    train_labels = labels[train_positions]
    test_labels = labels[test_positions]
    train_positive = int(np.count_nonzero(train_labels == 1))
    train_negative = int(np.count_nonzero(train_labels == 0))
    test_positive = int(np.count_nonzero(test_labels == 1))
    test_negative = int(np.count_nonzero(test_labels == 0))

    train_blocks = set(block_ids[train_positions].tolist())
    test_blocks = set(block_ids[test_positions].tolist())
    overlap = train_blocks.intersection(test_blocks)
    if (
        train_positive == 0
        or train_negative == 0
        or test_positive < int(min_positive_per_fold)
        or test_negative < int(min_negative_per_fold)
        or overlap
    ):
        raise ValueError(
            "Invalid spatial block CV fold: "
            f"site_id={site_id}, feature_set={feature_set_id}, seed={seed}, "
            f"fold_id={fold_id}, block_size_m={block_size_m}, "
            f"train_positive={train_positive}, train_negative={train_negative}, "
            f"test_positive={test_positive}, test_negative={test_negative}, "
            f"train_block_count={len(train_blocks)}, test_block_count={len(test_blocks)}, "
            f"overlap_block_count={len(overlap)}"
        )

    return {
        "train_positive": train_positive,
        "train_negative": train_negative,
        "test_positive": test_positive,
        "test_negative": test_negative,
        "train_block_count": int(len(train_blocks)),
        "test_block_count": int(len(test_blocks)),
    }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train final site models and predict full-site probability maps.

This stage uses final candidate combinations selected by
``11_analyze_validation_results.py``. It trains each final model with all
labeled samples for the same site, then applies the model to the full feature
stack using raster windows. It does not compute validation metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    configured_model_persistence,
    configured_final_mapping_policy,
    ensure_output_dirs,
    get_site_config,
    load_analysis_config,
    load_all_model_configs,
    load_experiment_matrix,
    load_publication_config,
    load_project_config,
    load_sites_config,
    publication_display_labels,
    validate_candidate_contract,
)
from utils.modeling import (
    build_model,
    extract_feature_importance,
    model_parts,
    predict_positive_scores,
    save_model,
)
from utils.raster_io import read_band_names
from utils.training_io import json_string, read_samples, write_feature_importance
from utils.plotting import add_map_cartography, configure_publication_font


TRAIN_SUMMARY_COLUMNS = [
    "project_id",
    "candidate_label",
    "site_id",
    "site_name",
    "feature_set",
    "model_id",
    "model_type",
    "seed",
    "samples_path",
    "raster_stack_path",
    "bands_path",
    "output_dir",
    "sample_count",
    "positive_count",
    "negative_count",
    "n_features",
    "feature_columns",
    "params",
    "training_scope",
    "sample_authority_path",
    "sample_authority_sha256",
    "samples_sha256",
    "common_valid_mask_path",
    "common_valid_mask_sha256",
    "model_config_path",
    "model_config_sha256",
    "analysis_config_path",
    "analysis_config_sha256",
    "candidate_authority_path",
    "candidate_authority_sha256",
    "estimator_sha256",
    "reference_mean_AP",
    "reference_within_AP",
    "reference_cross_AP",
    "reference_spatial_AP",
    "model_path",
    "feature_importance_path",
    "prediction_probability_path",
    "prediction_valid_mask_path",
    "preview_png_path",
    "resolved_font_family",
    "status",
    "message",
]
CANDIDATE_REQUIRED_COLUMNS = [
    "candidate_label",
    "feature_set",
    "model_id",
    "within_AP",
    "cross_AP",
    "spatial_AP",
    "mean_AP",
]
SAMPLE_PREFIX_COLUMNS = ["site_id", "feature_set", "x", "y", "label"]
PROBABILITY_NODATA = float("nan")


def _require_rasterio() -> Any:
    """Import rasterio with a clear dependency error."""

    try:
        import rasterio  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "rasterio is required for final map prediction. Install rasterio "
            "in the mbes_seaweed environment before running this stage."
        ) from exc
    return rasterio


def _require_matplotlib_pyplot() -> Any:
    """Import matplotlib.pyplot with a clear dependency error."""

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "matplotlib is required for prediction_preview.png. Run with "
            "--no-preview to skip preview generation."
        ) from exc
    return plt


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


def _require_existing_file(path: Path, label: str) -> Path:
    """Return an existing file path or raise a clear exception."""

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one required provenance file."""

    _require_existing_file(path, "Provenance input")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_final_model_persistence(project_config: Mapping[str, Any]) -> None:
    """Fail before training when final-estimator persistence is disabled."""

    persistence = configured_model_persistence(project_config)
    if not persistence["save_final_models"]:
        raise ValueError(
            "Stage 12 requires outputs.save_final_models=true because final maps "
            "must retain their fitted estimator."
        )


def _read_final_candidates(path: Path) -> list[dict[str, Any]]:
    """Read stage-2 final candidate combinations."""

    _require_existing_file(path, "stage2_final_candidate_comparison.csv")
    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Final candidate CSV is empty: {path}") from exc
    if data.empty:
        raise ValueError(f"Final candidate CSV contains zero rows: {path}")

    missing = [column for column in CANDIDATE_REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "stage2_final_candidate_comparison.csv is missing required "
            f"column(s): {', '.join(missing)}"
        )

    for column in ["within_AP", "cross_AP", "spatial_AP", "mean_AP"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    candidates: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        candidates.append(
            {
                "candidate_label": str(row["candidate_label"]),
                "feature_set": str(row["feature_set"]),
                "model_id": str(row["model_id"]),
                "reference_mean_AP": float(row["mean_AP"]) if pd.notna(row["mean_AP"]) else np.nan,
                "reference_within_AP": float(row["within_AP"]) if pd.notna(row["within_AP"]) else np.nan,
                "reference_cross_AP": float(row["cross_AP"]) if pd.notna(row["cross_AP"]) else np.nan,
                "reference_spatial_AP": float(row["spatial_AP"]) if pd.notna(row["spatial_AP"]) else np.nan,
                "is_custom": False,
            }
        )
    return candidates


def _candidate_from_custom(
    feature_set: str,
    model_id: str,
    candidate_label: str | None = None,
) -> dict[str, Any]:
    """Return one custom final-map candidate without validation references."""

    label = candidate_label or f"custom_{feature_set}_{model_id}"
    return {
        "candidate_label": label,
        "feature_set": feature_set,
        "model_id": model_id,
        "reference_mean_AP": np.nan,
        "reference_within_AP": np.nan,
        "reference_cross_AP": np.nan,
        "reference_spatial_AP": np.nan,
        "is_custom": True,
    }


def _filter_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    candidate_label: str | None = None,
    feature_set: str | None = None,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Filter final candidates or create a custom candidate."""

    if (feature_set is None) != (model_id is None):
        raise ValueError("--feature-set and --model-id must be provided together for custom runs.")

    selected = [dict(candidate) for candidate in candidates]
    if candidate_label is not None:
        selected = [
            candidate
            for candidate in selected
            if str(candidate["candidate_label"]) == str(candidate_label)
        ]
    if feature_set is not None and model_id is not None:
        selected = [
            candidate
            for candidate in selected
            if str(candidate["feature_set"]) == str(feature_set)
            and str(candidate["model_id"]) == str(model_id)
        ]
        if not selected:
            selected = [_candidate_from_custom(feature_set, model_id, candidate_label)]

    if not selected:
        raise ValueError(
            "No final-map candidates selected. Check --candidate-label, "
            "--feature-set, and --model-id."
        )
    return selected


def _resolve_samples_path(output_root: Path, site_id: str, feature_set: str) -> Path:
    """Return the sample CSV path for one site/feature set."""

    return output_root / "samples" / site_id / f"{feature_set}_samples.csv"


def _resolve_stack_paths(output_root: Path, site_id: str, feature_set: str) -> tuple[Path, Path]:
    """Return feature stack and bands.txt paths."""

    feature_dir = output_root / "feature_sets" / site_id
    return feature_dir / f"{feature_set}.tif", feature_dir / f"{feature_set}.bands.txt"


def _validate_feature_alignment(
    samples_feature_columns: Sequence[str],
    band_names: Sequence[str],
    raster_count: int,
    *,
    stack_path: Path,
    bands_path: Path,
) -> None:
    """Validate sample features, bands.txt names, and raster band count."""

    sample_list = list(samples_feature_columns)
    band_list = list(band_names)
    duplicates = sorted({feature for feature in sample_list if sample_list.count(feature) > 1})
    if duplicates:
        raise ValueError(f"Duplicate sample feature column(s): {', '.join(duplicates)}")
    duplicates = sorted({feature for feature in band_list if band_list.count(feature) > 1})
    if duplicates:
        raise ValueError(f"Duplicate band name(s) in {bands_path}: {', '.join(duplicates)}")
    if sample_list != band_list:
        raise ValueError(
            "Sample feature columns do not match bands.txt names in order: "
            f"stack_path={stack_path}, bands_path={bands_path}"
        )
    if len(sample_list) != int(raster_count):
        raise ValueError(
            "Sample feature count does not match raster band count: "
            f"samples={len(sample_list)}, raster_count={raster_count}, stack_path={stack_path}"
        )


def _manual_windows(width: int, height: int, window_size: int) -> Iterable[Any]:
    """Yield square rasterio windows covering a raster."""

    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("--window-size must be a positive integer.")

    rasterio = _require_rasterio()
    Window = rasterio.windows.Window
    for row_off in range(0, int(height), int(window_size)):
        win_height = min(int(window_size), int(height) - row_off)
        for col_off in range(0, int(width), int(window_size)):
            win_width = min(int(window_size), int(width) - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_width, height=win_height)


def _stack_windows(src: Any, window_size: int | None) -> Iterable[Any]:
    """Yield prediction windows from internal blocks or manual tiles."""

    if window_size is not None:
        yield from _manual_windows(src.width, src.height, window_size)
        return

    seen: set[tuple[float, float, float, float]] = set()
    for _, window in src.block_windows(1):
        key = (window.col_off, window.row_off, window.width, window.height)
        if key not in seen:
            seen.add(key)
            yield window


def _train_final_model(
    samples_df: Any,
    feature_columns: Sequence[str],
    label_col: str,
    model_id: str,
    model_configs: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> tuple[Any, dict[str, Any], str, np.ndarray, str]:
    """Train one final model with all labeled samples."""

    if model_id not in model_configs:
        raise KeyError(f"Model config was not loaded: {model_id}")
    model_config = model_configs[model_id]
    model_meta, _ = model_parts(model_id, model_config)
    model_type = str(model_meta["type"])

    x_train = samples_df[list(feature_columns)]
    y_train = samples_df[label_col].to_numpy(dtype=int, copy=False)
    model, effective_params = build_model(model_config, seed)
    model.fit(x_train, y_train)

    importance, importance_type = extract_feature_importance(
        model,
        model_id=model_id,
        model_type=model_type,
        feature_columns=feature_columns,
    )
    return model, effective_params, model_type, importance, importance_type


def _valid_mask_for_window(data: np.ndarray, nodata_values: Sequence[Any]) -> np.ndarray:
    """Return pixels finite and valid across all bands."""

    valid = np.all(np.isfinite(data), axis=0)
    for band_index, nodata in enumerate(nodata_values):
        if nodata is None:
            continue
        try:
            if np.isnan(nodata):
                continue
        except TypeError:
            pass
        valid &= data[band_index] != nodata
    return valid


def _predict_stack_windows(
    model: Any,
    *,
    model_id: str,
    stack_path: Path,
    feature_columns: Sequence[str],
    probability_path: Path,
    mask_path: Path,
    window_size: int | None,
) -> dict[str, int]:
    """Predict positive probabilities for a stack using raster windows."""

    rasterio = _require_rasterio()
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    total_pixels = 0
    valid_pixels = 0
    invalid_pixels = 0
    window_count = 0

    with rasterio.open(stack_path) as src:
        probability_profile = src.profile.copy()
        probability_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=PROBABILITY_NODATA,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )
        mask_profile = src.profile.copy()
        mask_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )
        nodata_values = list(src.nodatavals)

        with rasterio.open(probability_path, "w", **probability_profile) as prob_dst, rasterio.open(
            mask_path,
            "w",
            **mask_profile,
        ) as mask_dst:
            for window in _stack_windows(src, window_size):
                stack = src.read(window=window, masked=False).astype("float32", copy=False)
                band_count, win_height, win_width = stack.shape
                pixels = int(win_height * win_width)
                total_pixels += pixels
                window_count += 1

                valid = _valid_mask_for_window(stack, nodata_values)
                valid_flat = valid.reshape(-1)
                valid_count = int(np.count_nonzero(valid_flat))
                valid_pixels += valid_count
                invalid_pixels += int(pixels - valid_count)

                probability = np.full((win_height * win_width), np.nan, dtype="float32")
                if valid_count:
                    features = stack.reshape(band_count, -1).T
                    x_valid = pd.DataFrame(
                        features[valid_flat],
                        columns=list(feature_columns),
                    )
                    scores = predict_positive_scores(model, x_valid, model_id=model_id)
                    probability[valid_flat] = np.asarray(scores, dtype="float32")

                prob_dst.write(probability.reshape(win_height, win_width), 1, window=window)
                mask_dst.write(valid.astype("uint8", copy=False), 1, window=window)

    return {
        "window_count": window_count,
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "invalid_pixels": invalid_pixels,
    }


def _write_preview(
    probability_path: Path,
    preview_path: Path,
    *,
    site_name: str,
    candidate_label: str,
    feature_set: str,
    model_id: str,
) -> str:
    """Write a labeled, cartographically complete probability quicklook PNG."""

    rasterio = _require_rasterio()
    plt = _require_matplotlib_pyplot()
    resolved_font = configure_publication_font()

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(probability_path) as src:
        max_dim = 1600
        scale = max(src.width / max_dim, src.height / max_dim, 1.0)
        out_width = max(1, int(src.width / scale))
        out_height = max(1, int(src.height / scale))
        preview = src.read(1, out_shape=(out_height, out_width), masked=True)
        bounds = src.bounds
        if src.crs is None or not src.crs.is_projected:
            raise ValueError("Prediction preview cartography requires a projected CRS.")
        if float(src.transform.e) >= 0:
            raise ValueError("Prediction preview cartography requires a north-up raster.")

    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(
        preview,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
    )
    ax.set_title(
        f"{site_name} | {candidate_label} | {feature_set} | {model_id}\n"
        "Final predicted habitat probability",
        fontsize=11,
    )
    ax.set_xlabel("Easting (m, UTM Zone 52N)")
    ax.set_ylabel("Northing (m, UTM Zone 52N)")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_aspect("equal")
    add_map_cartography(ax, bounds)
    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Predicted habitat probability",
    )
    fig.tight_layout()
    fig.savefig(preview_path, dpi=180)
    plt.close(fig)
    return resolved_font


def _write_train_summary(path: Path, row: Mapping[str, Any]) -> None:
    """Write one final-map train summary CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRAIN_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(dict(row))


def _validate_output_paths(paths: Sequence[Path], overwrite: bool) -> None:
    """Raise if outputs exist and overwrite is disabled."""

    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "--no-overwrite was set and output file(s) already exist: " + "; ".join(existing)
        )


def _raster_metadata(stack_path: Path) -> dict[str, Any]:
    """Return raster metadata required for dry-run and alignment checks."""

    rasterio = _require_rasterio()
    with rasterio.open(stack_path) as src:
        return {
            "count": int(src.count),
            "width": int(src.width),
            "height": int(src.height),
            "crs": str(src.crs),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "descriptions": list(src.descriptions),
        }


def process_site_candidate(
    *,
    project_id: str,
    site_id: str,
    site_name: str,
    candidate: Mapping[str, Any],
    candidate_display_label: str | None,
    output_root: Path,
    final_maps_root: Path,
    model_configs: Mapping[str, Mapping[str, Any]],
    candidate_authority_path: Path,
    seed: int,
    window_size: int | None,
    overwrite: bool,
    no_preview: bool,
    dry_run: bool,
) -> Path | None:
    """Train and predict one site/candidate final map set."""

    candidate_label = str(candidate["candidate_label"])
    feature_set = str(candidate["feature_set"])
    model_id = str(candidate["model_id"])

    samples_path = _resolve_samples_path(output_root, site_id, feature_set)
    stack_path, bands_path = _resolve_stack_paths(output_root, site_id, feature_set)
    output_dir = final_maps_root / site_id / candidate_label
    model_path = output_dir / "final_model.joblib"
    feature_importance_path = output_dir / "feature_importance.csv"
    train_summary_path = output_dir / "train_summary.csv"
    probability_path = output_dir / "prediction_probability.tif"
    mask_path = output_dir / "prediction_valid_mask.tif"
    preview_path = output_dir / "prediction_preview.png"
    framework_root = Path(__file__).resolve().parents[1]
    analysis_config_path = framework_root / "03_experiments" / project_id / "analysis.yaml"
    model_config_path = framework_root / "03_experiments" / project_id / "models" / f"{model_id}.yaml"
    common_valid_mask_path = output_root / "prepared" / site_id / "common_valid_mask.tif"
    common_sample_path = output_root / "samples_common" / site_id / "sample_index.csv"
    sample_authority_path = common_sample_path if common_sample_path.is_file() else samples_path

    print(
        f"Predicting final map: {site_id} / {candidate_label} / {feature_set} / {model_id}",
        flush=True,
    )

    _require_existing_file(samples_path, "samples.csv")
    _require_existing_file(stack_path, "Feature stack")
    _require_existing_file(bands_path, "Band names file")
    _require_existing_file(sample_authority_path, "Sample authority")
    _require_existing_file(common_valid_mask_path, "Common-valid mask")
    _require_existing_file(model_config_path, "Model configuration")
    _require_existing_file(analysis_config_path, "Analysis configuration")
    _require_existing_file(candidate_authority_path, "Candidate authority")

    samples, feature_columns, labels = read_samples(
        samples_path,
        site_id=site_id,
        feature_set_id=feature_set,
        min_class_count=1,
    )
    band_names = read_band_names(bands_path)
    raster_meta = _raster_metadata(stack_path)
    _validate_feature_alignment(
        feature_columns,
        band_names,
        int(raster_meta["count"]),
        stack_path=stack_path,
        bands_path=bands_path,
    )

    if model_id not in model_configs:
        raise KeyError(f"Model config was not loaded: {model_id}")

    output_paths = [model_path, feature_importance_path, train_summary_path, probability_path, mask_path]
    if not no_preview:
        output_paths.append(preview_path)
    _validate_output_paths(output_paths, overwrite)

    dry_run_record = {
        "site_id": site_id,
        "candidate_label": candidate_label,
        "feature_set": feature_set,
        "model_id": model_id,
        "samples_path": str(samples_path),
        "samples_exists": samples_path.exists(),
        "stack_path": str(stack_path),
        "stack_exists": stack_path.exists(),
        "bands_path": str(bands_path),
        "bands_exists": bands_path.exists(),
        "sample_feature_count": len(feature_columns),
        "band_names_count": len(band_names),
        "raster_band_count": int(raster_meta["count"]),
        "raster_width": int(raster_meta["width"]),
        "raster_height": int(raster_meta["height"]),
        "output_dir": str(output_dir),
        "would_train": True,
        "would_predict": True,
    }
    if dry_run:
        print("Dry-run final map target: " + json_string(dry_run_record), flush=True)
        return None

    model, effective_params, model_type, importance, importance_type = _train_final_model(
        samples,
        feature_columns,
        "label",
        model_id,
        model_configs,
        seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, model_path)
    estimator_sha256 = _sha256_file(model_path)
    print(f"Trained final model: {site_id} / {candidate_label}", flush=True)

    write_feature_importance(
        feature_importance_path,
        site_id=site_id,
        feature_set_id=feature_set,
        model_id=model_id,
        seed=seed,
        feature_columns=feature_columns,
        importance=importance,
        importance_type=importance_type,
    )

    prediction_stats = _predict_stack_windows(
        model,
        model_id=model_id,
        stack_path=stack_path,
        feature_columns=feature_columns,
        probability_path=probability_path,
        mask_path=mask_path,
        window_size=window_size,
    )
    print(f"Wrote final probability map: {probability_path}", flush=True)

    preview_message = ""
    preview_value = "" if no_preview else str(preview_path)
    resolved_font_family = "not_applicable_preview_disabled" if no_preview else ""
    if not no_preview:
        try:
            resolved_font_family = _write_preview(
                probability_path,
                preview_path,
                site_name=site_name,
                candidate_label=candidate_display_label or candidate_label,
                feature_set=feature_set,
                model_id=model_id,
            )
            print(f"Resolved publication font: {resolved_font_family}", flush=True)
        except Exception as exc:  # pragma: no cover - preview is non-critical provenance.
            preview_message = f"Preview generation failed: {exc}"
            preview_value = ""
            print(preview_message, flush=True)

    positive_count = int(np.count_nonzero(labels == 1))
    negative_count = int(np.count_nonzero(labels == 0))
    summary_message = preview_message
    if prediction_stats:
        summary_message = (
            (summary_message + " | " if summary_message else "")
            + "prediction_windows={window_count}; valid_pixels={valid_pixels}; "
            "invalid_pixels={invalid_pixels}".format(**prediction_stats)
        )

    train_summary = {
        "project_id": project_id,
        "candidate_label": candidate_label,
        "site_id": site_id,
        "site_name": site_name,
        "feature_set": feature_set,
        "model_id": model_id,
        "model_type": model_type,
        "seed": seed,
        "samples_path": str(samples_path),
        "raster_stack_path": str(stack_path),
        "bands_path": str(bands_path),
        "output_dir": str(output_dir),
        "sample_count": int(samples.shape[0]),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "n_features": len(feature_columns),
        "feature_columns": json_string(list(feature_columns)),
        "params": json_string(effective_params),
        "training_scope": "site_specific_full_sample",
        "sample_authority_path": str(sample_authority_path),
        "sample_authority_sha256": _sha256_file(sample_authority_path),
        "samples_sha256": _sha256_file(samples_path),
        "common_valid_mask_path": str(common_valid_mask_path),
        "common_valid_mask_sha256": _sha256_file(common_valid_mask_path),
        "model_config_path": str(model_config_path),
        "model_config_sha256": _sha256_file(model_config_path),
        "analysis_config_path": str(analysis_config_path),
        "analysis_config_sha256": _sha256_file(analysis_config_path),
        "candidate_authority_path": str(candidate_authority_path),
        "candidate_authority_sha256": _sha256_file(candidate_authority_path),
        "estimator_sha256": estimator_sha256,
        "reference_mean_AP": candidate.get("reference_mean_AP", np.nan),
        "reference_within_AP": candidate.get("reference_within_AP", np.nan),
        "reference_cross_AP": candidate.get("reference_cross_AP", np.nan),
        "reference_spatial_AP": candidate.get("reference_spatial_AP", np.nan),
        "model_path": str(model_path),
        "feature_importance_path": str(feature_importance_path),
        "prediction_probability_path": str(probability_path),
        "prediction_valid_mask_path": str(mask_path),
        "preview_png_path": preview_value,
        "resolved_font_family": resolved_font_family,
        "status": "ok",
        "message": summary_message,
    }
    _write_train_summary(train_summary_path, train_summary)
    print(f"Wrote final train summary: {train_summary_path}", flush=True)
    return train_summary_path


def run_prediction_maps(
    project_id: str = DEFAULT_PROJECT_ID,
    *,
    site_filter: str | None = None,
    candidate_label: str | None = None,
    feature_set: str | None = None,
    model_id: str | None = None,
    output_dir: str | Path | None = None,
    no_preview: bool = False,
    overwrite: bool = True,
    dry_run: bool = False,
    window_size: int | None = None,
    seed: int | None = None,
) -> list[Path]:
    """Run final map prediction for selected sites and candidates."""

    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("--seed must be an integer when provided.")
    if window_size is not None and (
        isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0
    ):
        raise ValueError("--window-size must be a positive integer.")

    project_config = load_project_config(project_id=project_id)
    _require_final_model_persistence(project_config)
    sites_config = load_sites_config(project_id=project_id)
    experiment_matrix = load_experiment_matrix(project_id=project_id)
    analysis_config = load_analysis_config(project_id=project_id)
    model_configs = load_all_model_configs(project_id=project_id)
    display_labels = publication_display_labels(
        load_publication_config(project_id=project_id)
    )
    output_dirs = ensure_output_dirs(project_config, stage_name="12")
    output_root = output_dirs["output_root"]
    final_maps_root = Path(output_dir).expanduser() if output_dir else output_root / "final_maps"

    mapping_policy = configured_final_mapping_policy(analysis_config, experiment_matrix)
    site_ids = list(mapping_policy["sites"])
    effective_seed = mapping_policy["seed"] if seed is None else seed
    if site_filter is not None:
        if site_filter not in site_ids:
            raise ValueError(
                f"--site={site_filter!r} is not selected by analysis.final_mapping.sites."
            )
        site_ids = [site_filter]

    candidate_path = output_root / "analysis_compare" / "stage2_final_candidate_comparison.csv"
    candidates = _read_final_candidates(candidate_path)
    validate_candidate_contract(candidates, analysis_config)
    selected_candidates = _filter_candidates(
        candidates,
        candidate_label=candidate_label,
        feature_set=feature_set,
        model_id=model_id,
    )

    model_ids = _require_list(experiment_matrix.get("models"), "experiment_matrix.models")
    feature_set_ids = _require_list(
        experiment_matrix.get("feature_sets"),
        "experiment_matrix.feature_sets",
    )
    for candidate in selected_candidates:
        if candidate["model_id"] not in model_ids:
            raise ValueError(
                f"Candidate model_id={candidate['model_id']!r} is not listed in "
                "experiment_matrix.models."
            )
        if candidate["feature_set"] not in feature_set_ids:
            raise ValueError(
                f"Candidate feature_set={candidate['feature_set']!r} is not listed in "
                "experiment_matrix.feature_sets."
            )

    print(
        "Final map targets: "
        + json_string(
            {
                "sites": list(site_ids),
                "candidates": [
                    {
                        "candidate_label": candidate["candidate_label"],
                        "feature_set": candidate["feature_set"],
                        "model_id": candidate["model_id"],
                    }
                    for candidate in selected_candidates
                ],
                "output_root": str(final_maps_root),
                "dry_run": dry_run,
                "window_size": window_size,
                "no_preview": no_preview,
                "seed": effective_seed,
            }
        ),
        flush=True,
    )

    summary_paths: list[Path] = []
    for site_id in site_ids:
        if not isinstance(site_id, str):
            raise ValueError("experiment_matrix.sites must contain only strings.")
        site_config = get_site_config(sites_config, site_id)
        site_name = str(site_config.get("site_name", site_id))

        for candidate in selected_candidates:
            summary_path = process_site_candidate(
                project_id=project_id,
                site_id=site_id,
                site_name=site_name,
                candidate=candidate,
                candidate_display_label=display_labels.get(
                    str(candidate["candidate_label"]),
                    str(candidate["candidate_label"]),
                ),
                output_root=output_root,
                final_maps_root=final_maps_root,
                model_configs=model_configs,
                candidate_authority_path=candidate_path,
                seed=effective_seed,
                window_size=window_size,
                overwrite=overwrite,
                no_preview=no_preview,
                dry_run=dry_run,
            )
            if summary_path is not None:
                summary_paths.append(summary_path)

    return summary_paths


def regenerate_prediction_previews(
    project_id: str = DEFAULT_PROJECT_ID,
    *,
    site_filter: str | None = None,
    candidate_label: str | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = True,
) -> list[Path]:
    """Regenerate optional QC previews from frozen probability rasters only.

    This maintenance path deliberately does not load samples, estimators, or
    feature stacks and therefore cannot retrain a model or change a probability
    value.  Display metadata are read from the existing Stage 12 train summary.
    """

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    display_labels = publication_display_labels(
        load_publication_config(project_id=project_id)
    )
    output_dirs = ensure_output_dirs(project_config, stage_name="12")
    final_maps_root = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else output_dirs["output_root"] / "final_maps"
    )
    if not final_maps_root.is_dir():
        raise FileNotFoundError(f"Final-map root not found: {final_maps_root}")

    written: list[Path] = []
    for probability_path in sorted(final_maps_root.glob("*/*/prediction_probability.tif")):
        output_path = probability_path.parent
        site_id = output_path.parent.name
        directory_candidate = output_path.name
        if site_filter is not None and site_id != site_filter:
            continue
        summary_path = output_path / "train_summary.csv"
        _require_existing_file(summary_path, "Existing Stage 12 train summary")
        with summary_path.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError(f"Expected one Stage 12 train-summary row: {summary_path}")
        row = rows[0]
        candidate_id = row.get("candidate_label", directory_candidate)
        display_candidate = display_labels.get(candidate_id, candidate_id)
        if candidate_label is not None and candidate_label not in {
            directory_candidate,
            candidate_id,
            display_candidate,
        }:
            continue
        preview_path = output_path / "prediction_preview.png"
        if preview_path.exists() and not overwrite:
            raise FileExistsError(
                f"--no-overwrite was set and preview already exists: {preview_path}"
            )
        site = get_site_config(sites_config, site_id)
        resolved_font = _write_preview(
            probability_path,
            preview_path,
            site_name=str(site.get("site_name", site_id)),
            candidate_label=display_candidate,
            feature_set=row.get("feature_set", "unknown feature set"),
            model_id=row.get("model_id", "unknown model"),
        )
        print(
            f"Regenerated preview from frozen probability raster: {preview_path} "
            f"(resolved font: {resolved_font})",
            flush=True,
        )
        written.append(preview_path)
    if not written:
        raise FileNotFoundError(
            "No frozen probability rasters matched the requested preview filters."
        )
    return written


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train configured final site models and predict probability maps.",
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=f"Project id to process. Default: {DEFAULT_PROJECT_ID}",
    )
    parser.add_argument("--site", default=None, help="Optional site id filter.")
    parser.add_argument(
        "--candidate-label",
        default=None,
        help="Optional final candidate label filter.",
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        help="Optional custom feature set. Must be paired with --model-id.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional custom model id. Must be paired with --feature-set.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output root. Default: 05_outputs/<project_id>/final_maps",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip prediction_preview.png generation.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help=(
            "Regenerate prediction_preview.png from existing frozen probability "
            "rasters without training or prediction."
        ),
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing final-map outputs. Enabled by default.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print targets without training or writing outputs.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Optional square tile size. Default: raster internal block windows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional final-model seed override. Default: analysis.yaml final_mapping.seed.",
    )
    return parser.parse_args()


def main() -> None:
    """Run final map prediction from the command line."""

    args = parse_args()
    if args.preview_only:
        if args.no_preview:
            raise ValueError("--preview-only and --no-preview cannot be combined.")
        if args.feature_set is not None or args.model_id is not None:
            raise ValueError(
                "--preview-only reads feature/model metadata from train_summary.csv; "
                "do not combine it with --feature-set or --model-id."
            )
        regenerate_prediction_previews(
            project_id=args.project_id,
            site_filter=args.site,
            candidate_label=args.candidate_label,
            output_dir=args.output_dir,
            overwrite=bool(args.overwrite),
        )
        return
    summary_paths = run_prediction_maps(
        project_id=args.project_id,
        site_filter=args.site,
        candidate_label=args.candidate_label,
        feature_set=args.feature_set,
        model_id=args.model_id,
        output_dir=args.output_dir,
        no_preview=args.no_preview,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        window_size=args.window_size,
        seed=args.seed,
    )
    for summary_path in summary_paths:
        print(f"Final map output written to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create ML sample CSV files from feature stacks and vegetation labels.

This stage is the first stage that rasterizes vegetation label shapefiles. It
does not create spatial splits, train/test splits, thresholds, or models.
Feature stacks are read band-by-band so the full 3D stack is never held in
memory at once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    ensure_output_dirs,
    get_site_config,
    load_experiment_matrix,
    load_project_config,
    load_sites_config,
    resolve_repo_path,
)
from utils.raster_io import (
    get_raster_profile,
    read_band_names,
    read_raster,
    read_vector,
    reproject_vector,
)


SUMMARY_COLUMNS = [
    "site_id",
    "site_name",
    "feature_set",
    "stack_path",
    "bands_txt_path",
    "label_path",
    "samples_path",
    "width",
    "height",
    "crs",
    "resolution_m",
    "feature_count",
    "positive_field",
    "positive_value",
    "negative_value",
    "buffer_m",
    "all_touched",
    "clip_to_valid_feature_area",
    "clip_mode",
    "random_seed",
    "n_neg_per_pos",
    "finite_pixel_count",
    "label_total_features",
    "label_positive_features_before_clip",
    "label_positive_features_after_clip",
    "clipped_positive_features",
    "positive_raster_pixels",
    "positive_samples",
    "negative_candidate_pixels",
    "negative_samples",
    "target_negative_samples",
    "total_samples",
    "dropped_nan_inf_rows",
    "positive_fraction",
    "negative_fraction",
    "status",
    "message",
]
SAMPLE_PREFIX_COLUMNS = ["site_id", "feature_set", "x", "y", "label"]
COMMON_SAMPLE_COLUMNS = [
    "sample_id",
    "site_id",
    "x",
    "y",
    "raster_row",
    "raster_col",
    "label",
    "is_positive",
    "sampling_seed",
    "common_domain_policy",
]
COMMON_MATERIALIZED_PREFIX_COLUMNS = [
    "sample_id",
    "site_id",
    "feature_set",
    "x",
    "y",
    "raster_row",
    "raster_col",
    "label",
    "is_positive",
    "sampling_seed",
    "common_domain_policy",
]
NEGATIVE_SELECTION_CHUNK_SIZE = 5_000_000


def _require_pandas() -> Any:
    """Import pandas with a clear dependency error."""

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "pandas is required for sample CSV generation. Install pandas in "
            "the mbes_seaweed environment before running this stage."
        ) from exc
    return pd


def _require_rasterio_tools() -> tuple[Any, Any]:
    """Import rasterio rasterize and xy helpers with a clear error."""

    try:
        from rasterio.features import rasterize  # type: ignore[import-not-found]
        from rasterio.transform import xy  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "rasterio is required for label rasterization and coordinate "
            "calculation. Install rasterio in the mbes_seaweed environment "
            "before running this stage."
        ) from exc
    return rasterize, xy


def _require_rowcol() -> Any:
    """Import rasterio row/col transform helper with a clear error."""

    try:
        from rasterio.transform import rowcol  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "rasterio is required for label-to-raster coordinate filtering. "
            "Install rasterio in the mbes_seaweed environment before running this stage."
        ) from exc
    return rowcol


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


def _require_existing_file(path: str | Path, label: str) -> Path:
    """Return an existing file path or raise ``FileNotFoundError``."""

    checked = Path(path).expanduser()
    if not checked.exists():
        raise FileNotFoundError(f"{label} not found: {checked}")
    if not checked.is_file():
        raise ValueError(f"{label} is not a file: {checked}")
    return checked


def _resolution_from_profile(profile: Mapping[str, Any]) -> float | str:
    """Return raster resolution from a profile."""

    transform = profile.get("transform")
    if hasattr(transform, "a"):
        return abs(float(transform.a))
    return ""


def _as_integer_label(value: Any, name: str) -> int:
    """Return ``value`` as an integer label value."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer-like label value.") from exc
        if parsed.is_integer():
            return int(parsed)
    raise ValueError(f"{name} must be an integer-like label value.")


def _series_equals_config_value(series: Any, value: Any) -> Any:
    """Compare a GeoDataFrame column to a config value with a string fallback."""

    direct = series == value
    try:
        if bool(direct.any()):
            return direct
    except AttributeError:
        pass
    return series.astype(str) == str(value)


def _finite_mask_for_array(array: Any, nodata: int | float | None) -> np.ndarray:
    """Return finite valid pixels for one 2D raster band."""

    if np.ma.isMaskedArray(array):
        data = np.asarray(array.data)
        valid = ~np.ma.getmaskarray(array)
    else:
        data = np.asarray(array)
        valid = np.ones(data.shape, dtype=bool)

    valid &= np.isfinite(data)
    if nodata is not None:
        try:
            if not np.isnan(nodata):
                valid &= data != nodata
        except TypeError:
            valid &= data != nodata
    return valid


def _array_to_float32_nan(array: Any, nodata: int | float | None) -> np.ndarray:
    """Return a 2D raster band as float32 with invalid values as NaN."""

    if np.ma.isMaskedArray(array):
        data = np.ma.asarray(array, dtype="float32").filled(np.nan)
    else:
        data = np.asarray(array, dtype="float32")
        if not data.flags.writeable:
            data = data.copy()

    if nodata is not None:
        try:
            if not np.isnan(nodata):
                data[data == nodata] = np.nan
        except TypeError:
            data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data


def _read_sampling_config(project_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated sampling settings from project config."""

    sampling = _require_mapping(project_config.get("sampling"), "project_config.sampling")

    random_seed = sampling.get("random_seed")
    n_neg_per_pos = sampling.get("n_neg_per_pos")
    drop_nan_inf = sampling.get("drop_nan_inf")

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("sampling.random_seed must be an integer.")
    if isinstance(n_neg_per_pos, bool) or not isinstance(n_neg_per_pos, (int, float)):
        raise ValueError("sampling.n_neg_per_pos must be numeric.")
    if float(n_neg_per_pos) <= 0:
        raise ValueError("sampling.n_neg_per_pos must be >= 0.")
    if not isinstance(drop_nan_inf, bool):
        raise ValueError("sampling.drop_nan_inf must be a bool.")

    mode = sampling.get("mode", "feature_set_specific")
    if mode not in {"feature_set_specific", "site_level_common"}:
        raise ValueError(
            "sampling.mode must be 'feature_set_specific' or 'site_level_common'."
        )

    result = {
        "mode": mode,
        "random_seed": int(random_seed),
        "n_neg_per_pos": float(n_neg_per_pos),
        "drop_nan_inf": drop_nan_inf,
    }
    if mode == "site_level_common":
        common = _require_mapping(sampling.get("common_sample"), "project_config.sampling.common_sample")
        required = [
            "policy_id",
            "common_mask_pattern",
            "authority_directory",
            "reuse_across_feature_sets",
            "non_finite_feature_policy",
        ]
        missing = [key for key in required if key not in common]
        if missing:
            raise ValueError(
                "project_config.sampling.common_sample is missing required key(s): "
                + ", ".join(missing)
            )
        if common["reuse_across_feature_sets"] is not True:
            raise ValueError("site_level_common mode requires reuse_across_feature_sets=true.")
        if common["non_finite_feature_policy"] != "fail":
            raise ValueError("site_level_common mode requires non_finite_feature_policy='fail'.")
        result["common_sample"] = dict(common)
    return result


def _read_site_label_config(site_id: str, site_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return label path and label schema for one site."""

    paths = _require_mapping(site_config.get("paths"), f"sites.{site_id}.paths")
    label = _require_mapping(site_config.get("label"), f"sites.{site_id}.label")

    required = ["positive_field", "positive_value", "negative_value"]
    missing = [key for key in required if key not in label]
    if missing:
        raise ValueError(
            f"sites.{site_id}.label is missing required key(s): {', '.join(missing)}."
        )
    if "label_path" not in paths:
        raise ValueError(f"sites.{site_id}.paths is missing required key: label_path.")

    positive_value = _as_integer_label(label["positive_value"], f"sites.{site_id}.label.positive_value")
    negative_value = _as_integer_label(label["negative_value"], f"sites.{site_id}.label.negative_value")
    if positive_value == negative_value:
        raise ValueError(f"sites.{site_id}.label positive_value and negative_value must differ.")

    buffer_m = label.get("buffer_m", 0.0)
    if isinstance(buffer_m, bool) or not isinstance(buffer_m, (int, float)):
        raise ValueError(f"sites.{site_id}.label.buffer_m must be numeric.")
    if float(buffer_m) < 0:
        raise ValueError(f"sites.{site_id}.label.buffer_m must be >= 0.")

    all_touched = label.get("all_touched", True)
    if not isinstance(all_touched, bool):
        raise ValueError(f"sites.{site_id}.label.all_touched must be a bool.")

    clip_to_valid_feature_area = label.get("clip_to_valid_feature_area", False)
    if not isinstance(clip_to_valid_feature_area, bool):
        raise ValueError(f"sites.{site_id}.label.clip_to_valid_feature_area must be a bool.")

    clip_mode = label.get("clip_mode", "none")
    if not isinstance(clip_mode, str):
        raise ValueError(f"sites.{site_id}.label.clip_mode must be a string.")
    if clip_to_valid_feature_area and clip_mode != "finite_mask":
        raise ValueError(
            f"sites.{site_id}.label.clip_mode must be 'finite_mask' when "
            "clip_to_valid_feature_area is true."
        )

    return {
        "label_path": _require_existing_file(
            resolve_repo_path(paths["label_path"]),
            "Label shapefile",
        ),
        "positive_field": str(label["positive_field"]),
        "positive_value_raw": label["positive_value"],
        "negative_value_raw": label["negative_value"],
        "positive_value": positive_value,
        "negative_value": negative_value,
        "buffer_m": float(buffer_m),
        "all_touched": all_touched,
        "clip_to_valid_feature_area": clip_to_valid_feature_area,
        "clip_mode": clip_mode,
    }


def _validate_stack_and_bands(stack_path: Path, bands_txt_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Validate feature stack and bands.txt consistency."""

    _require_existing_file(stack_path, "Feature stack")
    _require_existing_file(bands_txt_path, "Band names file")

    profile = get_raster_profile(stack_path)
    band_names = read_band_names(bands_txt_path)
    stack_count = int(profile.get("count", 0))
    if stack_count != len(band_names):
        raise ValueError(
            f"Stack band count does not match bands.txt: stack={stack_count}, "
            f"bands.txt={len(band_names)}, stack_path={stack_path}, bands_txt_path={bands_txt_path}"
        )

    duplicates = sorted({name for name in band_names if band_names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate band name(s) in {bands_txt_path}: {', '.join(duplicates)}"
        )

    return profile, band_names


def _rasterize_positive_labels(
    *,
    site_id: str,
    feature_set_id: str,
    label_config: Mapping[str, Any],
    stack_profile: Mapping[str, Any],
    finite_mask: np.ndarray,
) -> tuple[np.ndarray, int, dict[str, int]]:
    """Filter positive point labels and rasterize them as a boolean mask."""

    rasterize, _ = _require_rasterio_tools()
    rowcol = _require_rowcol()
    label_path = Path(label_config["label_path"])
    positive_field = str(label_config["positive_field"])
    positive_value_raw = label_config["positive_value_raw"]
    buffer_m = float(label_config["buffer_m"])
    all_touched = bool(label_config["all_touched"])
    clip_to_valid_feature_area = bool(label_config["clip_to_valid_feature_area"])
    clip_mode = str(label_config["clip_mode"])

    if clip_to_valid_feature_area and clip_mode != "finite_mask":
        raise ValueError(
            f"Unsupported clip_mode={clip_mode!r}: only 'finite_mask' is supported "
            "when clip_to_valid_feature_area is true."
        )

    gdf = read_vector(label_path)
    if getattr(gdf, "crs", None) is None:
        raise ValueError(
            f"Label shapefile has no CRS: site_id={site_id}, "
            f"feature_set={feature_set_id}, label_path={label_path}"
        )
    gdf = reproject_vector(gdf, stack_profile["crs"])
    if positive_field not in gdf.columns:
        raise ValueError(
            f"positive_field not found in label shapefile: site_id={site_id}, "
            f"feature_set={feature_set_id}, positive_field={positive_field}, "
            f"label_path={label_path}"
        )

    positive_rows = _series_equals_config_value(gdf[positive_field], positive_value_raw)
    positive_gdf = gdf[positive_rows]
    label_total_features = int(len(gdf))
    positive_before_clip = int(len(positive_gdf))
    point_geometries = []
    for index, geom in positive_gdf.geometry.items():
        if geom is None or geom.is_empty:
            continue
        if getattr(geom, "geom_type", None) != "Point":
            raise ValueError(
                f"Positive label geometry must be Point for valid-area clipping: "
                f"site_id={site_id}, feature_set={feature_set_id}, "
                f"label_path={label_path}, row_index={index}, "
                f"geometry_type={getattr(geom, 'geom_type', type(geom).__name__)}"
            )
        point_geometries.append(geom)

    filtered_geometries = []
    for geom in point_geometries:
        keep = True
        if clip_to_valid_feature_area:
            rows, cols = rowcol(
                stack_profile["transform"],
                [float(geom.x)],
                [float(geom.y)],
            )
            row = int(rows[0])
            col = int(cols[0])
            keep = (
                0 <= row < finite_mask.shape[0]
                and 0 <= col < finite_mask.shape[1]
                and bool(finite_mask[row, col])
            )
        if keep:
            filtered_geometries.append(geom)

    positive_after_clip = int(len(filtered_geometries))
    label_stats = {
        "label_total_features": label_total_features,
        "label_positive_features_before_clip": positive_before_clip,
        "label_positive_features_after_clip": positive_after_clip,
        "clipped_positive_features": positive_before_clip - positive_after_clip,
    }

    geometries = filtered_geometries
    if buffer_m > 0:
        buffered_geometries = []
        for geom in geometries:
            buffered = geom.buffer(buffer_m)
            if buffered is not None and not buffered.is_empty:
                buffered_geometries.append(buffered)
        geometries = buffered_geometries
    if not geometries:
        raise ValueError(
            f"No positive geometry remains after valid-area filtering: "
            f"site_id={site_id}, feature_set={feature_set_id}, "
            f"label_path={label_path}, positive_field={positive_field}, "
            f"positive_value={positive_value_raw}"
        )

    positive_mask = rasterize(
        ((geom, 1) for geom in geometries),
        out_shape=(int(stack_profile["height"]), int(stack_profile["width"])),
        transform=stack_profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    ).astype(bool, copy=False)
    positive_raster_pixels = int(np.count_nonzero(positive_mask))
    if positive_raster_pixels == 0:
        raise ValueError(
            f"Rasterized positive pixel count is zero: site_id={site_id}, "
            f"feature_set={feature_set_id}, label_path={label_path}"
        )

    return positive_mask, positive_raster_pixels, label_stats


def _build_finite_mask(stack_path: Path, feature_count: int, height: int, width: int) -> np.ndarray:
    """Build a common finite mask by reading the stack one band at a time."""

    finite_mask = np.ones((height, width), dtype=bool)
    for band_index in range(1, feature_count + 1):
        array, profile = read_raster(stack_path, band=band_index, masked=True)
        finite_mask &= _finite_mask_for_array(array, profile.get("nodata"))
    return finite_mask


def _sample_mask_indices(
    mask: np.ndarray,
    target_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return random flat indices from ``mask`` without materializing all candidates."""

    candidate_count = int(np.count_nonzero(mask))
    if target_count >= candidate_count:
        return np.flatnonzero(mask)
    if target_count <= 0:
        return np.array([], dtype=np.int64)

    selected_ordinals = np.sort(rng.choice(candidate_count, size=target_count, replace=False))
    selected = np.empty(target_count, dtype=np.int64)

    flat_mask = mask.ravel()
    seen = 0
    written = 0
    for start in range(0, flat_mask.size, NEGATIVE_SELECTION_CHUNK_SIZE):
        stop = min(start + NEGATIVE_SELECTION_CHUNK_SIZE, flat_mask.size)
        chunk = flat_mask[start:stop]
        chunk_count = int(np.count_nonzero(chunk))
        if chunk_count == 0:
            continue

        upper = seen + chunk_count
        next_written = written
        while next_written < target_count and selected_ordinals[next_written] < upper:
            next_written += 1

        if next_written > written:
            chunk_indices = np.flatnonzero(chunk)
            local_ordinals = selected_ordinals[written:next_written] - seen
            selected[written:next_written] = start + chunk_indices[local_ordinals]
            written = next_written

        seen = upper
        if written == target_count:
            break

    if written != target_count:
        raise RuntimeError(
            f"Internal sampling error: selected {written} indices, expected {target_count}."
        )
    return selected


def _extract_band_values(
    stack_path: Path,
    band_index: int,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Read one stack band and extract selected flat-index values."""

    array, profile = read_raster(stack_path, band=band_index, masked=True)
    values = _array_to_float32_nan(array, profile.get("nodata")).ravel()[sample_indices]
    return values.astype("float32", copy=False)


def _coordinates_for_indices(
    transform: Any,
    height: int,
    width: int,
    sample_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel-centre x/y coordinates for flat sample indices."""

    _, xy = _require_rasterio_tools()
    rows, cols = np.unravel_index(sample_indices, (height, width))
    x_values, y_values = xy(transform, rows, cols, offset="center")
    return np.asarray(x_values, dtype="float64"), np.asarray(y_values, dtype="float64")


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_hash(frame: Any, columns: Sequence[str]) -> str:
    """Hash selected dataframe columns using deterministic CSV serialization."""

    payload = frame[list(columns)].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_common_sample_mask(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a site-level common-valid mask without changing its grid."""

    array, profile = read_raster(path, band=1, masked=True)
    mask = _finite_mask_for_array(array, profile.get("nodata"))
    values = np.ma.asarray(array).filled(0) if np.ma.isMaskedArray(array) else np.asarray(array)
    mask &= values.astype("float32") > 0
    if not mask.any():
        raise ValueError(f"Site-level common-valid mask contains zero valid pixels: {path}")
    return mask, profile


def create_site_common_sample_authority(
    *,
    project_id: str,
    site_id: str,
    site_config: Mapping[str, Any],
    common_mask_path: Path,
    authority_root: Path,
    sampling_config: Mapping[str, Any],
) -> Path:
    """Draw and persist one deterministic sample authority for a site."""

    pd = _require_pandas()
    common_cfg = _require_mapping(
        sampling_config.get("common_sample"),
        "sampling_config.common_sample",
    )
    mask_path = _require_existing_file(common_mask_path, "Site-level common-valid mask")
    common_mask, profile = _read_common_sample_mask(mask_path)
    height, width = common_mask.shape
    label_config = _read_site_label_config(site_id, site_config)
    positive_geometry_mask, positive_raster_pixels, label_stats = _rasterize_positive_labels(
        site_id=site_id,
        feature_set_id="site_level_common",
        label_config=label_config,
        stack_profile=profile,
        finite_mask=common_mask,
    )

    positive_indices = np.flatnonzero(positive_geometry_mask & common_mask)
    positive_count = int(positive_indices.size)
    if positive_count == 0:
        raise ValueError(f"Positive sample count is zero for site-level authority: {site_id}")

    negative_mask = (~positive_geometry_mask) & common_mask
    negative_candidate_count = int(np.count_nonzero(negative_mask))
    target_negative = int(round(float(sampling_config["n_neg_per_pos"]) * positive_count))
    if negative_candidate_count < target_negative:
        raise ValueError(
            "Site-level common-valid domain has fewer negative candidates than required: "
            f"site_id={site_id}, candidates={negative_candidate_count}, target={target_negative}"
        )

    seed = int(sampling_config["random_seed"])
    rng = np.random.default_rng(seed)
    negative_indices = _sample_mask_indices(negative_mask, target_negative, rng)
    labels = np.concatenate(
        [
            np.full(positive_count, int(label_config["positive_value"]), dtype=np.int16),
            np.full(target_negative, int(label_config["negative_value"]), dtype=np.int16),
        ]
    )
    sample_indices = np.concatenate([positive_indices, negative_indices])
    order = rng.permutation(sample_indices.size)
    sample_indices = sample_indices[order]
    labels = labels[order]
    rows, cols = np.unravel_index(sample_indices, (height, width))
    x_values, y_values = _coordinates_for_indices(
        profile["transform"],
        height,
        width,
        sample_indices,
    )
    sample_ids = np.asarray(
        [f"{site_id}:r{row:06d}:c{col:06d}:l{label}" for row, col, label in zip(rows, cols, labels)],
        dtype=object,
    )
    if len(set(sample_ids.tolist())) != sample_ids.size:
        raise ValueError(f"Duplicate deterministic sample_id generated for site {site_id}.")

    policy_id = str(common_cfg["policy_id"])
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "site_id": np.full(sample_indices.size, site_id, dtype=object),
            "x": x_values,
            "y": y_values,
            "raster_row": rows.astype("int64"),
            "raster_col": cols.astype("int64"),
            "label": labels,
            "is_positive": labels == int(label_config["positive_value"]),
            "sampling_seed": np.full(sample_indices.size, seed, dtype="int64"),
            "common_domain_policy": np.full(sample_indices.size, policy_id, dtype=object),
        },
        columns=COMMON_SAMPLE_COLUMNS,
    )
    site_dir = authority_root / site_id
    site_dir.mkdir(parents=True, exist_ok=True)
    authority_path = site_dir / "sample_index.csv"
    frame.to_csv(authority_path, index=False)

    provenance = {
        "schema_version": 1,
        "project_id": project_id,
        "site_id": site_id,
        "sampling_policy": "site_level_common",
        "common_domain_policy": policy_id,
        "sampling_seed": seed,
        "positive_buffer_m": float(label_config["buffer_m"]),
        "all_touched": bool(label_config["all_touched"]),
        "negative_to_positive_ratio": float(sampling_config["n_neg_per_pos"]),
        "common_valid_mask": str(mask_path),
        "common_valid_mask_sha256": _sha256_file(mask_path),
        "source_reference_labels": str(label_config["label_path"]),
        "source_reference_labels_sha256": _sha256_file(Path(label_config["label_path"])),
        "row_count": int(len(frame)),
        "positive_count": int(np.count_nonzero(frame["label"] == int(label_config["positive_value"]))),
        "negative_count": int(np.count_nonzero(frame["label"] == int(label_config["negative_value"]))),
        "negative_candidate_count": negative_candidate_count,
        "positive_raster_pixels": positive_raster_pixels,
        "common_valid_count": int(np.count_nonzero(common_mask)),
        "label_stats": label_stats,
        "grid": {
            "width": int(profile["width"]),
            "height": int(profile["height"]),
            "crs": str(profile["crs"]),
            "transform": list(profile["transform"]),
        },
        "coordinate_hash": _frame_hash(
            frame,
            ["sample_id", "x", "y", "raster_row", "raster_col"],
        ),
        "label_hash": _frame_hash(frame, ["sample_id", "label"]),
        "sample_table_sha256": _sha256_file(authority_path),
    }
    provenance_path = site_dir / "sample_index.provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return authority_path


def materialize_feature_set_from_common_authority(
    *,
    site_id: str,
    site_config: Mapping[str, Any],
    feature_set_id: str,
    feature_sets_root: Path,
    samples_root: Path,
    authority_path: Path,
    sampling_config: Mapping[str, Any],
) -> Path:
    """Attach one feature stack to a frozen site-level sample authority."""

    pd = _require_pandas()
    authority_file = _require_existing_file(authority_path, "Site-level sample authority")
    authority = pd.read_csv(authority_file)
    missing = [column for column in COMMON_SAMPLE_COLUMNS if column not in authority.columns]
    if missing:
        raise ValueError(
            "Site-level sample authority is missing required column(s): " + ", ".join(missing)
        )
    if authority.empty:
        raise ValueError(f"Site-level sample authority is empty: {authority_file}")
    if not (authority["site_id"].astype(str) == site_id).all():
        raise ValueError(f"Site-level sample authority contains a different site: {authority_file}")
    if authority["sample_id"].duplicated().any():
        raise ValueError(f"Site-level sample authority contains duplicate sample_id: {authority_file}")

    stack_path = feature_sets_root / site_id / f"{feature_set_id}.tif"
    bands_txt_path = feature_sets_root / site_id / f"{feature_set_id}.bands.txt"
    profile, band_names = _validate_stack_and_bands(stack_path, bands_txt_path)
    height, width = int(profile["height"]), int(profile["width"])
    rows = authority["raster_row"].to_numpy(dtype="int64", copy=False)
    cols = authority["raster_col"].to_numpy(dtype="int64", copy=False)
    if not np.all((rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)):
        raise ValueError(
            f"Site-level sample authority contains coordinates outside stack grid: {stack_path}"
        )
    sample_indices = np.ravel_multi_index((rows, cols), (height, width))
    expected_x, expected_y = _coordinates_for_indices(
        profile["transform"],
        height,
        width,
        sample_indices,
    )
    if not np.allclose(authority["x"].to_numpy(dtype="float64"), expected_x, rtol=0, atol=1e-8):
        raise ValueError(f"Authority x coordinates do not match feature-stack grid: {stack_path}")
    if not np.allclose(authority["y"].to_numpy(dtype="float64"), expected_y, rtol=0, atol=1e-8):
        raise ValueError(f"Authority y coordinates do not match feature-stack grid: {stack_path}")

    feature_values: dict[str, np.ndarray] = {}
    for band_index, band_name in enumerate(band_names, start=1):
        values = _extract_band_values(stack_path, band_index, sample_indices)
        invalid_count = int(np.count_nonzero(~np.isfinite(values)))
        if invalid_count:
            raise ValueError(
                "Non-finite feature values at frozen common sample coordinates: "
                f"site_id={site_id}, feature_set={feature_set_id}, "
                f"feature={band_name}, invalid_count={invalid_count}"
            )
        feature_values[band_name] = values

    labels = authority["label"].to_numpy(dtype="int16", copy=False)
    materialized = authority.copy()
    materialized.insert(2, "feature_set", feature_set_id)
    for name, values in feature_values.items():
        materialized[name] = values
    materialized = materialized[COMMON_MATERIALIZED_PREFIX_COLUMNS + band_names]
    samples_dir = samples_root / site_id
    samples_dir.mkdir(parents=True, exist_ok=True)
    samples_path = samples_dir / f"{feature_set_id}_samples.csv"
    materialized.to_csv(samples_path, index=False)

    provenance_path = authority_file.with_name("sample_index.provenance.json")
    provenance = json.loads(_require_existing_file(provenance_path, "Sample authority provenance").read_text())
    label_config = _read_site_label_config(site_id, site_config)
    positive_count = int(np.count_nonzero(labels == int(label_config["positive_value"])))
    negative_count = int(np.count_nonzero(labels == int(label_config["negative_value"])))
    summary_row = {
        "site_id": site_id,
        "site_name": str(site_config.get("site_name", site_id)),
        "feature_set": feature_set_id,
        "stack_path": str(stack_path),
        "bands_txt_path": str(bands_txt_path),
        "label_path": str(label_config["label_path"]),
        "samples_path": str(samples_path),
        "width": width,
        "height": height,
        "crs": str(profile["crs"]),
        "resolution_m": _resolution_from_profile(profile),
        "feature_count": len(band_names),
        "positive_field": label_config["positive_field"],
        "positive_value": label_config["positive_value"],
        "negative_value": label_config["negative_value"],
        "buffer_m": label_config["buffer_m"],
        "all_touched": label_config["all_touched"],
        "clip_to_valid_feature_area": label_config["clip_to_valid_feature_area"],
        "clip_mode": label_config["clip_mode"],
        "random_seed": int(sampling_config["random_seed"]),
        "n_neg_per_pos": sampling_config["n_neg_per_pos"],
        "finite_pixel_count": provenance["common_valid_count"],
        "label_total_features": provenance["label_stats"]["label_total_features"],
        "label_positive_features_before_clip": provenance["label_stats"]["label_positive_features_before_clip"],
        "label_positive_features_after_clip": provenance["label_stats"]["label_positive_features_after_clip"],
        "clipped_positive_features": provenance["label_stats"]["clipped_positive_features"],
        "positive_raster_pixels": provenance["positive_raster_pixels"],
        "positive_samples": positive_count,
        "negative_candidate_pixels": provenance["negative_candidate_count"],
        "negative_samples": negative_count,
        "target_negative_samples": negative_count,
        "total_samples": int(len(materialized)),
        "dropped_nan_inf_rows": 0,
        "positive_fraction": positive_count / len(materialized),
        "negative_fraction": negative_count / len(materialized),
        "status": "ok",
        "message": "materialized from site-level common sample authority",
    }
    summary_path = samples_dir / f"{feature_set_id}_sample_summary.csv"
    _write_sample_summary(summary_path, summary_row)
    return summary_path


def _write_sample_summary(path: Path, row: Mapping[str, Any]) -> None:
    """Write one sample summary CSV with one row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def process_feature_set_samples(
    *,
    site_id: str,
    site_config: Mapping[str, Any],
    feature_set_id: str,
    feature_sets_root: Path,
    samples_root: Path,
    sampling_config: Mapping[str, Any],
) -> Path:
    """Create samples for one site and feature set."""

    pd = _require_pandas()

    print(f"Sampling: {site_id} / {feature_set_id}", flush=True)
    site_name = str(site_config.get("site_name", site_id))
    stack_path = feature_sets_root / site_id / f"{feature_set_id}.tif"
    bands_txt_path = feature_sets_root / site_id / f"{feature_set_id}.bands.txt"
    stack_profile, band_names = _validate_stack_and_bands(stack_path, bands_txt_path)

    height = int(stack_profile["height"])
    width = int(stack_profile["width"])
    feature_count = len(band_names)
    label_config = _read_site_label_config(site_id, site_config)

    print(f"Building finite mask: {site_id} / {feature_set_id}", flush=True)
    finite_mask = _build_finite_mask(stack_path, feature_count, height, width)
    finite_pixel_count = int(np.count_nonzero(finite_mask))
    print(
        f"Built finite mask: {site_id} / {feature_set_id} / "
        f"finite_pixels={finite_pixel_count}",
        flush=True,
    )

    print(f"Clipping/rasterizing labels: {site_id} / {feature_set_id}", flush=True)
    positive_geometry_mask, positive_raster_pixels, label_stats = _rasterize_positive_labels(
        site_id=site_id,
        feature_set_id=feature_set_id,
        label_config=label_config,
        stack_profile=stack_profile,
        finite_mask=finite_mask,
    )
    print(
        f"Clipped/rasterized labels: {site_id} / {feature_set_id} / "
        f"label_positive_features_before_clip="
        f"{label_stats['label_positive_features_before_clip']} / "
        f"label_positive_features_after_clip="
        f"{label_stats['label_positive_features_after_clip']} / "
        f"positive_raster_pixels={positive_raster_pixels}",
        flush=True,
    )

    positive_value = int(label_config["positive_value"])
    negative_value = int(label_config["negative_value"])
    positive_mask = positive_geometry_mask & finite_mask
    positive_indices = np.flatnonzero(positive_mask)
    positive_count = int(positive_indices.size)
    if positive_count == 0:
        raise ValueError(
            f"Positive sample count is zero after finite-mask filtering: "
            f"site_id={site_id}, feature_set={feature_set_id}, "
            f"label_path={label_config['label_path']}, "
            f"positive_field={label_config['positive_field']}, "
            f"positive_value={label_config['positive_value_raw']}"
        )

    negative_mask = (~positive_geometry_mask) & finite_mask
    negative_candidate_count = int(np.count_nonzero(negative_mask))
    target_negative = int(round(float(sampling_config["n_neg_per_pos"]) * positive_count))
    warning_messages: list[str] = []
    if negative_candidate_count >= target_negative:
        negative_target_to_draw = target_negative
    else:
        negative_target_to_draw = negative_candidate_count
        warning_messages.append("negative candidates fewer than target")

    rng = np.random.default_rng(int(sampling_config["random_seed"]))
    negative_indices = _sample_mask_indices(
        negative_mask,
        negative_target_to_draw,
        rng,
    )
    negative_count = int(negative_indices.size)
    if negative_count == 0:
        raise ValueError(
            f"Negative sample count is zero: site_id={site_id}, "
            f"feature_set={feature_set_id}, target_negative_samples={target_negative}, "
            f"negative_candidate_pixels={negative_candidate_count}"
        )

    sample_indices = np.concatenate([positive_indices, negative_indices])
    sample_labels = np.concatenate(
        [
            np.full(positive_count, positive_value, dtype=np.int16),
            np.full(negative_count, negative_value, dtype=np.int16),
        ]
    )
    order = rng.permutation(sample_indices.size)
    sample_indices = sample_indices[order]
    sample_labels = sample_labels[order]

    x_values, y_values = _coordinates_for_indices(
        stack_profile["transform"],
        height,
        width,
        sample_indices,
    )

    feature_values: dict[str, np.ndarray] = {}
    final_valid = np.ones(sample_indices.size, dtype=bool)
    for band_index, band_name in enumerate(band_names, start=1):
        values = _extract_band_values(stack_path, band_index, sample_indices)
        final_valid &= np.isfinite(values)
        feature_values[band_name] = values

    dropped_nan_inf_rows = 0
    if bool(sampling_config["drop_nan_inf"]):
        dropped_nan_inf_rows = int(sample_indices.size - np.count_nonzero(final_valid))
        if dropped_nan_inf_rows:
            sample_indices = sample_indices[final_valid]
            sample_labels = sample_labels[final_valid]
            x_values = x_values[final_valid]
            y_values = y_values[final_valid]
            for band_name in band_names:
                feature_values[band_name] = feature_values[band_name][final_valid]
            warning_messages.append("dropped non-finite feature rows")
    elif not bool(np.all(final_valid)):
        warning_messages.append("non-finite feature rows retained because drop_nan_inf is false")

    total_samples = int(sample_labels.size)
    if total_samples == 0:
        raise ValueError(f"Final sample count is zero: site_id={site_id}, feature_set={feature_set_id}")

    final_positive_count = int(np.count_nonzero(sample_labels == positive_value))
    final_negative_count = int(np.count_nonzero(sample_labels == negative_value))

    samples_dir = samples_root / site_id
    samples_dir.mkdir(parents=True, exist_ok=True)
    samples_path = samples_dir / f"{feature_set_id}_samples.csv"
    summary_path = samples_dir / f"{feature_set_id}_sample_summary.csv"

    sample_data: dict[str, Any] = {
        "site_id": np.full(total_samples, site_id, dtype=object),
        "feature_set": np.full(total_samples, feature_set_id, dtype=object),
        "x": x_values,
        "y": y_values,
        "label": sample_labels,
    }
    sample_data.update(feature_values)
    columns = SAMPLE_PREFIX_COLUMNS + band_names
    pd.DataFrame(sample_data, columns=columns).to_csv(samples_path, index=False)
    print(f"Sample CSV written to: {samples_path}", flush=True)

    status = "warning" if warning_messages else "ok"
    message = "; ".join(warning_messages)
    summary_row = {
        "site_id": site_id,
        "site_name": site_name,
        "feature_set": feature_set_id,
        "stack_path": str(stack_path),
        "bands_txt_path": str(bands_txt_path),
        "label_path": str(label_config["label_path"]),
        "samples_path": str(samples_path),
        "width": width,
        "height": height,
        "crs": str(stack_profile["crs"]),
        "resolution_m": _resolution_from_profile(stack_profile),
        "feature_count": feature_count,
        "positive_field": label_config["positive_field"],
        "positive_value": positive_value,
        "negative_value": negative_value,
        "buffer_m": label_config["buffer_m"],
        "all_touched": label_config["all_touched"],
        "clip_to_valid_feature_area": label_config["clip_to_valid_feature_area"],
        "clip_mode": label_config["clip_mode"],
        "random_seed": int(sampling_config["random_seed"]),
        "n_neg_per_pos": sampling_config["n_neg_per_pos"],
        "finite_pixel_count": finite_pixel_count,
        "label_total_features": label_stats["label_total_features"],
        "label_positive_features_before_clip": label_stats["label_positive_features_before_clip"],
        "label_positive_features_after_clip": label_stats["label_positive_features_after_clip"],
        "clipped_positive_features": label_stats["clipped_positive_features"],
        "positive_raster_pixels": positive_raster_pixels,
        "positive_samples": final_positive_count,
        "negative_candidate_pixels": negative_candidate_count,
        "negative_samples": final_negative_count,
        "target_negative_samples": target_negative,
        "total_samples": total_samples,
        "dropped_nan_inf_rows": dropped_nan_inf_rows,
        "positive_fraction": final_positive_count / total_samples,
        "negative_fraction": final_negative_count / total_samples,
        "status": status,
        "message": message,
    }
    _write_sample_summary(summary_path, summary_row)
    print(
        f"Sampled: {site_id} / {feature_set_id} / "
        f"label_positive_features_before_clip="
        f"{label_stats['label_positive_features_before_clip']} / "
        f"label_positive_features_after_clip="
        f"{label_stats['label_positive_features_after_clip']} / "
        f"positive_samples={final_positive_count} / "
        f"total_samples={total_samples}",
        flush=True,
    )
    return summary_path


def run_sampling(project_id: str = DEFAULT_PROJECT_ID) -> list[Path]:
    """Create sample CSV files for all configured site/feature-set combinations."""

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    experiment_matrix = load_experiment_matrix(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="04")
    sampling_config = _read_sampling_config(project_config)

    site_ids = _require_list(experiment_matrix.get("sites"), "experiment_matrix.sites")
    feature_set_ids = _require_list(
        experiment_matrix.get("feature_sets"),
        "experiment_matrix.feature_sets",
    )

    summary_paths: list[Path] = []
    if sampling_config["mode"] == "site_level_common":
        common_cfg = _require_mapping(
            sampling_config.get("common_sample"),
            "sampling_config.common_sample",
        )
        authority_directory = Path(str(common_cfg["authority_directory"]))
        if authority_directory.is_absolute() or ".." in authority_directory.parts:
            raise ValueError("sampling.common_sample.authority_directory must be output-root-relative.")
        authority_root = output_dirs["output_root"] / authority_directory
        mask_pattern = str(common_cfg["common_mask_pattern"])
        if "{site_id}" not in mask_pattern:
            raise ValueError("sampling.common_sample.common_mask_pattern must contain '{site_id}'.")

        for site_id in site_ids:
            if not isinstance(site_id, str):
                raise ValueError("experiment_matrix.sites must contain only strings.")
            site_config = get_site_config(sites_config, site_id)
            common_mask_path = resolve_repo_path(mask_pattern.format(site_id=site_id))
            authority_path = create_site_common_sample_authority(
                project_id=project_id,
                site_id=site_id,
                site_config=site_config,
                common_mask_path=common_mask_path,
                authority_root=authority_root,
                sampling_config=sampling_config,
            )
            for feature_set_id in feature_set_ids:
                if not isinstance(feature_set_id, str):
                    raise ValueError("experiment_matrix.feature_sets must contain only strings.")
                summary_paths.append(
                    materialize_feature_set_from_common_authority(
                        site_id=site_id,
                        site_config=site_config,
                        feature_set_id=feature_set_id,
                        feature_sets_root=output_dirs["feature_sets"],
                        samples_root=output_dirs["samples"],
                        authority_path=authority_path,
                        sampling_config=sampling_config,
                    )
                )
        return summary_paths

    for site_id in site_ids:
        if not isinstance(site_id, str):
            raise ValueError("experiment_matrix.sites must contain only strings.")
        site_config = get_site_config(sites_config, site_id)

        for feature_set_id in feature_set_ids:
            if not isinstance(feature_set_id, str):
                raise ValueError("experiment_matrix.feature_sets must contain only strings.")
            summary_paths.append(
                process_feature_set_samples(
                    site_id=site_id,
                    site_config=site_config,
                    feature_set_id=feature_set_id,
                    feature_sets_root=output_dirs["feature_sets"],
                    samples_root=output_dirs["samples"],
                    sampling_config=sampling_config,
                )
            )

    return summary_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create configured project ML sample CSV files from feature stacks and labels.",
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=f"Project id to process. Default: {DEFAULT_PROJECT_ID}",
    )
    return parser.parse_args()


def main() -> None:
    """Run sampling from the command line."""

    args = parse_args()
    summary_paths = run_sampling(project_id=args.project_id)
    for summary_path in summary_paths:
        print(f"Sample summary written to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

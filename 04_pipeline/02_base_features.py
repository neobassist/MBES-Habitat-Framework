#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate configured base features from aligned MBES rasters.

Frequency inventory comes from each site's configuration, while normalized
difference pairs and their stable output names come from project configuration.
This stage does not read or rasterize label shapefiles.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    configured_frequencies,
    configured_ndi_pairs,
    ensure_output_dirs,
    get_site_config,
    load_project_config,
    load_sites_config,
)
from utils.raster_io import (
    check_same_grid,
    read_raster,
    summarize_raster,
    write_raster,
)


SUMMARY_COLUMNS = [
    "site_id",
    "site_name",
    "feature_name",
    "feature_group",
    "frequency",
    "source_inputs",
    "output_path",
    "width",
    "height",
    "crs",
    "resolution_m",
    "dtype",
    "nodata",
    "valid_count",
    "invalid_count",
    "valid_fraction",
    "min",
    "max",
    "mean",
    "std",
    "status",
    "message",
]


def _require_scipy() -> Any:
    """Import SciPy ndimage or raise a clear dependency error."""

    try:
        import scipy.ndimage as ndi  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "SciPy is required for 02_base_features.py. Install scipy in the "
            "mbes_seaweed environment before running this stage."
        ) from exc
    return ndi


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Validate that ``value`` is a mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _as_path(path: str | Path) -> Path:
    """Return ``path`` as an expanded ``Path``."""

    return Path(path).expanduser()


def _read_float_raster(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one raster band as float32 with nodata converted to NaN."""

    array, profile = read_raster(path, band=1, masked=True)
    nodata = profile.get("nodata")

    if np.ma.isMaskedArray(array):
        values = np.ma.asarray(array, dtype="float32").filled(np.nan)
    else:
        values = np.asarray(array, dtype="float32")

    if nodata is not None and np.isfinite(nodata):
        values = np.where(values == nodata, np.nan, values)
    values[~np.isfinite(values)] = np.nan
    return values.astype("float32", copy=False), profile


def _read_common_valid_mask(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the common valid mask as a boolean array."""

    array, profile = read_raster(path, band=1, masked=True)
    if np.ma.isMaskedArray(array):
        values = np.ma.asarray(array).filled(0)
    else:
        values = np.asarray(array)
    mask = np.isfinite(values) & (values.astype("float32") > 0)
    return mask.astype(bool, copy=False), profile


def apply_common_mask(array: np.ndarray, common_valid_mask: np.ndarray) -> np.ndarray:
    """Return ``array`` with invalid/common-mask-off pixels set to NaN."""

    result = np.asarray(array, dtype="float32").copy()
    result[~common_valid_mask] = np.nan
    result[~np.isfinite(result)] = np.nan
    return result


def robust_bin_scale_by_depth(
    band: np.ndarray,
    depth: np.ndarray,
    common_valid_mask: np.ndarray,
    depth_bins_q: int,
    min_per_bin: int,
    eps: float,
) -> np.ndarray:
    """Depth-bin robust normalization for one acoustic band.

    The transform is ``(x - median) / max(p90 - p10, eps)`` within depth
    quantile bins. Sparse bins fall back to global robust statistics.
    """

    out = np.full_like(band, np.nan, dtype="float32")
    finite = common_valid_mask & np.isfinite(band) & np.isfinite(depth)
    if not finite.any():
        return out

    x = band[finite]
    d = depth[finite]
    global_med = float(np.nanmedian(x))
    global_p10, global_p90 = np.nanpercentile(x, [10, 90])
    global_scale = max(float(global_p90 - global_p10), eps)

    try:
        qs = np.linspace(0, 100, int(depth_bins_q) + 1)
        edges = np.unique(np.nanpercentile(d, qs))
    except Exception:
        out[finite] = (x - global_med) / global_scale
        return out

    if edges.size < 2:
        out[finite] = (x - global_med) / global_scale
        return out

    for index in range(len(edges) - 1):
        lo = edges[index]
        hi = edges[index + 1]
        if index == len(edges) - 2:
            bin_mask = finite & (depth >= lo) & (depth <= hi)
        else:
            bin_mask = finite & (depth >= lo) & (depth < hi)

        n = int(np.count_nonzero(bin_mask))
        if n == 0:
            continue

        xx = band[bin_mask]
        if n >= min_per_bin:
            med = float(np.nanmedian(xx))
            p10, p90 = np.nanpercentile(xx, [10, 90])
            scale = max(float(p90 - p10), eps)
            out[bin_mask] = (xx - med) / scale
        else:
            out[bin_mask] = (xx - global_med) / global_scale

    return apply_common_mask(out, common_valid_mask)


def compute_dpth_lite(
    depth: np.ndarray,
    common_valid_mask: np.ndarray,
    clip: float,
    eps: float,
) -> np.ndarray:
    """Compute robust median/p5/p95 depth-lite feature."""

    out = np.full_like(depth, np.nan, dtype="float32")
    finite = common_valid_mask & np.isfinite(depth)
    if not finite.any():
        return out

    values = depth[finite]
    median = float(np.nanmedian(values))
    p5, p95 = np.nanpercentile(values, [5, 95])
    scale = max(float(p95 - p5), eps)

    out[finite] = (values - median) / scale
    out = np.clip(out, -float(clip), float(clip)).astype("float32")
    return apply_common_mask(out, common_valid_mask)


def resize_nn(array: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize using NumPy indexing."""

    if array.ndim != 2:
        raise ValueError(f"resize_nn expects a 2D array, got shape {array.shape}.")
    h, w = array.shape
    if h <= 0 or w <= 0 or new_h <= 0 or new_w <= 0:
        raise ValueError("resize_nn dimensions must be positive.")

    scale_h = new_h / h
    scale_w = new_w / w
    yy = np.floor(np.arange(new_h) / scale_h).astype(int)
    xx = np.floor(np.arange(new_w) / scale_w).astype(int)
    yy[yy >= h] = h - 1
    xx[xx >= w] = w - 1
    return array[yy[:, None], xx[None, :]]


def compute_relative_depth(
    depth: np.ndarray,
    native_res: float,
    common_valid_mask: np.ndarray,
    coarse_res_m: float,
    radius_m: float,
    iqr_eps: float,
    clip: float,
) -> np.ndarray:
    """Compute local median/IQR relative depth."""

    ndi = _require_scipy()

    out = np.full_like(depth, np.nan, dtype="float32")
    valid_depth = apply_common_mask(depth, common_valid_mask)
    finite = np.isfinite(valid_depth)
    if not finite.any():
        return out

    fill_value = float(np.nanmedian(valid_depth))
    filled = np.where(finite, valid_depth, fill_value).astype("float32")

    scale = max(1, int(round(float(coarse_res_m) / float(native_res))))
    height, width = filled.shape
    coarse_h = int(math.ceil(height / scale))
    coarse_w = int(math.ceil(width / scale))
    pad_h = coarse_h * scale - height
    pad_w = coarse_w * scale - width

    filled_pad = np.pad(filled, ((0, pad_h), (0, pad_w)), mode="edge")
    finite_pad = np.pad(finite.astype("float32"), ((0, pad_h), (0, pad_w)), mode="edge")

    blocks = filled_pad.reshape(coarse_h, scale, coarse_w, scale).transpose(0, 2, 1, 3)
    finite_blocks = finite_pad.reshape(coarse_h, scale, coarse_w, scale).transpose(0, 2, 1, 3)

    finite_counts = finite_blocks.sum(axis=(2, 3))
    value_sums = (blocks * finite_blocks).sum(axis=(2, 3))
    coarse_depth = np.divide(value_sums, finite_counts, out=np.full_like(value_sums, fill_value, dtype="float32"), where=finite_counts > 0).astype("float32")

    win = max(3, int(round((float(radius_m) / float(coarse_res_m)) * 2 + 1)))
    if win % 2 == 0:
        win += 1

    p50 = ndi.percentile_filter(coarse_depth, 50, size=win, mode="reflect").astype("float32")
    p25 = ndi.percentile_filter(coarse_depth, 25, size=win, mode="reflect").astype("float32")
    p75 = ndi.percentile_filter(coarse_depth, 75, size=win, mode="reflect").astype("float32")
    iqr = np.maximum(p75 - p25, float(iqr_eps)).astype("float32")

    coarse_rel = ((coarse_depth - p50) / iqr).astype("float32")
    native_rel = resize_nn(coarse_rel, height, width).astype("float32")
    native_rel = np.clip(native_rel, -float(clip), float(clip)).astype("float32")

    out[finite] = native_rel[finite]
    return apply_common_mask(out, common_valid_mask)


def compute_ndi(
    high: np.ndarray,
    low: np.ndarray,
    common_valid_mask: np.ndarray,
) -> np.ndarray:
    """Compute normalized difference index without [0, 1] rescaling."""

    denominator = np.abs(high) + np.abs(low)
    finite = common_valid_mask & np.isfinite(high) & np.isfinite(low) & np.isfinite(denominator)
    finite &= denominator != 0

    out = np.full_like(high, np.nan, dtype="float32")
    out[finite] = (high[finite] - low[finite]) / denominator[finite]
    return apply_common_mask(out, common_valid_mask)


def _robust_texture_clip_scale(
    array: np.ndarray,
    valid_mask: np.ndarray,
    clip: float,
) -> np.ndarray:
    """Scale texture output by p5/p95 midpoint and clip."""

    out = np.full_like(array, np.nan, dtype="float32")
    finite = valid_mask & np.isfinite(array)
    if not finite.any():
        return out

    p5, p95 = np.nanpercentile(array[finite], [5, 95])
    scale = max(float(p95 - p5), 1e-6)
    center = float((p5 + p95) / 2.0)
    out[finite] = (array[finite] - center) / scale
    out = np.clip(out, -float(clip), float(clip)).astype("float32")
    return apply_common_mask(out, valid_mask)


def compute_textures(
    acoustic_norm: np.ndarray,
    common_valid_mask: np.ndarray,
    gauss_sigma_px: float,
    log_sigma_px: float,
    use_log1p: bool,
    clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute low/high/LoG texture channels from normalized acoustic input."""

    ndi = _require_scipy()

    acoustic = apply_common_mask(acoustic_norm, common_valid_mask)
    nan_mask = ~np.isfinite(acoustic)
    fill_value = 0.0
    if np.isfinite(acoustic).any():
        fill_value = float(np.nanmedian(acoustic))
    filled = np.where(nan_mask, fill_value, acoustic).astype("float32")

    low_raw = ndi.gaussian_filter(filled, sigma=float(gauss_sigma_px)).astype("float32")
    high_raw = (filled - low_raw).astype("float32")
    blurred = ndi.gaussian_filter(filled, sigma=float(log_sigma_px)).astype("float32")
    log_raw = np.abs(ndi.laplace(blurred).astype("float32"))
    if use_log1p:
        log_raw = np.log1p(log_raw).astype("float32")

    valid_mask = common_valid_mask & ~nan_mask
    low = _robust_texture_clip_scale(low_raw, valid_mask, clip)
    high = _robust_texture_clip_scale(high_raw, valid_mask, clip)
    log1p = _robust_texture_clip_scale(log_raw, valid_mask, clip)
    return low, high, log1p


def write_feature(
    output_path: str | Path,
    array: np.ndarray,
    profile: Mapping[str, Any],
) -> Path:
    """Write one float32 base feature raster with NaN nodata."""

    feature = np.asarray(array, dtype="float32")
    output_profile = dict(profile)
    output_profile.update(count=1, dtype="float32", nodata=float("nan"))
    return write_raster(
        path=output_path,
        array=feature,
        profile=output_profile,
        nodata=float("nan"),
        dtype="float32",
    )


def summarize_feature(array: np.ndarray) -> dict[str, int | float | None]:
    """Summarize one feature array."""

    summary = summarize_raster(array)
    count = int(summary["count"])
    valid_count = int(summary["valid_count"])
    summary["valid_fraction"] = valid_count / count if count else 0.0
    return summary


def _summary_row(
    *,
    site_id: str,
    site_name: str,
    feature_name: str,
    feature_group: str,
    frequency: int | str,
    source_inputs: Sequence[str | Path],
    output_path: str | Path,
    profile: Mapping[str, Any],
    summary: Mapping[str, Any],
    status: str = "ok",
    message: str = "",
) -> dict[str, Any]:
    """Build one base feature summary row."""

    transform = profile["transform"]
    resolution_m = abs(float(transform.a)) if hasattr(transform, "a") else ""
    return {
        "site_id": site_id,
        "site_name": site_name,
        "feature_name": feature_name,
        "feature_group": feature_group,
        "frequency": frequency,
        "source_inputs": ";".join(str(path) for path in source_inputs),
        "output_path": str(output_path),
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "crs": str(profile["crs"]),
        "resolution_m": resolution_m,
        "dtype": "float32",
        "nodata": "nan",
        "valid_count": summary.get("valid_count"),
        "invalid_count": summary.get("invalid_count"),
        "valid_fraction": summary.get("valid_fraction"),
        "min": summary.get("min"),
        "max": summary.get("max"),
        "mean": summary.get("mean"),
        "std": summary.get("std"),
        "status": status,
        "message": message,
    }


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-site base feature summary CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _required_input_paths(
    prepared_site_dir: Path,
    frequencies: Sequence[int],
) -> list[Path]:
    """Return the required aligned input paths for one site."""

    paths = []
    for frequency in frequencies:
        paths.append(prepared_site_dir / f"depth_{frequency}_aligned.tif")
        paths.append(prepared_site_dir / f"backscatter_{frequency}_aligned.tif")
    paths.append(prepared_site_dir / "common_valid_mask.tif")
    return paths


def _validate_required_inputs(paths: Sequence[Path]) -> None:
    """Raise a clear error when required inputs are missing."""

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required base feature input file(s): " + "; ".join(missing))


def _feature_group(feature_name: str) -> str:
    """Return the feature group name for summaries."""

    if feature_name.startswith("a") and feature_name.endswith("_norm"):
        return "acoustic"
    if feature_name.startswith("NDI_"):
        return "frequency_difference"
    if feature_name.startswith("dpth_lite"):
        return "depth_lite"
    if feature_name.startswith("dpth_rel"):
        return "relative_depth"
    return "texture"


def _write_and_summarize_feature(
    rows: list[dict[str, Any]],
    *,
    site_id: str,
    site_name: str,
    feature_name: str,
    array: np.ndarray,
    profile: Mapping[str, Any],
    output_dir: Path,
    source_inputs: Sequence[str | Path],
    frequency: int | str,
) -> None:
    """Write one feature raster and append its summary row."""

    output_path = output_dir / f"{feature_name}.tif"
    write_feature(output_path, array, profile)
    rows.append(
        _summary_row(
            site_id=site_id,
            site_name=site_name,
            feature_name=feature_name,
            feature_group=_feature_group(feature_name),
            frequency=frequency,
            source_inputs=source_inputs,
            output_path=output_path,
            profile=profile,
            summary=summarize_feature(array),
        )
    )


def process_site(
    site_id: str,
    site_config: Mapping[str, Any],
    prepared_root: Path,
    base_features_root: Path,
    base_feature_config: Mapping[str, Any],
) -> Path:
    """Generate all configured base features for one site."""

    print(f"Processing site: {site_id}", flush=True)
    site_name = str(site_config.get("site_name", site_id))
    prepared_site_dir = prepared_root / site_id
    output_dir = base_features_root / site_id
    output_dir.mkdir(parents=True, exist_ok=True)

    frequencies = configured_frequencies(site_config)
    ndi_pairs = configured_ndi_pairs(base_feature_config)
    available = set(frequencies)
    unavailable_pairs = [
        pair
        for pair in ndi_pairs
        if pair["high_frequency"] not in available or pair["low_frequency"] not in available
    ]
    if unavailable_pairs:
        raise ValueError(
            f"Configured NDI frequencies are unavailable for site {site_id}: {unavailable_pairs}."
        )

    required_paths = _required_input_paths(prepared_site_dir, frequencies)
    _validate_required_inputs(required_paths)
    check_same_grid(required_paths)

    mask, mask_profile = _read_common_valid_mask(prepared_site_dir / "common_valid_mask.tif")
    if not mask.any():
        raise ValueError(f"common_valid_mask contains zero valid pixels for site {site_id}.")

    depths: dict[int, np.ndarray] = {}
    backscatter: dict[int, np.ndarray] = {}
    profiles: dict[int, dict[str, Any]] = {}
    for frequency in frequencies:
        print(f"Loading aligned rasters: {site_id} / {frequency} kHz", flush=True)
        depth_path = prepared_site_dir / f"depth_{frequency}_aligned.tif"
        backscatter_path = prepared_site_dir / f"backscatter_{frequency}_aligned.tif"
        depth, profile = _read_float_raster(depth_path)
        acoustic, _ = _read_float_raster(backscatter_path)

        depths[frequency] = apply_common_mask(depth, mask)
        backscatter[frequency] = apply_common_mask(acoustic, mask)
        profiles[frequency] = profile
        print(f"Loaded aligned rasters: {site_id} / {frequency} kHz", flush=True)

    reference_profile = profiles[frequencies[0]]
    if reference_profile["width"] != mask_profile["width"] or reference_profile["height"] != mask_profile["height"]:
        raise ValueError(f"common_valid_mask dimensions do not match rasters for site {site_id}.")

    acoustic_cfg = _require_mapping(
        base_feature_config["acoustic_normalization"],
        "base_features.acoustic_normalization",
    )
    depth_lite_cfg = _require_mapping(base_feature_config["depth_lite"], "base_features.depth_lite")
    relative_cfg = _require_mapping(
        base_feature_config["relative_depth"],
        "base_features.relative_depth",
    )
    texture_cfg = _require_mapping(base_feature_config["texture"], "base_features.texture")

    rows: list[dict[str, Any]] = []
    acoustic_norm: dict[int, np.ndarray] = {}

    for frequency in frequencies:
        feature_name = f"a{frequency}_norm"
        print(f"Generating acoustic normalization: {site_id} / {frequency} kHz", flush=True)
        acoustic_norm[frequency] = robust_bin_scale_by_depth(
            band=backscatter[frequency],
            depth=depths[frequency],
            common_valid_mask=mask,
            depth_bins_q=int(acoustic_cfg["depth_bins_q"]),
            min_per_bin=int(acoustic_cfg["min_per_bin"]),
            eps=float(acoustic_cfg["eps"]),
        )
        _write_and_summarize_feature(
            rows,
            site_id=site_id,
            site_name=site_name,
            feature_name=feature_name,
            array=acoustic_norm[frequency],
            profile=profiles[frequency],
            output_dir=output_dir,
            source_inputs=[
                prepared_site_dir / f"backscatter_{frequency}_aligned.tif",
                prepared_site_dir / f"depth_{frequency}_aligned.tif",
                prepared_site_dir / "common_valid_mask.tif",
            ],
            frequency=frequency,
        )
        print(f"Generated acoustic normalization: {site_id} / {frequency} kHz", flush=True)

    for pair in ndi_pairs:
        feature_name = str(pair["feature_name"])
        high_freq = int(pair["high_frequency"])
        low_freq = int(pair["low_frequency"])
        print(f"Generating frequency-difference feature: {site_id} / {feature_name}", flush=True)
        feature = compute_ndi(acoustic_norm[high_freq], acoustic_norm[low_freq], mask)
        _write_and_summarize_feature(
            rows,
            site_id=site_id,
            site_name=site_name,
            feature_name=feature_name,
            array=feature,
            profile=reference_profile,
            output_dir=output_dir,
            source_inputs=[
                output_dir / f"a{high_freq}_norm.tif",
                output_dir / f"a{low_freq}_norm.tif",
                prepared_site_dir / "common_valid_mask.tif",
            ],
            frequency="",
        )
        print(f"Generated frequency-difference feature: {site_id} / {feature_name}", flush=True)

    for frequency in frequencies:
        feature_name = f"dpth_lite_{frequency}"
        print(f"Generating depth-lite feature: {site_id} / {frequency} kHz", flush=True)
        feature = compute_dpth_lite(
            depth=depths[frequency],
            common_valid_mask=mask,
            clip=float(depth_lite_cfg["clip"]),
            eps=float(depth_lite_cfg["eps"]),
        )
        _write_and_summarize_feature(
            rows,
            site_id=site_id,
            site_name=site_name,
            feature_name=feature_name,
            array=feature,
            profile=profiles[frequency],
            output_dir=output_dir,
            source_inputs=[
                prepared_site_dir / f"depth_{frequency}_aligned.tif",
                prepared_site_dir / "common_valid_mask.tif",
            ],
            frequency=frequency,
        )
        print(f"Generated depth-lite feature: {site_id} / {frequency} kHz", flush=True)

    native_res = abs(float(reference_profile["transform"].a))
    for frequency in frequencies:
        feature_name = f"dpth_rel_{frequency}"
        print(f"Generating relative-depth feature: {site_id} / {frequency} kHz", flush=True)
        feature = compute_relative_depth(
            depth=depths[frequency],
            native_res=native_res,
            common_valid_mask=mask,
            coarse_res_m=float(relative_cfg["coarse_res_m"]),
            radius_m=float(relative_cfg["radius_m"]),
            iqr_eps=float(relative_cfg["iqr_eps"]),
            clip=float(relative_cfg["clip"]),
        )
        _write_and_summarize_feature(
            rows,
            site_id=site_id,
            site_name=site_name,
            feature_name=feature_name,
            array=feature,
            profile=profiles[frequency],
            output_dir=output_dir,
            source_inputs=[
                prepared_site_dir / f"depth_{frequency}_aligned.tif",
                prepared_site_dir / "common_valid_mask.tif",
            ],
            frequency=frequency,
        )
        print(f"Generated relative-depth feature: {site_id} / {frequency} kHz", flush=True)

    for frequency in frequencies:
        print(f"Generating texture features: {site_id} / {frequency} kHz", flush=True)
        low, high, log1p = compute_textures(
            acoustic_norm=acoustic_norm[frequency],
            common_valid_mask=mask,
            gauss_sigma_px=float(texture_cfg["gauss_sigma_px"]),
            log_sigma_px=float(texture_cfg["log_sigma_px"]),
            use_log1p=bool(texture_cfg["use_log1p"]),
            clip=float(texture_cfg["clip"]),
        )
        texture_outputs = {
            f"a{frequency}_low": low,
            f"a{frequency}_high": high,
            f"a{frequency}_log1p": log1p,
        }
        for feature_name, feature in texture_outputs.items():
            _write_and_summarize_feature(
                rows,
                site_id=site_id,
                site_name=site_name,
                feature_name=feature_name,
                array=feature,
                profile=profiles[frequency],
                output_dir=output_dir,
                source_inputs=[
                    output_dir / f"a{frequency}_norm.tif",
                    prepared_site_dir / "common_valid_mask.tif",
                ],
                frequency=frequency,
            )
        print(f"Generated texture features: {site_id} / {frequency} kHz", flush=True)

    summary_path = output_dir / "base_feature_summary.csv"
    _write_summary(summary_path, rows)
    print(f"Completed site: {site_id}", flush=True)
    return summary_path


def run_base_features(project_id: str = DEFAULT_PROJECT_ID) -> list[Path]:
    """Generate base features for all configured sites."""

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="02")

    base_feature_config = _require_mapping(
        project_config.get("base_features"),
        "project_config.base_features",
    )
    prepared_root = output_dirs["prepared"]
    base_features_root = output_dirs["base_features"]

    sites = _require_mapping(sites_config.get("sites"), "sites_config.sites")
    summary_paths: list[Path] = []
    for site_id in sites:
        site_config = get_site_config(sites_config, str(site_id))
        summary_paths.append(
            process_site(
                site_id=str(site_id),
                site_config=site_config,
                prepared_root=prepared_root,
                base_features_root=base_features_root,
                base_feature_config=base_feature_config,
            )
        )

    return summary_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate configured base feature rasters from aligned rasters.",
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=f"Project id to process. Default: {DEFAULT_PROJECT_ID}",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_args()
    summary_paths = run_base_features(project_id=args.project_id)
    for summary_path in summary_paths:
        print(f"Base feature summary written to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

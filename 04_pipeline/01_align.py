#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Align source depth/backscatter rasters to per-site common grids.

This stage is intentionally limited to:
- bathymetry alignment
- backscatter alignment
- crop mask generation
- common valid mask generation

Label shapefile rasterization belongs to ``04_sampling.py`` and is not
performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    ensure_output_dirs,
    get_site_config,
    load_project_config,
    load_sites_config,
    resolve_repo_path,
)
from utils.raster_io import (
    build_reference_grid,
    check_same_grid,
    make_crop_mask,
    make_valid_data_mask,
    read_raster,
    read_vector,
    reproject_raster_to_grid,
    reproject_vector,
    summarize_raster,
    write_raster,
    write_esri_ascii_pair,
)


BATHYMETRY_RESAMPLING = "bilinear"
BACKSCATTER_RESAMPLING = "bilinear"
MASK_RESAMPLING = "nearest"
SUMMARY_COLUMNS = [
    "site_id",
    "site_name",
    "data_type",
    "frequency",
    "source_path",
    "output_path",
    "resampling",
    "crs_out",
    "resolution_m",
    "width",
    "height",
    "nodata",
    "dtype",
    "valid_count",
    "invalid_count",
    "valid_fraction",
    "min",
    "max",
    "mean",
    "std",
    "grid_match",
    "status",
    "message",
]
COMMON_VALID_SUMMARY_COLUMNS = [
    "site_id",
    "site_name",
    "asset_type",
    "frequency_khz",
    "source_prepared_path",
    "common_valid_mask_path",
    "tif_path",
    "asc_path",
    "txt_path",
    "source_sha256",
    "mask_sha256",
    "crs",
    "resolution_m",
    "width",
    "height",
    "dtype",
    "nodata",
    "row_order",
    "numeric_precision",
    "resampling",
    "valid_pixel_count",
    "outside_mask_nodata_count",
    "exact_value_equality",
    "status",
]
COMMON_VALID_FLOAT_NODATA = -9999.0
COMMON_VALID_MASK_NODATA = -9999


def _require_mapping(value: Any, name: str) -> Mapping[Any, Any]:
    """Validate that ``value`` is a mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    """Validate that ``value`` is a list."""

    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list, got {type(value).__name__}.")
    return value


def _lookup_frequency_value(mapping: Mapping[Any, Any], frequency: Any) -> Any | None:
    """Return a frequency-keyed value while allowing int or string YAML keys."""

    if frequency in mapping:
        return mapping[frequency]

    frequency_str = str(frequency)
    if frequency_str in mapping:
        return mapping[frequency_str]

    try:
        frequency_int = int(frequency)
    except (TypeError, ValueError):
        return None

    return mapping.get(frequency_int)


def _resolve_alignment_inputs(site_config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve bathymetry, backscatter, and crop paths for one site.

    Label paths are deliberately ignored in this stage.
    """

    paths = _require_mapping(site_config.get("paths"), "site_config.paths")
    input_files = _require_mapping(site_config.get("input_files"), "site_config.input_files")
    frequencies = _require_list(site_config.get("frequencies"), "site_config.frequencies")

    bathymetry_files = _require_mapping(
        input_files.get("bathymetry"),
        "site_config.input_files.bathymetry",
    )
    backscatter_files = _require_mapping(
        input_files.get("backscatter"),
        "site_config.input_files.backscatter",
    )

    bathy_dir = resolve_repo_path(paths.get("bathy_dir"))
    sonar_dir = resolve_repo_path(paths.get("sonar_dir"))
    crop_shp = resolve_repo_path(paths.get("crop_shp"))

    bathymetry: dict[int, Path] = {}
    backscatter: dict[int, Path] = {}
    missing: list[str] = []

    if not crop_shp.exists():
        missing.append(str(crop_shp))

    for frequency in frequencies:
        bathy_name = _lookup_frequency_value(bathymetry_files, frequency)
        backscatter_name = _lookup_frequency_value(backscatter_files, frequency)
        if bathy_name is None:
            raise ValueError(f"Missing bathymetry filename for frequency {frequency}.")
        if backscatter_name is None:
            raise ValueError(f"Missing backscatter filename for frequency {frequency}.")

        frequency_int = int(frequency)
        bathy_path = bathy_dir / str(bathy_name)
        backscatter_path = sonar_dir / str(backscatter_name)
        bathymetry[frequency_int] = bathy_path
        backscatter[frequency_int] = backscatter_path

        if not bathy_path.exists():
            missing.append(str(bathy_path))
        if not backscatter_path.exists():
            missing.append(str(backscatter_path))

    if missing:
        raise FileNotFoundError("Missing required alignment input file(s): " + "; ".join(missing))

    return {
        "frequencies": [int(frequency) for frequency in frequencies],
        "bathymetry": bathymetry,
        "backscatter": backscatter,
        "crop_shp": crop_shp,
    }


def _build_site_grid(
    crop_shp: Path,
    crs_out: str,
    resolution_m: float,
) -> dict[str, Any]:
    """Build a site common grid from the crop polygon in output CRS."""

    if resolution_m <= 0:
        raise ValueError(f"target_resolution_m must be > 0, got {resolution_m}.")

    crop_gdf = read_vector(crop_shp)
    if getattr(crop_gdf, "crs", None) is None:
        raise ValueError(f"Crop shapefile has no CRS: {crop_shp}")

    crop_gdf = reproject_vector(crop_gdf, crs_out)
    if crop_gdf.empty:
        raise ValueError(f"Crop shapefile contains no features after reprojection: {crop_shp}")

    bounds = crop_gdf.total_bounds
    if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
        raise ValueError(f"Crop shapefile has invalid bounds after reprojection: {crop_shp}")

    return build_reference_grid(
        bounds=bounds,
        resolution=resolution_m,
        crs=crs_out,
        dtype="float32",
        nodata=float("nan"),
    )


def _continuous_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return a profile for aligned continuous float32 rasters."""

    output_profile = profile.copy()
    output_profile.update(count=1, dtype="float32", nodata=float("nan"))
    return output_profile


def _mask_profile(grid_profile: dict[str, Any]) -> dict[str, Any]:
    """Return a uint8 0/1 mask profile without nodata."""

    output_profile = grid_profile.copy()
    output_profile.update(count=1, dtype="uint8", nodata=None)
    return output_profile


def _to_float32_nan(array: Any) -> np.ndarray:
    """Return a float32 array with all non-finite values set to NaN."""

    result = np.asarray(array, dtype="float32")
    result[~np.isfinite(result)] = np.nan
    return result


def _summarize_continuous(array: np.ndarray) -> dict[str, int | float | None]:
    """Summarize a continuous raster array."""

    summary = summarize_raster(array)
    valid_count = int(summary["valid_count"])
    count = int(summary["count"])
    summary["valid_fraction"] = valid_count / count if count else 0.0
    return summary


def _summarize_mask(mask: np.ndarray) -> dict[str, int | float]:
    """Summarize a uint8 0/1 mask."""

    values = np.asarray(mask, dtype="uint8")
    count = int(values.size)
    valid_count = int(np.count_nonzero(values == 1))
    invalid_count = count - valid_count
    return {
        "count": count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "valid_fraction": valid_count / count if count else 0.0,
        "min": float(values.min()) if count else None,
        "max": float(values.max()) if count else None,
        "mean": float(values.mean()) if count else None,
        "std": float(values.std()) if count else None,
    }


def _summary_row(
    *,
    site_id: str,
    site_name: str,
    data_type: str,
    frequency: int | str,
    source_path: str | Path,
    output_path: str | Path,
    resampling: str,
    crs_out: str,
    resolution_m: float,
    profile: Mapping[str, Any],
    summary: Mapping[str, Any],
    grid_match: bool,
    status: str = "ok",
    message: str = "",
) -> dict[str, Any]:
    """Build one alignment summary row."""

    return {
        "site_id": site_id,
        "site_name": site_name,
        "data_type": data_type,
        "frequency": frequency,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "resampling": resampling,
        "crs_out": crs_out,
        "resolution_m": resolution_m,
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "nodata": profile.get("nodata"),
        "dtype": profile.get("dtype"),
        "valid_count": summary.get("valid_count"),
        "invalid_count": summary.get("invalid_count"),
        "valid_fraction": summary.get("valid_fraction"),
        "min": summary.get("min"),
        "max": summary.get("max"),
        "mean": summary.get("mean"),
        "std": summary.get("std"),
        "grid_match": grid_match,
        "status": status,
        "message": message,
    }


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write alignment summary rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one provenance input."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_output_path(path: Path, output_root: Path) -> str:
    """Return an output-root-relative path for portable project metadata."""

    return path.resolve(strict=False).relative_to(output_root.resolve(strict=False)).as_posix()


def _write_common_valid_readme(path: Path) -> None:
    """Write the canonical physical-raster export contract."""

    text = """# Common valid physical rasters

This Stage 01 product contains physical bathymetry and backscatter values on the native prepared grid. The source prepared rasters are aligned and resampled to a common north-up EPSG:32652 grid cropped to the configured crop polygon bounding box; they are not themselves pixel-mask-applied. This directory applies `common_valid_mask.tif`: physical values are unchanged inside the mask and are NoData outside it.

Each site directory contains the common-valid mask and every configured bathymetry/backscatter frequency as GeoTIFF, ESRI ASCII Grid (`.asc`), and an identical `.txt` copy. ASCII rows run north to south from the upper raster row. Coordinates are projected metres in EPSG:32652, `xllcorner` and `yllcorner` describe the lower-left outer grid corner, and the native 0.07 m resolution is preserved without additional resampling. Continuous values use six decimal places and NoData `-9999`; the mask uses integer 0/1 values. `common_valid_rasters_summary.csv` records provenance, grid properties, precision, and exact-value equality checks against the historical `prepared/<site>/... + common_valid_mask.tif` access path.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_common_valid_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMMON_VALID_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_common_valid_site(
    *,
    site_id: str,
    site_name: str,
    frequencies: list[int],
    prepared_dir: Path,
    common_valid_root: Path,
    output_root: Path,
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Materialize mask-applied physical rasters from saved Stage 01 outputs."""

    site_prepared = prepared_dir / site_id
    mask_path = site_prepared / "common_valid_mask.tif"
    mask_array, mask_profile = read_raster(mask_path, masked=False)
    mask = np.asarray(mask_array, dtype="uint8") == 1
    site_output = common_valid_root / site_id
    site_output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    mask_output = site_output / "common_valid_mask.tif"
    write_raster(mask_output, mask.astype("uint8"), _mask_profile(mask_profile), nodata=None, dtype="uint8")
    mask_ascii = write_esri_ascii_pair(
        mask_output,
        site_output / "common_valid_mask.asc",
        site_output / "common_valid_mask.txt",
        nodata=COMMON_VALID_MASK_NODATA,
        integer=True,
        precision=0,
        overwrite=overwrite,
    )
    mask_roundtrip, _ = read_raster(mask_output, masked=False)
    rows.append(
        {
            "site_id": site_id,
            "site_name": site_name,
            "asset_type": "common_valid_mask",
            "frequency_khz": "",
            "source_prepared_path": _portable_output_path(mask_path, output_root),
            "common_valid_mask_path": _portable_output_path(mask_path, output_root),
            "tif_path": _portable_output_path(mask_output, output_root),
            "asc_path": _portable_output_path(Path(mask_ascii["asc_path"]), output_root),
            "txt_path": _portable_output_path(Path(mask_ascii["txt_path"]), output_root),
            "source_sha256": _sha256_file(mask_path),
            "mask_sha256": _sha256_file(mask_path),
            "crs": str(mask_profile.get("crs")),
            "resolution_m": abs(float(mask_profile["transform"].a)),
            "width": int(mask_profile["width"]),
            "height": int(mask_profile["height"]),
            "dtype": "uint8",
            "nodata": "",
            "row_order": "north_to_south",
            "numeric_precision": 0,
            "resampling": "none_native_grid",
            "valid_pixel_count": int(mask.sum()),
            "outside_mask_nodata_count": 0,
            "exact_value_equality": bool(np.array_equal(mask_roundtrip, mask.astype("uint8"))),
            "status": "ok",
        }
    )

    for asset_type, source_prefix, output_prefix in (
        ("bathymetry", "depth", "bathymetry"),
        ("backscatter", "backscatter", "backscatter"),
    ):
        for frequency in frequencies:
            source_path = site_prepared / f"{source_prefix}_{frequency}_aligned.tif"
            source_array, source_profile = read_raster(source_path, masked=True)
            physical = np.ma.asarray(source_array, dtype="float32").filled(np.nan)
            masked_physical = np.where(mask, physical, np.nan).astype("float32", copy=False)
            stem = f"{output_prefix}_{frequency}"
            tif_path = site_output / f"{stem}.tif"
            write_raster(
                tif_path,
                masked_physical,
                _continuous_profile(source_profile),
                nodata=float("nan"),
                dtype="float32",
            )
            ascii_record = write_esri_ascii_pair(
                tif_path,
                site_output / f"{stem}.asc",
                site_output / f"{stem}.txt",
                nodata=COMMON_VALID_FLOAT_NODATA,
                integer=False,
                precision=6,
                overwrite=overwrite,
            )
            roundtrip, _ = read_raster(tif_path, masked=False)
            inside_equal = np.array_equal(
                np.asarray(roundtrip)[mask],
                masked_physical[mask],
                equal_nan=True,
            )
            outside_nodata = int(np.count_nonzero(~np.isfinite(np.asarray(roundtrip)[~mask])))
            rows.append(
                {
                    "site_id": site_id,
                    "site_name": site_name,
                    "asset_type": asset_type,
                    "frequency_khz": frequency,
                    "source_prepared_path": _portable_output_path(source_path, output_root),
                    "common_valid_mask_path": _portable_output_path(mask_path, output_root),
                    "tif_path": _portable_output_path(tif_path, output_root),
                    "asc_path": _portable_output_path(Path(ascii_record["asc_path"]), output_root),
                    "txt_path": _portable_output_path(Path(ascii_record["txt_path"]), output_root),
                    "source_sha256": _sha256_file(source_path),
                    "mask_sha256": _sha256_file(mask_path),
                    "crs": str(source_profile.get("crs")),
                    "resolution_m": abs(float(source_profile["transform"].a)),
                    "width": int(source_profile["width"]),
                    "height": int(source_profile["height"]),
                    "dtype": "float32",
                    "nodata": "NaN",
                    "row_order": "north_to_south",
                    "numeric_precision": 6,
                    "resampling": "none_native_grid",
                    "valid_pixel_count": int(np.count_nonzero(mask & np.isfinite(physical))),
                    "outside_mask_nodata_count": outside_nodata,
                    "exact_value_equality": bool(inside_equal and outside_nodata == int((~mask).sum())),
                    "status": "ok" if inside_equal else "failed",
                }
            )
    return rows


def materialize_common_valid_rasters_from_prepared(
    project_id: str = DEFAULT_PROJECT_ID,
    *,
    overwrite: bool = True,
) -> Path:
    """Regenerate the canonical Stage 01 export from existing prepared rasters."""

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="01")
    output_root = output_dirs["output_root"]
    prepared_dir = output_dirs["prepared"]
    common_valid_root = output_dirs["common_valid_rasters"]
    sites = _require_mapping(sites_config.get("sites"), "sites_config.sites")
    rows: list[dict[str, Any]] = []
    for site_id in sites:
        site_config = get_site_config(sites_config, str(site_id))
        frequencies = [int(value) for value in _require_list(site_config.get("frequencies"), "site_config.frequencies")]
        rows.extend(
            _write_common_valid_site(
                site_id=str(site_id),
                site_name=str(site_config.get("site_name", site_id)),
                frequencies=frequencies,
                prepared_dir=prepared_dir,
                common_valid_root=common_valid_root,
                output_root=output_root,
                overwrite=overwrite,
            )
        )
    _write_common_valid_readme(common_valid_root / "README.md")
    summary_path = common_valid_root / "common_valid_rasters_summary.csv"
    _write_common_valid_summary(summary_path, rows)
    if not all(str(row["exact_value_equality"]).lower() == "true" for row in rows):
        raise ValueError("Common-valid physical raster equality QA failed.")
    return summary_path


def _align_raster(
    src_path: Path,
    output_path: Path,
    grid_profile: dict[str, Any],
    resampling: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align one continuous raster to the site common grid and write it."""

    array, profile = reproject_raster_to_grid(
        src_path=src_path,
        grid_profile=grid_profile,
        resampling=resampling,
        dst_nodata=float("nan"),
    )
    aligned = _to_float32_nan(array)
    output_profile = _continuous_profile(profile)
    write_raster(
        path=output_path,
        array=aligned,
        profile=output_profile,
        nodata=float("nan"),
        dtype="float32",
    )
    return aligned, output_profile


def process_site(
    site_id: str,
    site_config: Mapping[str, Any],
    prepared_dir: Path,
    crs_out: str,
    resolution_m: float,
) -> list[dict[str, Any]]:
    """Align all bathymetry/backscatter rasters for one site."""

    print(f"Processing site: {site_id}", flush=True)
    site_name = str(site_config.get("site_name", site_id))
    inputs = _resolve_alignment_inputs(site_config)
    site_dir = prepared_dir / site_id
    site_dir.mkdir(parents=True, exist_ok=True)

    grid_profile = _build_site_grid(
        crop_shp=inputs["crop_shp"],
        crs_out=crs_out,
        resolution_m=resolution_m,
    )
    mask_profile = _mask_profile(grid_profile)

    rows: list[dict[str, Any]] = []
    aligned_arrays: list[np.ndarray] = []
    output_paths: list[Path] = []

    for frequency in inputs["frequencies"]:
        source_path = inputs["bathymetry"][frequency]
        output_path = site_dir / f"depth_{frequency}_aligned.tif"
        print(f"Aligning bathymetry: {site_id} / {frequency} kHz", flush=True)
        aligned, output_profile = _align_raster(
            src_path=source_path,
            output_path=output_path,
            grid_profile=grid_profile,
            resampling=BATHYMETRY_RESAMPLING,
        )
        aligned_arrays.append(aligned)
        output_paths.append(output_path)
        rows.append(
            _summary_row(
                site_id=site_id,
                site_name=site_name,
                data_type="bathymetry",
                frequency=frequency,
                source_path=source_path,
                output_path=output_path,
                resampling=BATHYMETRY_RESAMPLING,
                crs_out=crs_out,
                resolution_m=resolution_m,
                profile=output_profile,
                summary=_summarize_continuous(aligned),
                grid_match=True,
            )
        )
        print(f"Aligned bathymetry: {site_id} / {frequency} kHz", flush=True)

    for frequency in inputs["frequencies"]:
        source_path = inputs["backscatter"][frequency]
        output_path = site_dir / f"backscatter_{frequency}_aligned.tif"
        print(f"Aligning backscatter: {site_id} / {frequency} kHz", flush=True)
        aligned, output_profile = _align_raster(
            src_path=source_path,
            output_path=output_path,
            grid_profile=grid_profile,
            resampling=BACKSCATTER_RESAMPLING,
        )
        aligned_arrays.append(aligned)
        output_paths.append(output_path)
        rows.append(
            _summary_row(
                site_id=site_id,
                site_name=site_name,
                data_type="backscatter",
                frequency=frequency,
                source_path=source_path,
                output_path=output_path,
                resampling=BACKSCATTER_RESAMPLING,
                crs_out=crs_out,
                resolution_m=resolution_m,
                profile=output_profile,
                summary=_summarize_continuous(aligned),
                grid_match=True,
            )
        )
        print(f"Aligned backscatter: {site_id} / {frequency} kHz", flush=True)

    print(f"Applying crop mask: {site_id}", flush=True)
    crop_mask = make_crop_mask(inputs["crop_shp"], grid_profile).astype("uint8")
    crop_mask_path = site_dir / "crop_mask.tif"
    write_raster(
        path=crop_mask_path,
        array=crop_mask,
        profile=mask_profile,
        nodata=None,
        dtype="uint8",
    )
    output_paths.append(crop_mask_path)
    rows.append(
        _summary_row(
            site_id=site_id,
            site_name=site_name,
            data_type="crop_mask",
            frequency="",
            source_path=inputs["crop_shp"],
            output_path=crop_mask_path,
            resampling=MASK_RESAMPLING,
            crs_out=crs_out,
            resolution_m=resolution_m,
            profile=mask_profile,
            summary=_summarize_mask(crop_mask),
            grid_match=True,
        )
    )

    valid_data_mask = make_valid_data_mask(aligned_arrays)
    common_valid_mask = (crop_mask.astype(bool) & valid_data_mask).astype("uint8")
    if int(common_valid_mask.sum()) == 0:
        raise ValueError(f"Common valid mask contains zero valid pixels for site {site_id}.")

    common_valid_mask_path = site_dir / "common_valid_mask.tif"
    write_raster(
        path=common_valid_mask_path,
        array=common_valid_mask,
        profile=mask_profile,
        nodata=None,
        dtype="uint8",
    )
    output_paths.append(common_valid_mask_path)
    rows.append(
        _summary_row(
            site_id=site_id,
            site_name=site_name,
            data_type="common_valid_mask",
            frequency="",
            source_path="aligned_depth_backscatter_and_crop_mask",
            output_path=common_valid_mask_path,
            resampling=MASK_RESAMPLING,
            crs_out=crs_out,
            resolution_m=resolution_m,
            profile=mask_profile,
            summary=_summarize_mask(common_valid_mask),
            grid_match=True,
        )
    )

    if not check_same_grid(output_paths):
        raise ValueError(f"Aligned outputs do not share the same grid for site {site_id}.")

    print(f"Applied crop mask: {site_id}", flush=True)
    print(f"Completed site: {site_id}", flush=True)
    return rows


def run_alignment(project_id: str = DEFAULT_PROJECT_ID) -> Path:
    """Run alignment for every configured site and return the summary path."""

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="01")

    spatial_config = _require_mapping(project_config.get("spatial"), "project_config.spatial")
    crs_out = str(spatial_config.get("crs_out"))
    resolution_m = float(spatial_config.get("target_resolution_m"))
    if resolution_m <= 0:
        raise ValueError(f"target_resolution_m must be > 0, got {resolution_m}.")

    sites = _require_mapping(sites_config.get("sites"), "sites_config.sites")
    prepared_dir = output_dirs["prepared"]
    all_rows: list[dict[str, Any]] = []

    for site_id in sites:
        site_config = get_site_config(sites_config, str(site_id))
        all_rows.extend(
            process_site(
                site_id=str(site_id),
                site_config=site_config,
                prepared_dir=prepared_dir,
                crs_out=crs_out,
                resolution_m=resolution_m,
            )
        )

    summary_path = prepared_dir / "alignment_summary.csv"
    _write_summary(summary_path, all_rows)
    # Materialize mask-applied physical products from the just-written prepared authority.
    materialize_common_valid_rasters_from_prepared(project_id=project_id, overwrite=True)
    return summary_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Align MBES bathymetry/backscatter rasters to per-site common grids.",
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=f"Project id to process. Default: {DEFAULT_PROJECT_ID}",
    )
    parser.add_argument(
        "--common-valid-only",
        action="store_true",
        help="Regenerate common_valid_rasters from existing prepared Stage 01 outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_args()
    if args.common_valid_only:
        summary_path = materialize_common_valid_rasters_from_prepared(project_id=args.project_id)
        print(f"Common-valid exports written to: {summary_path}", flush=True)
    else:
        summary_path = run_alignment(project_id=args.project_id)
        print(f"Alignment complete. Summary written to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

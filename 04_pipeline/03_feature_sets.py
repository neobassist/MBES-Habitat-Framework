#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build multi-band feature stacks from configured project base feature rasters.

This stage only combines existing ``base_features/<site_id>/*.tif`` rasters
according to the feature-set YAML definitions. It does not create new features,
resample, reproject, or use labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    ensure_output_dirs,
    get_site_config,
    load_all_feature_set_configs,
    load_experiment_matrix,
    load_project_config,
    load_sites_config,
)
from utils.raster_io import (
    check_same_grid,
    get_raster_profile,
    read_raster,
    summarize_raster,
    write_band_names,
)


FEATURE_GROUP_ORDER = (
    "acoustic",
    "frequency_difference",
    "terrain",
    "texture",
)
SUMMARY_COLUMNS = [
    "site_id",
    "site_name",
    "feature_set_id",
    "feature_set_type",
    "feature_count_expected",
    "band_count_written",
    "band_index",
    "feature_group",
    "feature_name",
    "input_path",
    "output_stack_path",
    "bands_txt_path",
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
    "grid_match",
    "status",
    "message",
]


def _require_rasterio() -> Any:
    """Import rasterio with a clear dependency error."""

    try:
        import rasterio  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "rasterio is required for streaming feature-stack writes. Install "
            "rasterio in the mbes_seaweed environment before running this stage."
        ) from exc
    return rasterio


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


def _resolution_from_profile(profile: Mapping[str, Any]) -> float | str:
    """Return the pixel size in metres from a raster profile."""

    transform = profile.get("transform")
    if hasattr(transform, "a"):
        return abs(float(transform.a))
    return ""


def _format_nodata(value: Any) -> str:
    """Return a stable CSV representation for nodata."""

    if value is None:
        return ""
    try:
        if np.isnan(value):
            return "nan"
    except TypeError:
        pass
    return str(value)


def _as_float32_nan(array: Any, nodata: int | float | None) -> np.ndarray:
    """Return one raster band as float32 with nodata converted to NaN."""

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


def _summarize_band(array: Any, nodata: int | float | None) -> dict[str, int | float | None]:
    """Summarize one input band and include valid fraction."""

    summary = summarize_raster(array, nodata=nodata)
    count = int(summary["count"])
    valid_count = int(summary["valid_count"])
    summary["valid_fraction"] = valid_count / count if count else 0.0
    return summary


def _flatten_feature_set(
    feature_set_id: str,
    feature_set_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Return feature-set metadata and ordered band definitions."""

    root = _require_mapping(feature_set_config, "feature_set_config")
    feature_set = dict(_require_mapping(root.get("feature_set"), "feature_set"))
    features = _require_mapping(root.get("features"), "features")

    declared_id = feature_set.get("id")
    if declared_id != feature_set_id:
        raise ValueError(
            f"Feature-set id mismatch: expected {feature_set_id!r}, found {declared_id!r}."
        )

    expected_count = int(feature_set.get("feature_count"))
    band_defs: list[dict[str, str]] = []
    for group_name in FEATURE_GROUP_ORDER:
        group_features = _require_list(features.get(group_name), f"features.{group_name}")
        for feature_name in group_features:
            if not isinstance(feature_name, str):
                raise ValueError(
                    f"features.{group_name} must contain only strings, "
                    f"got {type(feature_name).__name__}."
                )
            band_defs.append(
                {
                    "feature_group": group_name,
                    "feature_name": feature_name,
                }
            )

    if len(band_defs) != expected_count:
        raise ValueError(
            f"feature_count mismatch for feature_set_id={feature_set_id}: "
            f"feature_set.feature_count={expected_count}, "
            f"flattened feature count={len(band_defs)}."
        )

    return feature_set, band_defs


def _build_input_records(
    *,
    site_id: str,
    feature_set_id: str,
    base_features_root: Path,
    band_defs: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Return ordered input records for one site and feature set."""

    records: list[dict[str, Any]] = []
    site_base_dir = base_features_root / site_id
    for band_index, band_def in enumerate(band_defs, start=1):
        feature_name = band_def["feature_name"]
        records.append(
            {
                "site_id": site_id,
                "feature_set_id": feature_set_id,
                "band_index": band_index,
                "feature_group": band_def["feature_group"],
                "feature_name": feature_name,
                "input_path": site_base_dir / f"{feature_name}.tif",
            }
        )
    return records


def _validate_required_base_features(records: Sequence[Mapping[str, Any]]) -> None:
    """Raise when any required base feature raster is missing."""

    missing = []
    for record in records:
        input_path = Path(record["input_path"])
        if not input_path.exists():
            missing.append(
                "site_id={site_id}; feature_set_id={feature_set_id}; "
                "feature_name={feature_name}; expected path={path}".format(
                    site_id=record["site_id"],
                    feature_set_id=record["feature_set_id"],
                    feature_name=record["feature_name"],
                    path=input_path,
                )
            )

    if missing:
        raise FileNotFoundError(
            "Missing required base feature raster(s): " + " | ".join(missing)
        )


def _validate_single_band_inputs(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return input profiles after validating that each raster is single-band."""

    profiles: list[dict[str, Any]] = []
    invalid = []
    for record in records:
        input_path = Path(record["input_path"])
        profile = get_raster_profile(input_path)
        profiles.append(profile)

        band_count = int(profile.get("count", 0))
        if band_count != 1:
            invalid.append(
                "site_id={site_id}; feature_set_id={feature_set_id}; "
                "feature_name={feature_name}; path={path}; count={count}".format(
                    site_id=record["site_id"],
                    feature_set_id=record["feature_set_id"],
                    feature_name=record["feature_name"],
                    path=input_path,
                    count=band_count,
                )
            )

    if invalid:
        raise ValueError("Base feature rasters must be single-band: " + " | ".join(invalid))

    return profiles


def _validate_grid(
    profiles: Sequence[dict[str, Any]],
    *,
    site_id: str,
    feature_set_id: str,
) -> None:
    """Validate that all selected rasters share one grid."""

    try:
        check_same_grid(profiles)
    except ValueError as exc:
        raise ValueError(
            f"Raster grid mismatch for site_id={site_id}, "
            f"feature_set_id={feature_set_id}: {exc}"
        ) from exc


def _stream_write_stack(
    *,
    records: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    output_tif: Path,
    dtype: str = "float32",
    nodata: float = float("nan"),
) -> list[dict[str, int | float | None]]:
    """Write a feature stack one band at a time and return input summaries."""

    rasterio = _require_rasterio()
    if not records:
        raise ValueError("records must contain at least one band.")

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    output_profile = dict(profiles[0])
    output_profile.update(
        driver=output_profile.get("driver", "GTiff"),
        count=len(records),
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )

    summaries: list[dict[str, int | float | None]] = []
    with rasterio.open(output_tif, "w", **output_profile) as dst:
        for band_index, record in enumerate(records, start=1):
            input_path = Path(record["input_path"])
            array, input_profile = read_raster(input_path, band=1, masked=False)
            input_nodata = input_profile.get("nodata")
            summaries.append(_summarize_band(array, nodata=input_nodata))
            dst.write(_as_float32_nan(array, nodata=input_nodata), band_index)

    return summaries


def _summary_row(
    *,
    site_id: str,
    site_name: str,
    feature_set: Mapping[str, Any],
    band_count_written: int,
    record: Mapping[str, Any],
    profile: Mapping[str, Any],
    summary: Mapping[str, Any],
    output_tif: Path,
    bands_txt: Path,
) -> dict[str, Any]:
    """Build one band-level feature-stack summary row."""

    return {
        "site_id": site_id,
        "site_name": site_name,
        "feature_set_id": feature_set["id"],
        "feature_set_type": feature_set["type"],
        "feature_count_expected": int(feature_set["feature_count"]),
        "band_count_written": band_count_written,
        "band_index": record["band_index"],
        "feature_group": record["feature_group"],
        "feature_name": record["feature_name"],
        "input_path": str(record["input_path"]),
        "output_stack_path": str(output_tif),
        "bands_txt_path": str(bands_txt),
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "crs": str(profile["crs"]),
        "resolution_m": _resolution_from_profile(profile),
        "dtype": str(profile.get("dtype", "float32")),
        "nodata": _format_nodata(profile.get("nodata")),
        "valid_count": summary.get("valid_count"),
        "invalid_count": summary.get("invalid_count"),
        "valid_fraction": summary.get("valid_fraction"),
        "min": summary.get("min"),
        "max": summary.get("max"),
        "mean": summary.get("mean"),
        "std": summary.get("std"),
        "grid_match": True,
        "status": "ok",
        "message": "",
    }


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one feature-set summary CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def process_feature_set(
    *,
    site_id: str,
    site_config: Mapping[str, Any],
    feature_set_id: str,
    feature_set_config: Mapping[str, Any],
    base_features_root: Path,
    feature_sets_root: Path,
) -> Path:
    """Build one feature stack for one site and return its summary path."""

    print(f"Building feature stack: {site_id} / {feature_set_id}", flush=True)
    site_name = str(site_config.get("site_name", site_id))
    feature_set, band_defs = _flatten_feature_set(feature_set_id, feature_set_config)

    records = _build_input_records(
        site_id=site_id,
        feature_set_id=feature_set_id,
        base_features_root=base_features_root,
        band_defs=band_defs,
    )
    _validate_required_base_features(records)
    profiles = _validate_single_band_inputs(records)
    _validate_grid(profiles, site_id=site_id, feature_set_id=feature_set_id)

    output_dir = feature_sets_root / site_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tif = output_dir / f"{feature_set_id}.tif"
    bands_txt = output_dir / f"{feature_set_id}.bands.txt"
    summary_path = output_dir / f"{feature_set_id}_summary.csv"

    summaries = _stream_write_stack(
        records=records,
        profiles=profiles,
        output_tif=output_tif,
    )

    band_names = [str(record["feature_name"]) for record in records]
    write_band_names(output_tif, band_names)
    write_band_names(bands_txt, band_names)

    rows = [
        _summary_row(
            site_id=site_id,
            site_name=site_name,
            feature_set=feature_set,
            band_count_written=len(records),
            record=record,
            profile=profile,
            summary=summary,
            output_tif=output_tif,
            bands_txt=bands_txt,
        )
        for record, profile, summary in zip(records, profiles, summaries, strict=True)
    ]
    _write_summary(summary_path, rows)
    print(f"Built feature stack: {site_id} / {feature_set_id}", flush=True)
    return summary_path


def run_feature_sets(project_id: str = DEFAULT_PROJECT_ID) -> list[Path]:
    """Build all configured feature stacks for all configured sites."""

    project_config = load_project_config(project_id=project_id)
    sites_config = load_sites_config(project_id=project_id)
    experiment_matrix = load_experiment_matrix(project_id=project_id)
    feature_set_configs = load_all_feature_set_configs(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="03")

    feature_set_ids = _require_list(
        experiment_matrix.get("feature_sets"),
        "experiment_matrix.feature_sets",
    )
    sites = _require_mapping(sites_config.get("sites"), "sites_config.sites")

    summary_paths: list[Path] = []
    for site_id in sites:
        site_id_str = str(site_id)
        print(f"Processing site: {site_id_str}", flush=True)
        site_config = get_site_config(sites_config, site_id_str)
        for feature_set_id in feature_set_ids:
            if not isinstance(feature_set_id, str):
                raise ValueError("experiment_matrix.feature_sets must contain only strings.")
            if feature_set_id not in feature_set_configs:
                raise KeyError(f"Feature-set config was not loaded: {feature_set_id}")

            summary_paths.append(
                process_feature_set(
                    site_id=site_id_str,
                    site_config=site_config,
                    feature_set_id=feature_set_id,
                    feature_set_config=feature_set_configs[feature_set_id],
                    base_features_root=output_dirs["base_features"],
                    feature_sets_root=output_dirs["feature_sets"],
                )
            )
        print(f"Completed site: {site_id_str}", flush=True)

    return summary_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build configured project feature-set stacks from base feature rasters.",
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=f"Project id to process. Default: {DEFAULT_PROJECT_ID}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the feature-set stack stage from the command line."""

    args = parse_args()
    summary_paths = run_feature_sets(project_id=args.project_id)
    for summary_path in summary_paths:
        print(f"Feature-set summary written to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Raster and vector I/O helpers for MBES habitat processing.

This module supports the alignment-stage responsibilities only: depth and
backscatter rasters, crop masks, common valid masks, grid checks, and raster
stacking. Label shapefile rasterization is intentionally left to
``04_sampling.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence


def _require_numpy() -> Any:
    """Import NumPy with a clear dependency error."""

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "NumPy is required for raster array operations. Install it with "
            "`pip install numpy` or a project environment file."
        ) from exc
    return np


def _require_rasterio() -> Any:
    """Import rasterio with a clear dependency error."""

    try:
        import rasterio  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "rasterio is required for raster I/O and alignment. Install it with "
            "`pip install rasterio` or a conda/geospatial environment."
        ) from exc
    return rasterio


def _require_geopandas() -> Any:
    """Import GeoPandas with a clear dependency error."""

    try:
        import geopandas as gpd  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "GeoPandas is required for vector I/O. Install it with "
            "`pip install geopandas` or a conda/geospatial environment."
        ) from exc
    return gpd


def _as_path(path: str | Path) -> Path:
    """Return ``path`` as an expanded ``Path``."""

    return Path(path).expanduser()


def _path_suffix(path: str | Path) -> str:
    """Return the lower-case suffix for ``path``."""

    return _as_path(path).suffix.lower()


def _is_tiff_path(path: str | Path) -> bool:
    """Return ``True`` for GeoTIFF-like file paths."""

    return _path_suffix(path) in {".tif", ".tiff"}


def _is_txt_path(path: str | Path) -> bool:
    """Return ``True`` for plain-text band-name files."""

    return _path_suffix(path) == ".txt"


def _ensure_file(path: str | Path, label: str = "file") -> Path:
    """Return an existing file path or raise ``FileNotFoundError``."""

    checked = _as_path(path)
    if not checked.exists():
        raise FileNotFoundError(f"{label} not found: {checked}")
    if not checked.is_file():
        raise ValueError(f"{label} is not a file: {checked}")
    return checked


def _resampling_enum(name: str | Any) -> Any:
    """Return a rasterio ``Resampling`` enum from a string or enum value."""

    rasterio = _require_rasterio()
    from rasterio.enums import Resampling

    if isinstance(name, Resampling):
        return name
    if not isinstance(name, str):
        raise ValueError(f"Resampling must be a string or Resampling enum, got {type(name).__name__}.")

    try:
        return getattr(Resampling, name)
    except AttributeError as exc:
        allowed = ", ".join(item.name for item in Resampling)
        raise ValueError(f"Unknown resampling method {name!r}. Allowed: {allowed}") from exc


def _dtype_for_nodata(source_dtype: str | Any, nodata: int | float) -> str:
    """Return an output dtype that can store source values and ``nodata``."""

    np = _require_numpy()
    dtype = np.dtype(source_dtype)

    if dtype.kind == "f":
        return dtype.name

    if dtype.kind in {"i", "u"}:
        source_info = np.iinfo(dtype)
        source_min = source_info.min
        source_max = source_info.max
        nodata_value = float(nodata)

        candidates = [dtype, np.dtype("int16"), np.dtype("int32"), np.dtype("int64")]
        for candidate in candidates:
            if candidate.kind not in {"i", "u"}:
                continue
            candidate_info = np.iinfo(candidate)
            if (
                candidate_info.min <= nodata_value <= candidate_info.max
                and candidate_info.min <= source_min
                and source_max <= candidate_info.max
            ):
                return candidate.name

        return "float64"

    return "float32"


def _normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return a compact grid profile used for comparisons and writing."""

    required = ["crs", "transform", "width", "height"]
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"Raster profile is missing key(s): {', '.join(missing)}.")

    return {
        "crs": profile["crs"],
        "transform": profile["transform"],
        "width": int(profile["width"]),
        "height": int(profile["height"]),
    }


def open_raster(path: str | Path) -> Any:
    """Open a raster dataset for reading.

    The caller owns the returned dataset context and should close it, preferably
    with ``with open_raster(path) as src:``.
    """

    rasterio = _require_rasterio()
    raster_path = _ensure_file(path, "Raster")
    return rasterio.open(raster_path)


def read_raster(
    path: str | Path,
    band: int | None = 1,
    masked: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Read raster data and return ``(array, profile)``.

    Parameters
    ----------
    path:
        Raster path.
    band:
        Band number to read. Use ``None`` to read all bands.
    masked:
        If ``True``, return a NumPy masked array where rasterio can infer nodata.
    """

    with open_raster(path) as src:
        if band is None:
            array = src.read(masked=masked)
        else:
            if band < 1 or band > src.count:
                raise ValueError(f"Band {band} is out of range for {path}; count={src.count}.")
            array = src.read(band, masked=masked)
        profile = src.profile.copy()
    return array, profile


def get_raster_profile(path: str | Path) -> dict[str, Any]:
    """Return the rasterio profile for ``path``."""

    with open_raster(path) as src:
        return src.profile.copy()


def write_raster(
    path: str | Path,
    array: Any,
    profile: Mapping[str, Any],
    nodata: int | float | None = None,
    dtype: str | None = None,
    compress: str | None = "deflate",
) -> Path:
    """Write a 2D or 3D array to a GeoTIFF-like raster.

    Arrays are expected as ``(rows, cols)`` or ``(bands, rows, cols)``.
    """

    rasterio = _require_rasterio()
    np = _require_numpy()

    if not isinstance(profile, Mapping):
        raise ValueError("profile must be a mapping/rasterio profile.")

    output_path = _as_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = array
    output_nodata = nodata if nodata is not None else profile.get("nodata")
    if np.ma.isMaskedArray(data):
        if output_nodata is None:
            data = np.ma.asarray(data, dtype="float64").filled(np.nan)
        else:
            data = data.filled(output_nodata)

    data = np.asarray(data)
    if data.ndim == 2:
        write_data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        write_data = data
    else:
        raise ValueError(f"array must be 2D or 3D, got shape {data.shape}.")

    output_dtype = dtype or str(write_data.dtype)
    out_profile = dict(profile)
    out_profile.update(
        driver=out_profile.get("driver", "GTiff"),
        height=int(write_data.shape[1]),
        width=int(write_data.shape[2]),
        count=int(write_data.shape[0]),
        dtype=output_dtype,
        nodata=output_nodata,
    )
    if compress and out_profile["driver"] == "GTiff":
        out_profile["compress"] = compress

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(write_data.astype(output_dtype, copy=False))

    return output_path


def esri_ascii_header(
    width: int,
    height: int,
    transform: Any,
    nodata: int | float,
) -> str:
    """Return a north-to-south ESRI ASCII Grid header for square pixels."""

    cellsize_x = float(transform.a)
    cellsize_y = abs(float(transform.e))
    if abs(cellsize_x - cellsize_y) > max(1e-9, abs(cellsize_x) * 1e-6):
        raise ValueError(
            f"ASCII export requires square pixels, got {cellsize_x} x {cellsize_y}."
        )
    xllcorner = float(transform.c)
    yllcorner = float(transform.f) + float(transform.e) * int(height)
    return (
        f"ncols {int(width)}\n"
        f"nrows {int(height)}\n"
        f"xllcorner {xllcorner:.12f}\n"
        f"yllcorner {yllcorner:.12f}\n"
        f"cellsize {cellsize_x:.12f}\n"
        f"NODATA_value {nodata}\n"
    )


def write_esri_ascii_pair(
    src_path: str | Path,
    asc_path: str | Path,
    txt_path: str | Path,
    *,
    nodata: int | float,
    integer: bool = False,
    precision: int = 6,
    chunk_rows: int = 256,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write identical ASC/TXT grids from a native raster without resampling.

    Rows are written from the raster's upper (north) edge to its lower (south)
    edge, matching ESRI ASCII Grid conventions. The source CRS, transform,
    width, height, and native resolution are preserved exactly.
    """

    rasterio = _require_rasterio()
    np = _require_numpy()
    source = _ensure_file(src_path, "source raster")
    asc = _as_path(asc_path)
    txt = _as_path(txt_path)
    existing = [path for path in (asc, txt) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "ASCII output exists and overwrite is disabled: "
            + "; ".join(str(path) for path in existing)
        )
    asc.parent.mkdir(parents=True, exist_ok=True)
    txt.parent.mkdir(parents=True, exist_ok=True)
    temporary = asc.with_name(f".{asc.name}.tmp")
    fmt = "%d" if integer else f"%.{int(precision)}f"
    try:
        with rasterio.open(source) as dataset, temporary.open("w", encoding="utf-8") as stream:
            stream.write(esri_ascii_header(dataset.width, dataset.height, dataset.transform, nodata))
            for row_off in range(0, dataset.height, int(chunk_rows)):
                height = min(int(chunk_rows), dataset.height - row_off)
                window = rasterio.windows.Window(0, row_off, dataset.width, height)
                chunk = dataset.read(1, window=window, masked=True)
                data = np.ma.asarray(chunk).filled(nodata)
                if integer:
                    output = np.where(np.isfinite(data), data, nodata).astype("int32", copy=False)
                else:
                    output = np.where(np.isfinite(data), data, nodata).astype("float32", copy=False)
                np.savetxt(stream, output, fmt=fmt, delimiter=" ")
        temporary.replace(asc)
        shutil.copy2(asc, txt)
    finally:
        if temporary.exists():
            temporary.unlink()
    with rasterio.open(source) as dataset:
        return {
            "source_path": str(source),
            "asc_path": str(asc),
            "txt_path": str(txt),
            "width": int(dataset.width),
            "height": int(dataset.height),
            "crs": str(dataset.crs),
            "resolution_x": float(dataset.res[0]),
            "resolution_y": float(dataset.res[1]),
            "row_order": "north_to_south",
            "nodata": nodata,
            "precision": 0 if integer else int(precision),
            "resampling": "none_native_grid",
        }


def read_vector(path: str | Path) -> Any:
    """Read a vector dataset with GeoPandas and return a GeoDataFrame."""

    gpd = _require_geopandas()
    vector_path = _ensure_file(path, "Vector")
    gdf = gpd.read_file(vector_path)

    if gdf.empty:
        raise ValueError(f"Vector dataset contains no features: {vector_path}")
    if gdf.geometry.isna().all():
        raise ValueError(f"Vector dataset contains no valid geometry: {vector_path}")

    return gdf


def reproject_vector(gdf: Any, dst_crs: Any) -> Any:
    """Reproject a GeoDataFrame to ``dst_crs``."""

    if not hasattr(gdf, "to_crs"):
        raise ValueError("gdf must be a GeoPandas GeoDataFrame-like object.")
    if getattr(gdf, "crs", None) is None:
        raise ValueError("Vector data has no CRS and cannot be reprojected safely.")
    if dst_crs is None:
        raise ValueError("dst_crs must not be None.")
    if str(gdf.crs) == str(dst_crs):
        return gdf
    return gdf.to_crs(dst_crs)


def build_reference_grid(
    bounds: Sequence[float],
    resolution: float,
    crs: Any,
    dtype: str = "float32",
    nodata: int | float | None = None,
) -> dict[str, Any]:
    """Build a rasterio profile from bounds, pixel size, and CRS.

    ``bounds`` must be ``(minx, miny, maxx, maxy)`` in the output CRS.
    """

    if len(bounds) != 4:
        raise ValueError("bounds must contain four values: minx, miny, maxx, maxy.")
    if resolution <= 0:
        raise ValueError("resolution must be > 0.")
    if crs is None:
        raise ValueError("crs must not be None.")

    rasterio = _require_rasterio()
    minx, miny, maxx, maxy = [float(value) for value in bounds]
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"Invalid bounds: {bounds}")

    width = int(ceil((maxx - minx) / resolution))
    height = int(ceil((maxy - miny) / resolution))
    transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)

    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }


def crop_raster_to_polygon(
    src_path: str | Path,
    polygon_path: str | Path,
    filled: bool = True,
    crop: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Crop a raster to a polygon and return ``(array, profile)``.

    This is a geometric crop helper only. It does not rasterize labels.
    """

    rasterio = _require_rasterio()
    from rasterio.mask import mask

    gdf = read_vector(polygon_path)
    with open_raster(src_path) as src:
        gdf = reproject_vector(gdf, src.crs)
        geometries = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
        if not geometries:
            raise ValueError(f"No non-empty crop geometries found in {polygon_path}.")

        out_image, out_transform = mask(src, geometries, crop=crop, filled=filled)
        out_profile = src.profile.copy()
        out_profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
        )

    return out_image, out_profile


def reproject_raster_to_grid(
    src_path: str | Path,
    grid_profile: dict[str, Any],
    resampling: str | Any = "bilinear",
    dst_nodata: int | float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Reproject a raster to a target grid profile."""

    np = _require_numpy()
    rasterio = _require_rasterio()
    from rasterio.warp import reproject

    _normalize_profile(grid_profile)
    resampling_method = _resampling_enum(resampling)

    with open_raster(src_path) as src:
        dst_count = src.count
        dst_dtype = grid_profile.get("dtype", src.dtypes[0])
        output_nodata = dst_nodata
        if output_nodata is None:
            output_nodata = grid_profile.get("nodata", src.nodata)

        destination = np.full(
            (dst_count, int(grid_profile["height"]), int(grid_profile["width"])),
            output_nodata if output_nodata is not None else 0,
            dtype=dst_dtype,
        )

        for band_index in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_index),
                destination=destination[band_index - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=grid_profile["transform"],
                dst_crs=grid_profile["crs"],
                dst_nodata=output_nodata,
                resampling=resampling_method,
            )

        output_profile = src.profile.copy()
        output_profile.update(
            height=int(grid_profile["height"]),
            width=int(grid_profile["width"]),
            transform=grid_profile["transform"],
            crs=grid_profile["crs"],
            count=dst_count,
            dtype=str(destination.dtype),
            nodata=output_nodata,
        )

    if dst_count == 1:
        return destination[0], output_profile
    return destination, output_profile


def align_raster_to_template(
    src_path: str | Path,
    template: str | Path | dict[str, Any],
    resampling: str | Any = "bilinear",
    dst_nodata: int | float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Align a raster to a template raster path or rasterio profile."""

    template_profile = get_raster_profile(template) if isinstance(template, (str, Path)) else template
    return reproject_raster_to_grid(
        src_path=src_path,
        grid_profile=template_profile,
        resampling=resampling,
        dst_nodata=dst_nodata,
    )


def make_crop_mask(
    vector_path: str | Path,
    template_profile: dict[str, Any],
    all_touched: bool = False,
) -> Any:
    """Rasterize a crop polygon to a boolean mask on ``template_profile``.

    Pixels inside the crop polygon are ``True``. This function is only for crop
    masks and does not rasterize label attributes.
    """

    np = _require_numpy()
    _require_rasterio()
    from rasterio.features import rasterize

    profile = _normalize_profile(template_profile)
    gdf = read_vector(vector_path)
    gdf = reproject_vector(gdf, profile["crs"])
    geometries = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        raise ValueError(f"No non-empty crop geometries found in {vector_path}.")

    mask = rasterize(
        ((geom, 1) for geom in geometries),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    return mask.astype(bool, copy=False)


def make_valid_data_mask(
    arrays: Any | Sequence[Any],
    nodata_values: int | float | Sequence[int | float | None] | None = None,
) -> Any:
    """Return a common boolean valid-data mask across arrays.

    A pixel is valid when it is finite, not masked, and not equal to its nodata
    value in every supplied array/band.
    """

    np = _require_numpy()

    array_list = list(arrays) if isinstance(arrays, (list, tuple)) else [arrays]
    if not array_list:
        raise ValueError("arrays must contain at least one array.")

    if isinstance(nodata_values, (list, tuple)):
        nodata_list = list(nodata_values)
        if len(nodata_list) != len(array_list):
            raise ValueError("nodata_values length must match arrays length.")
    else:
        nodata_list = [nodata_values] * len(array_list)

    common_mask: Any | None = None
    for array, nodata in zip(array_list, nodata_list):
        if np.ma.isMaskedArray(array):
            invalid = np.ma.getmaskarray(array)
            data = np.asarray(array.filled(np.nan if np.issubdtype(array.dtype, np.floating) else 0))
        else:
            data = np.asarray(array)
            invalid = np.zeros(data.shape, dtype=bool)

        valid = np.isfinite(data) & ~invalid
        if nodata is not None:
            valid &= data != nodata

        if valid.ndim == 3:
            valid = valid.all(axis=0)
        elif valid.ndim != 2:
            raise ValueError(f"Each array must be 2D or 3D, got shape {data.shape}.")

        common_mask = valid if common_mask is None else (common_mask & valid)

    return common_mask


def check_same_grid(
    rasters_or_profiles: Sequence[str | Path | dict[str, Any]],
    raise_on_mismatch: bool = True,
) -> bool:
    """Check that rasters/profiles share CRS, transform, width, and height."""

    if len(rasters_or_profiles) < 2:
        return True

    profiles = [
        get_raster_profile(item) if isinstance(item, (str, Path)) else item
        for item in rasters_or_profiles
    ]
    reference = _normalize_profile(profiles[0])

    for index, profile in enumerate(profiles[1:], start=1):
        current = _normalize_profile(profile)
        mismatch: list[str] = []
        for key in ["crs", "width", "height"]:
            if str(current[key]) != str(reference[key]):
                mismatch.append(key)
        transform_matches = (
            current["transform"].almost_equals(reference["transform"])
            if hasattr(current["transform"], "almost_equals")
            else current["transform"] == reference["transform"]
        )
        if not transform_matches:
            mismatch.append("transform")

        if mismatch:
            if raise_on_mismatch:
                raise ValueError(
                    f"Raster grid mismatch at item {index}: {', '.join(mismatch)}."
                )
            return False

    return True


def stack_rasters(
    raster_paths: Sequence[str | Path],
    feature_names: Sequence[str] | None = None,
    masked: bool = True,
) -> tuple[Any, dict[str, Any], list[str]]:
    """Read single-band rasters on the same grid into a 3D stack."""

    np = _require_numpy()
    if not raster_paths:
        raise ValueError("raster_paths must contain at least one raster.")
    if feature_names is not None and len(feature_names) != len(raster_paths):
        raise ValueError("feature_names length must match raster_paths length.")

    check_same_grid(list(raster_paths))

    arrays = []
    profiles = []
    for path in raster_paths:
        array, profile = read_raster(path, band=1, masked=masked)
        arrays.append(array)
        profiles.append(profile)

    stack = np.ma.stack(arrays, axis=0) if any(np.ma.isMaskedArray(a) for a in arrays) else np.stack(arrays, axis=0)
    output_profile = profiles[0].copy()
    output_profile.update(count=len(raster_paths), dtype=str(stack.dtype))

    names = list(feature_names) if feature_names is not None else [
        Path(path).stem for path in raster_paths
    ]
    return stack, output_profile, names


def write_band_names(path: str | Path, band_names: Sequence[str]) -> None:
    """Write band names to a GeoTIFF or ``bands.txt`` sidecar file.

    GeoTIFF paths use raster band descriptions. ``.txt`` paths use the legacy
    pipeline format ``01. feature_name``.
    """

    if not band_names:
        raise ValueError("band_names must contain at least one name.")

    output_path = _as_path(path)

    if _is_txt_path(output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{index:02d}. {name}" for index, name in enumerate(band_names, start=1)]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    if _is_tiff_path(output_path):
        rasterio = _require_rasterio()
        raster_path = _ensure_file(output_path, "Raster")

        with rasterio.open(raster_path, "r+") as dst:
            if len(band_names) != dst.count:
                raise ValueError(
                    f"band_names length ({len(band_names)}) must match raster count ({dst.count})."
                )
            for band_index, name in enumerate(band_names, start=1):
                dst.set_band_description(band_index, str(name))
        return

    raise ValueError(f"Unsupported band-name path extension: {output_path.suffix}")


def read_band_names(path: str | Path) -> list[str]:
    """Read band names from a GeoTIFF or ``bands.txt`` sidecar file."""

    input_path = _as_path(path)

    if _is_txt_path(input_path):
        text_path = _ensure_file(input_path, "Band names text file")
        names: list[str] = []
        for line_number, raw_line in enumerate(text_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            prefix, separator, name = line.partition(".")
            if not separator or not prefix.isdigit() or not name.strip():
                raise ValueError(
                    f"Invalid band-name line {line_number} in {text_path}: {raw_line!r}"
                )
            names.append(name.strip())

        if not names:
            raise ValueError(f"Band names text file is empty: {text_path}")
        return names

    if _is_tiff_path(input_path):
        with open_raster(input_path) as src:
            names = []
            for band_index, description in enumerate(src.descriptions, start=1):
                names.append(description or f"band_{band_index}")
        return names

    raise ValueError(f"Unsupported band-name path extension: {input_path.suffix}")


def array_to_nan(array: Any, nodata: int | float | None = None) -> Any:
    """Return a float array where nodata and masked pixels are ``NaN``."""

    np = _require_numpy()

    if np.ma.isMaskedArray(array):
        result = np.ma.asarray(array, dtype="float64").filled(np.nan)
    else:
        result = np.asarray(array, dtype="float64").copy()

    if nodata is not None:
        result[result == nodata] = np.nan

    return result


def nan_to_nodata(array: Any, nodata: int | float, dtype: str | None = None) -> Any:
    """Return an array where non-finite values are replaced by ``nodata``."""

    np = _require_numpy()
    result = np.asarray(array).copy()
    result = np.where(np.isfinite(result), result, nodata)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def summarize_raster(
    array: Any,
    nodata: int | float | None = None,
) -> dict[str, int | float | None]:
    """Summarize valid raster values while ignoring nodata and ``NaN``."""

    np = _require_numpy()
    data = array_to_nan(array, nodata=nodata)
    valid = np.isfinite(data)
    valid_values = data[valid]

    summary: dict[str, int | float | None] = {
        "count": int(data.size),
        "valid_count": int(valid.sum()),
        "invalid_count": int(data.size - valid.sum()),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    if valid_values.size:
        summary.update(
            {
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values)),
            }
        )
    return summary


def resample_to_asc(
    src_path: str | Path,
    dst_path: str | Path,
    resolution: float | None = None,
    template_profile: dict[str, Any] | None = None,
    resampling: str | Any = "bilinear",
    nodata: int | float = -9999,
) -> Path:
    """Resample a raster and write the first band as an ESRI ASCII grid.

    Source nodata values are converted to the requested ASC ``NODATA_VALUE``.
    Use ``resampling='bilinear'`` for probability maps and
    ``resampling='nearest'`` for binary maps.
    """

    rasterio = _require_rasterio()
    np = _require_numpy()
    output_path = _as_path(dst_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if template_profile is not None and resolution is not None:
        raise ValueError("Provide either template_profile or resolution, not both.")

    with open_raster(src_path) as src:
        source_nodata = src.nodata
        output_dtype = _dtype_for_nodata(src.dtypes[0], nodata)

        if template_profile is not None:
            target_profile = template_profile.copy()
        elif resolution is not None:
            target_profile = build_reference_grid(
                bounds=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                resolution=resolution,
                crs=src.crs,
                dtype=output_dtype,
                nodata=nodata,
            )
        else:
            target_profile = src.profile.copy()
        target_profile.update(dtype=output_dtype, nodata=nodata, count=1)

    array, profile = reproject_raster_to_grid(
        src_path=src_path,
        grid_profile=target_profile,
        resampling=resampling,
        dst_nodata=nodata,
    )
    if np.asarray(array).ndim == 3:
        array = array[0]

    array = np.asarray(array)
    if source_nodata is not None and source_nodata != nodata:
        if np.issubdtype(array.dtype, np.floating):
            source_nodata_mask = np.isclose(array, float(source_nodata), equal_nan=False)
        else:
            source_nodata_mask = array == source_nodata
        array = np.where(source_nodata_mask, nodata, array)
    array = np.where(np.isfinite(array), array, nodata).astype(output_dtype, copy=False)

    asc_profile = profile.copy()
    asc_profile.update(driver="AAIGrid", count=1, dtype=str(array.dtype), nodata=nodata)
    with rasterio.open(output_path, "w", **asc_profile) as dst:
        dst.write(array, 1)

    return output_path

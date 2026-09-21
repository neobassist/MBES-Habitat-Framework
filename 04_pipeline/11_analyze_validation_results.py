#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analyze configured validation summaries across validation strategies.

This stage reads summary CSV files produced by within-site, cross-site, and
spatial block CV evaluation stages, then writes comparison tables under
``analysis_compare``. It does not retrain models, generate predictions, or
modify existing evaluation outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    configured_analysis_candidates,
    configured_focused_model_comparisons,
    configured_site_rank_comparisons,
    configured_validation_sources,
    ensure_output_dirs,
    load_analysis_config,
    load_project_config,
)


METRIC_COLUMNS = ["AP_mean", "AP_sd", "AUC_mean", "AUC_sd", "F1_mean", "F1_sd"]
WITHIN_SPATIAL_REQUIRED_COLUMNS = [
    "validation_type",
    "site_id",
    "feature_set",
    "model_id",
    *METRIC_COLUMNS,
]
CROSS_REQUIRED_COLUMNS = [
    "validation_type",
    "train_site_id",
    "test_site_id",
    "site_id",
    "feature_set",
    "model_id",
    *METRIC_COLUMNS,
]
VALIDATION_AP_COLUMNS = ["within_AP", "cross_AP", "spatial_AP"]
INTEGRATED_STAT_COLUMNS = [
    "within_AP",
    "cross_AP",
    "spatial_AP",
    "mean_AP",
    "median_AP",
    "min_AP",
    "max_AP",
    "range_AP",
    "std_AP",
    "cv_AP",
    "skew_AP",
    "within_minus_cross",
    "within_minus_spatial",
    "spatial_minus_cross",
]
CANDIDATE_STAT_COLUMNS = [
    "feature_set",
    "model_id",
    "within_AP",
    "cross_AP",
    "spatial_AP",
    "mean_AP",
    "median_AP",
    "min_AP",
    "max_AP",
    "range_AP",
    "std_AP",
    "cv_AP",
    "within_minus_cross",
    "within_minus_spatial",
    "spatial_minus_cross",
]


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    """Read a required non-empty CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} is empty: {path}") from exc
    if data.empty:
        raise ValueError(f"{label} contains zero rows: {path}")
    return data


def _validate_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    """Validate required columns."""

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")


def _read_inputs(
    project_id: str,
    analysis_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """Read within-site, cross-site, and spatial-block summary CSV files."""

    project_config = load_project_config(project_id=project_id)
    output_dirs = ensure_output_dirs(project_config, stage_name="11")
    output_root = output_dirs["output_root"]

    source_specs = {
        "within_site": (output_dirs["within_site"] / "within_site_summary.csv", WITHIN_SPATIAL_REQUIRED_COLUMNS),
        "cross_site": (output_dirs["cross_site"] / "cross_site_summary.csv", CROSS_REQUIRED_COLUMNS),
        "spatial_block_cv": (
            output_dirs["spatial_block_cv"] / "spatial_block_cv_summary.csv",
            WITHIN_SPATIAL_REQUIRED_COLUMNS,
        ),
    }
    loaded: dict[str, pd.DataFrame] = {}
    for source in configured_validation_sources(analysis_config):
        source_id = source["id"]
        path, required_columns = source_specs[source_id]
        data = _read_csv(path, path.name)
        _validate_columns(data, required_columns, path.name)
        loaded[source_id] = data

    within = loaded["within_site"]
    cross = loaded["cross_site"]
    spatial = loaded["spatial_block_cv"]

    for label, data in [
        ("within_site_summary.csv", within),
        ("cross_site_summary.csv", cross),
        ("spatial_block_cv_summary.csv", spatial),
    ]:
        for column in METRIC_COLUMNS:
            values = pd.to_numeric(data[column], errors="coerce")
            if not np.isfinite(values.to_numpy(dtype="float64", copy=False)).all():
                raise ValueError(f"{label} has non-finite numeric values in {column}.")
            data[column] = values

    return within, cross, spatial, output_root


def _write_csv(df: pd.DataFrame, out_path: Path, *, overwrite: bool) -> None:
    """Write a CSV, respecting overwrite protection."""

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists and overwrite=False: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _rank_table(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    out_path: Path,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    """Write a grouped metric rank table sorted by AP_mean descending."""

    grouped = (
        df.groupby(list(group_cols), dropna=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
        .sort_values(["AP_mean", *group_cols], ascending=[False, *([True] * len(group_cols))])
        .reset_index(drop=True)
    )
    grouped.insert(0, "rank", np.arange(1, grouped.shape[0] + 1, dtype=int))
    _write_csv(grouped, out_path, overwrite=overwrite)
    return grouped


def _ap_by_validation(df: pd.DataFrame, group_cols: Sequence[str], output_column: str) -> pd.DataFrame:
    """Aggregate AP_mean by group for one validation strategy."""

    return (
        df.groupby(list(group_cols), dropna=False)["AP_mean"]
        .mean()
        .reset_index()
        .rename(columns={"AP_mean": output_column})
    )


def _merge_validation_ap(
    within: pd.DataFrame,
    cross: pd.DataFrame,
    spatial: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Merge validation AP summaries for common comparison columns."""

    merged = _ap_by_validation(within, group_cols, "within_AP")
    merged = merged.merge(
        _ap_by_validation(cross, group_cols, "cross_AP"),
        on=list(group_cols),
        how="outer",
    )
    merged = merged.merge(
        _ap_by_validation(spatial, group_cols, "spatial_AP"),
        on=list(group_cols),
        how="outer",
    )
    return merged


def _add_integrated_comparison_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add first-stage integrated AP comparison columns."""

    out = df.copy()
    values = out[VALIDATION_AP_COLUMNS]
    out["mean_AP_3validations"] = values.mean(axis=1, skipna=False)
    out["min_AP_3validations"] = values.min(axis=1, skipna=False)
    out["within_minus_spatial"] = out["within_AP"] - out["spatial_AP"]
    out["within_minus_cross"] = out["within_AP"] - out["cross_AP"]
    return out


def _integrated_comparison(
    within: pd.DataFrame,
    cross: pd.DataFrame,
    spatial: pd.DataFrame,
    group_cols: Sequence[str],
    out_path: Path,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    """Write first-stage integrated comparison table."""

    out = _add_integrated_comparison_columns(
        _merge_validation_ap(within, cross, spatial, group_cols)
    )
    sort_cols = ["mean_AP_3validations", *group_cols]
    out = (
        out.sort_values(sort_cols, ascending=[False, *([True] * len(group_cols))], na_position="last")
        .reset_index(drop=True)
    )
    out.insert(0, "rank", np.arange(1, out.shape[0] + 1, dtype=int))
    _write_csv(out, out_path, overwrite=overwrite)
    return out


def _run_stage1_analysis(
    within: pd.DataFrame,
    cross: pd.DataFrame,
    spatial: pd.DataFrame,
    out_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, pd.DataFrame]:
    """Write stage-1 rank and integrated comparison tables."""

    outputs: dict[str, pd.DataFrame] = {}

    outputs["within_by_site_model"] = _rank_table(
        within,
        ["site_id", "model_id"],
        out_dir / "within_by_site_model.csv",
        overwrite=overwrite,
    )
    outputs["within_by_site_feature"] = _rank_table(
        within,
        ["site_id", "feature_set"],
        out_dir / "within_by_site_feature.csv",
        overwrite=overwrite,
    )
    outputs["within_by_site_feature_model"] = _rank_table(
        within,
        ["site_id", "feature_set", "model_id"],
        out_dir / "within_by_site_feature_model.csv",
        overwrite=overwrite,
    )
    outputs["within_overall_model"] = _rank_table(
        within,
        ["model_id"],
        out_dir / "within_overall_model.csv",
        overwrite=overwrite,
    )
    outputs["within_overall_feature"] = _rank_table(
        within,
        ["feature_set"],
        out_dir / "within_overall_feature.csv",
        overwrite=overwrite,
    )
    outputs["within_overall_feature_model"] = _rank_table(
        within,
        ["feature_set", "model_id"],
        out_dir / "within_overall_feature_model.csv",
        overwrite=overwrite,
    )

    cross_direction_cols = ["train_site_id", "test_site_id", "site_id"]
    outputs["cross_by_direction_model"] = _rank_table(
        cross,
        [*cross_direction_cols, "model_id"],
        out_dir / "cross_by_direction_model.csv",
        overwrite=overwrite,
    )
    outputs["cross_by_direction_feature"] = _rank_table(
        cross,
        [*cross_direction_cols, "feature_set"],
        out_dir / "cross_by_direction_feature.csv",
        overwrite=overwrite,
    )
    outputs["cross_by_direction_feature_model"] = _rank_table(
        cross,
        [*cross_direction_cols, "feature_set", "model_id"],
        out_dir / "cross_by_direction_feature_model.csv",
        overwrite=overwrite,
    )
    outputs["cross_overall_model"] = _rank_table(
        cross,
        ["model_id"],
        out_dir / "cross_overall_model.csv",
        overwrite=overwrite,
    )
    outputs["cross_overall_feature"] = _rank_table(
        cross,
        ["feature_set"],
        out_dir / "cross_overall_feature.csv",
        overwrite=overwrite,
    )
    outputs["cross_overall_feature_model"] = _rank_table(
        cross,
        ["feature_set", "model_id"],
        out_dir / "cross_overall_feature_model.csv",
        overwrite=overwrite,
    )

    outputs["spatial_by_site_model"] = _rank_table(
        spatial,
        ["site_id", "model_id"],
        out_dir / "spatial_by_site_model.csv",
        overwrite=overwrite,
    )
    outputs["spatial_by_site_feature"] = _rank_table(
        spatial,
        ["site_id", "feature_set"],
        out_dir / "spatial_by_site_feature.csv",
        overwrite=overwrite,
    )
    outputs["spatial_by_site_feature_model"] = _rank_table(
        spatial,
        ["site_id", "feature_set", "model_id"],
        out_dir / "spatial_by_site_feature_model.csv",
        overwrite=overwrite,
    )
    outputs["spatial_overall_model"] = _rank_table(
        spatial,
        ["model_id"],
        out_dir / "spatial_overall_model.csv",
        overwrite=overwrite,
    )
    outputs["spatial_overall_feature"] = _rank_table(
        spatial,
        ["feature_set"],
        out_dir / "spatial_overall_feature.csv",
        overwrite=overwrite,
    )
    outputs["spatial_overall_feature_model"] = _rank_table(
        spatial,
        ["feature_set", "model_id"],
        out_dir / "spatial_overall_feature_model.csv",
        overwrite=overwrite,
    )

    outputs["integrated_model_comparison"] = _integrated_comparison(
        within,
        cross,
        spatial,
        ["model_id"],
        out_dir / "integrated_model_comparison.csv",
        overwrite=overwrite,
    )
    outputs["integrated_feature_comparison"] = _integrated_comparison(
        within,
        cross,
        spatial,
        ["feature_set"],
        out_dir / "integrated_feature_comparison.csv",
        overwrite=overwrite,
    )
    outputs["integrated_feature_model_comparison"] = _integrated_comparison(
        within,
        cross,
        spatial,
        ["feature_set", "model_id"],
        out_dir / "integrated_feature_model_comparison.csv",
        overwrite=overwrite,
    )
    return outputs


def _triplet_stats(row: pd.Series) -> dict[str, float]:
    """Return integrated AP statistics for one row."""

    values = row[VALIDATION_AP_COLUMNS].to_numpy(dtype="float64", copy=False)
    if not np.isfinite(values).all():
        return {
            "mean_AP": np.nan,
            "median_AP": np.nan,
            "min_AP": np.nan,
            "max_AP": np.nan,
            "range_AP": np.nan,
            "std_AP": np.nan,
            "cv_AP": np.nan,
            "skew_AP": np.nan,
        }

    mean_ap = float(np.mean(values))
    median_ap = float(np.median(values))
    min_ap = float(np.min(values))
    max_ap = float(np.max(values))
    range_ap = float(max_ap - min_ap)
    std_ap = float(np.std(values, ddof=0))
    cv_ap = float(std_ap / mean_ap) if mean_ap != 0 else np.nan
    if std_ap == 0:
        skew_ap = 0.0
    else:
        centered = (values - mean_ap) / std_ap
        skew_ap = float(np.mean(centered ** 3))

    return {
        "mean_AP": mean_ap,
        "median_AP": median_ap,
        "min_AP": min_ap,
        "max_AP": max_ap,
        "range_AP": range_ap,
        "std_AP": std_ap,
        "cv_AP": cv_ap,
        "skew_AP": skew_ap,
    }


def _add_integrated_stats_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add second-stage integrated AP statistics."""

    out = df.copy()
    stats = pd.DataFrame([_triplet_stats(row) for _, row in out.iterrows()])
    out = pd.concat([out.reset_index(drop=True), stats], axis=1)
    out["within_minus_cross"] = out["within_AP"] - out["cross_AP"]
    out["within_minus_spatial"] = out["within_AP"] - out["spatial_AP"]
    out["spatial_minus_cross"] = out["spatial_AP"] - out["cross_AP"]
    return out


def _integrated_stats(
    within: pd.DataFrame,
    cross: pd.DataFrame,
    spatial: pd.DataFrame,
    group_cols: Sequence[str],
    out_path: Path,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    """Write second-stage integrated AP statistics."""

    out = _add_integrated_stats_columns(_merge_validation_ap(within, cross, spatial, group_cols))
    out = (
        out.sort_values(
            ["mean_AP", "min_AP", "range_AP", *group_cols],
            ascending=[False, False, True, *([True] * len(group_cols))],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    out.insert(0, "rank_mean", np.arange(1, out.shape[0] + 1, dtype=int))
    ordered = ["rank_mean", *group_cols, *INTEGRATED_STAT_COLUMNS]
    out = out[ordered]
    _write_csv(out, out_path, overwrite=overwrite)
    return out


def _rank_by_validation(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    validation_name: str,
) -> pd.DataFrame:
    """Aggregate AP_mean and assign rank for one validation strategy."""

    ap_col = f"{validation_name}_AP"
    rank_col = f"{validation_name}_rank"
    out = _ap_by_validation(df, group_cols, ap_col)
    out[rank_col] = out[ap_col].rank(method="min", ascending=False).astype(int)
    return out


def _rank_stability(
    within: pd.DataFrame,
    cross: pd.DataFrame,
    spatial: pd.DataFrame,
    group_cols: Sequence[str],
    out_path: Path,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    """Write validation rank stability table."""

    out = _rank_by_validation(within, group_cols, "within")
    out = out.merge(_rank_by_validation(cross, group_cols, "cross"), on=list(group_cols), how="outer")
    out = out.merge(
        _rank_by_validation(spatial, group_cols, "spatial"),
        on=list(group_cols),
        how="outer",
    )

    ap_values = out[VALIDATION_AP_COLUMNS]
    rank_values = out[["within_rank", "cross_rank", "spatial_rank"]]
    out["mean_rank"] = rank_values.mean(axis=1, skipna=False)
    out["median_rank"] = rank_values.median(axis=1, skipna=False)
    out["best_rank"] = rank_values.min(axis=1, skipna=False)
    out["worst_rank"] = rank_values.max(axis=1, skipna=False)
    out["rank_range"] = out["worst_rank"] - out["best_rank"]
    out["mean_AP"] = ap_values.mean(axis=1, skipna=False)
    out["min_AP"] = ap_values.min(axis=1, skipna=False)
    out["max_AP"] = ap_values.max(axis=1, skipna=False)
    out["range_AP"] = out["max_AP"] - out["min_AP"]

    out = (
        out.sort_values(
            ["mean_rank", "rank_range", "mean_AP", *group_cols],
            ascending=[True, True, False, *([True] * len(group_cols))],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    out.insert(0, "rank_stability_order", np.arange(1, out.shape[0] + 1, dtype=int))
    ordered = [
        "rank_stability_order",
        *group_cols,
        "within_AP",
        "cross_AP",
        "spatial_AP",
        "within_rank",
        "cross_rank",
        "spatial_rank",
        "mean_rank",
        "median_rank",
        "best_rank",
        "worst_rank",
        "rank_range",
        "mean_AP",
        "min_AP",
        "max_AP",
        "range_AP",
    ]
    out = out[ordered]
    _write_csv(out, out_path, overwrite=overwrite)
    return out


def _write_rank_shift(
    feature_model_rank_stability: pd.DataFrame,
    out_path: Path,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    """Write feature-model rank shift table."""

    out = feature_model_rank_stability.copy()
    out["within_to_cross_rank_shift"] = out["cross_rank"] - out["within_rank"]
    out["within_to_spatial_rank_shift"] = out["spatial_rank"] - out["within_rank"]
    out["cross_to_spatial_rank_shift"] = out["spatial_rank"] - out["cross_rank"]
    _write_csv(out, out_path, overwrite=overwrite)
    return out


def _spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Return Spearman rho and p-value."""

    if len(x) < 2:
        return np.nan, np.nan
    try:
        from scipy.stats import spearmanr  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        x_rank = pd.Series(x).rank(method="average").to_numpy(dtype="float64")
        y_rank = pd.Series(y).rank(method="average").to_numpy(dtype="float64")
        if np.std(x_rank) == 0 or np.std(y_rank) == 0:
            return np.nan, np.nan
        return float(np.corrcoef(x_rank, y_rank)[0, 1]), np.nan

    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _site_rank_correlation(
    df_map: Mapping[str, pd.DataFrame],
    group_col: str,
    out_path: Path,
    *,
    comparison_id: str,
    site_ids: Sequence[str],
    overwrite: bool,
) -> pd.DataFrame:
    """Write one configured two-site consistency correlation."""

    if len(site_ids) != 2:
        raise ValueError("Site-rank correlation requires exactly two configured sites.")
    left_site, right_site = site_ids
    correlation_column = f"spearman_rho_{comparison_id}"

    rows: list[dict[str, Any]] = []
    for validation_type, data in df_map.items():
        grouped = (
            data.groupby(["site_id", group_col], dropna=False)["AP_mean"]
            .mean()
            .reset_index()
        )
        pivot = grouped.pivot(index=group_col, columns="site_id", values="AP_mean").dropna()
        required_sites = [left_site, right_site]
        missing_sites = [site_id for site_id in required_sites if site_id not in pivot.columns]
        if missing_sites:
            raise ValueError(
                f"Cannot compute site correlation for {validation_type}/{group_col}; "
                f"missing site column(s): {', '.join(missing_sites)}"
            )
        rho, p_value = _spearman(pivot[left_site], pivot[right_site])
        rows.append(
            {
                "validation_type": validation_type,
                "group": group_col,
                correlation_column: rho,
                "p_value": p_value,
                "n": int(pivot.shape[0]),
            }
        )

    out = pd.DataFrame(rows)
    _write_csv(out, out_path, overwrite=overwrite)
    return out


def _candidate_comparison(
    combo_stats: pd.DataFrame,
    candidates: Sequence[Mapping[str, str]],
    out_path: Path,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    """Write a labeled candidate comparison table."""

    candidate_df = pd.DataFrame(candidates)
    out = candidate_df.merge(combo_stats, on=["feature_set", "model_id"], how="left")
    missing = out[VALIDATION_AP_COLUMNS].isna().any(axis=1)
    if missing.any():
        unavailable = out.loc[missing, ["candidate_label", "feature_set", "model_id"]]
        raise ValueError(
            "Configured candidates are absent from integrated validation results: "
            f"{unavailable.to_dict(orient='records')}"
        )
    columns = ["candidate_label", *CANDIDATE_STAT_COLUMNS]
    out = out[columns]
    _write_csv(out, out_path, overwrite=overwrite)
    return out


def _model_comparison_for_feature(
    combo_stats: pd.DataFrame,
    feature_set: str,
    model_ids: Sequence[str],
    out_path: Path,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    """Write model comparison stats for one feature set."""

    requested = pd.DataFrame(
        [{"feature_set": feature_set, "model_id": model_id} for model_id in model_ids]
    )
    out = requested.merge(combo_stats, on=["feature_set", "model_id"], how="left")
    out = out[CANDIDATE_STAT_COLUMNS]
    _write_csv(out, out_path, overwrite=overwrite)
    return out


def _run_stage2_analysis(
    within: pd.DataFrame,
    cross: pd.DataFrame,
    spatial: pd.DataFrame,
    out_dir: Path,
    analysis_config: Mapping[str, Any],
    *,
    overwrite: bool,
) -> dict[str, pd.DataFrame]:
    """Write stage-2 integrated statistics and candidate comparison outputs."""

    outputs: dict[str, pd.DataFrame] = {}
    outputs["stage2_integrated_model_stats"] = _integrated_stats(
        within,
        cross,
        spatial,
        ["model_id"],
        out_dir / "stage2_integrated_model_stats.csv",
        overwrite=overwrite,
    )
    outputs["stage2_integrated_feature_stats"] = _integrated_stats(
        within,
        cross,
        spatial,
        ["feature_set"],
        out_dir / "stage2_integrated_feature_stats.csv",
        overwrite=overwrite,
    )
    combo_stats = _integrated_stats(
        within,
        cross,
        spatial,
        ["feature_set", "model_id"],
        out_dir / "stage2_integrated_feature_model_stats.csv",
        overwrite=overwrite,
    )
    outputs["stage2_integrated_feature_model_stats"] = combo_stats

    outputs["stage2_model_rank_stability"] = _rank_stability(
        within,
        cross,
        spatial,
        ["model_id"],
        out_dir / "stage2_model_rank_stability.csv",
        overwrite=overwrite,
    )
    outputs["stage2_feature_rank_stability"] = _rank_stability(
        within,
        cross,
        spatial,
        ["feature_set"],
        out_dir / "stage2_feature_rank_stability.csv",
        overwrite=overwrite,
    )
    feature_model_stability = _rank_stability(
        within,
        cross,
        spatial,
        ["feature_set", "model_id"],
        out_dir / "stage2_feature_model_rank_stability.csv",
        overwrite=overwrite,
    )
    outputs["stage2_feature_model_rank_stability"] = feature_model_stability
    outputs["stage2_feature_model_rank_shift"] = _write_rank_shift(
        feature_model_stability,
        out_dir / "stage2_feature_model_rank_shift.csv",
        overwrite=overwrite,
    )

    source_data = {
        "within_site": within,
        "cross_site": cross,
        "spatial_block_cv": spatial,
    }
    comparison = configured_site_rank_comparisons(analysis_config)[0]
    site_map = {
        source_id: source_data[source_id]
        for source_id in comparison["validation_sources"]
    }
    outputs["stage2_site_feature_rank_correlation"] = _site_rank_correlation(
        site_map,
        "feature_set",
        out_dir / "stage2_site_feature_rank_correlation.csv",
        comparison_id=comparison["id"],
        site_ids=comparison["sites"],
        overwrite=overwrite,
    )
    outputs["stage2_site_model_rank_correlation"] = _site_rank_correlation(
        site_map,
        "model_id",
        out_dir / "stage2_site_model_rank_correlation.csv",
        comparison_id=comparison["id"],
        site_ids=comparison["sites"],
        overwrite=overwrite,
    )

    candidates = configured_analysis_candidates(analysis_config)
    outputs["stage2_final_candidate_comparison"] = _candidate_comparison(
        combo_stats,
        candidates,
        out_dir / "stage2_final_candidate_comparison.csv",
        overwrite=overwrite,
    )

    for comparison_spec in configured_focused_model_comparisons(analysis_config):
        output_stem = comparison_spec["output_stem"]
        outputs[output_stem] = _model_comparison_for_feature(
            combo_stats,
            comparison_spec["feature_set"],
            comparison_spec["models"],
            out_dir / f"{output_stem}.csv",
            overwrite=overwrite,
        )
    return outputs


def run_analysis(
    project_id: str,
    output_dir: str | Path | None = None,
    overwrite: bool = True,
) -> list[Path]:
    """Run validation-result comparison analysis."""

    analysis_config = load_analysis_config(project_id=project_id)
    within, cross, spatial, output_root = _read_inputs(project_id, analysis_config)
    out_dir = Path(output_dir).expanduser() if output_dir is not None else output_root / "analysis_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing validation analysis outputs to: {out_dir}", flush=True)
    _run_stage1_analysis(within, cross, spatial, out_dir, overwrite=overwrite)
    _run_stage2_analysis(
        within,
        cross,
        spatial,
        out_dir,
        analysis_config,
        overwrite=overwrite,
    )
    return sorted(out_dir.glob("*.csv"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Analyze configured validation summary results.",
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=f"Project id to process. Default: {DEFAULT_PROJECT_ID}",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: 05_outputs/<project_id>/analysis_compare",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing analysis CSV files. Enabled by default.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if an analysis CSV already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Run validation-result analysis from the command line."""

    args = parse_args()
    output_paths = run_analysis(
        project_id=args.project_id,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    for output_path in output_paths:
        print(f"Analysis output written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()

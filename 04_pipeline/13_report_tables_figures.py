#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate report tables, figures, map exports, and QC diagnostics.

This reporting stage reads outputs from stages 05-12 and writes publication
and supplementary artifacts under ``05_outputs/<project_id>/report_outputs``.
It does not retrain models, recompute validation metrics, modify existing
pipeline outputs, or write outside the report output tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    configured_analysis_candidates,
    configured_final_mapping_policy,
    configured_figure_specs,
    configured_focused_model_comparisons,
    configured_publication_contract,
    configured_publication_display,
    configured_representative_runs,
    configured_source_data_specs,
    configured_table_specs,
    get_site_config,
    load_all_feature_set_configs,
    load_all_model_configs,
    load_analysis_config,
    load_experiment_matrix,
    load_project_config,
    load_publication_config,
    load_sites_config,
    resolve_project_paths,
)
from utils.plotting import (
    add_map_cartography,
    add_north_arrow as shared_add_north_arrow,
    add_scale_bar as shared_add_scale_bar,
    configure_publication_font,
)
from utils.raster_io import write_esri_ascii_pair


RESOLVED_FONT_FAMILY = configure_publication_font()


CANDIDATE_ORDER: list[str] = []
SUPPLEMENTARY_CANDIDATE_COLORS: dict[str, str] = {}
SITE_ORDER: list[str] = []
FEATURE_ORDER: list[str] = []
MODEL_ORDER: list[str] = []
FEATURE_GROUP_ORDER: list[str] = []
MAIN_TITLE_FONTSIZE = 15
FEATURE_GROUP_COLORS: dict[str, str] = {}
PANEL_DPI = 600
PANEL_STYLE = {
    "title": 16,
    "axis": 13,
    "tick": 11,
    "legend": 11,
    "cbar_label": 12,
    "cbar_tick": 11,
    "value": 10,
}
PANEL_CANVAS = {
    "landscape": {
        "figsize": (6.0, 5.0),
        "margins": {"left": 0.20, "right": 0.94, "bottom": 0.19, "top": 0.86},
    },
    "landscape_wide_left": {
        "figsize": (6.0, 5.0),
        "margins": {"left": 0.34, "right": 0.94, "bottom": 0.19, "top": 0.86},
    },
    "map": {
        "figsize": (5.0, 6.0),
        "margins": {"left": 0.24, "right": 0.70, "bottom": 0.17, "top": 0.88},
    },
    "box": {
        "figsize": (6.0, 5.0),
        "margins": {"left": 0.06, "right": 0.94, "bottom": 0.08, "top": 0.92},
    },
}

DISPLAY_LABELS: dict[str, str] = {}
VALIDATION_DISPLAY: dict[str, str] = {}

CORE_ANALYSIS_FILES = [
    "stage2_final_candidate_comparison.csv",
    "stage2_integrated_model_stats.csv",
    "stage2_integrated_feature_stats.csv",
    "stage2_integrated_feature_model_stats.csv",
    "stage2_model_rank_stability.csv",
    "stage2_feature_rank_stability.csv",
    "stage2_feature_model_rank_stability.csv",
    "stage2_feature_model_rank_shift.csv",
    "stage2_site_feature_rank_correlation.csv",
    "stage2_site_model_rank_correlation.csv",
]

REPORT_ANALYSIS_FILES = [
    "within_overall_model.csv",
    "within_overall_feature.csv",
    "within_overall_feature_model.csv",
    "cross_overall_model.csv",
    "cross_overall_feature.csv",
    "cross_overall_feature_model.csv",
    "spatial_overall_model.csv",
    "spatial_overall_feature.csv",
    "spatial_overall_feature_model.csv",
    "within_by_site_model.csv",
    "within_by_site_feature.csv",
    "within_by_site_feature_model.csv",
    "cross_by_direction_model.csv",
    "cross_by_direction_feature.csv",
    "cross_by_direction_feature_model.csv",
    "spatial_by_site_model.csv",
    "spatial_by_site_feature.csv",
    "spatial_by_site_feature_model.csv",
]

MAP_REQUIRED_FILES = [
    "final_model.joblib",
    "feature_importance.csv",
    "train_summary.csv",
    "prediction_probability.tif",
    "prediction_valid_mask.tif",
]
MAP_OPTIONAL_FILES = ["prediction_preview.png"]

REPORT_SUBDIRS = [
    "tables",
    "table_main",
    "table_supplementary",
    "table_source_data",
    "table_source_validation",
    "table_source_figures",
    "table_source_maps",
    "table_source_inventories",
    "table_internal",
    "table_internal_supporting",
    "table_internal_qc",
    "figures",
    "main_composites",
    "main_panels",
    "supplementary_composites",
    "supplementary_panels",
    "maps",
    "diagnostics",
    "asciis",
    "paper_manifest",
]

PROBABILITY_NODATA_ASC = -9999.0
BINARY_NODATA = 255
BINARY_NODATA_ASC = -9999
ASCII_RESAMPLED_RESOLUTION = 0.5
ASCII_CHUNK_ROWS = 256
HISTOGRAM_BINS = 10000
SAVE_VECTOR_OUTPUTS = True
FIGURE_GENERATED_AT = "NA"
FIGURE_GIT_COMMIT = "NA"
FIGURE_SCRIPT_PATH = "04_pipeline/13_report_tables_figures.py"

MAIN_FIGURE_SPECS = [
    ("Figure 1", "Study areas", "fig_01_study_areas"),
    ("Figure 2", "Workflow and validation framework", "fig_02_workflow_validation_framework"),
    ("Figure 3", "MBES-derived environmental features", "fig_03_mbes_feature_definitions"),
    ("Figure 4", "Feature-set comparison", "fig_04_feature_set_comparison"),
    ("Figure 5", "Model comparison", "fig_05_model_comparison"),
    ("Figure 6", "Feature x model integrated performance", "fig_06_feature_model_heatmap"),
    ("Figure 7", "Final candidate comparison", "fig_07_final_candidate_comparison"),
    ("Figure 8", "Feature importance", "fig_08_feature_importance"),
    ("Figure 9", "Predicted habitat maps", "fig_09_predicted_habitat_maps"),
]

SUPPLEMENTARY_FIGURE_SPECS = [
    ("Figure S1", "Within-site validation", "fig_s01_within_site_validation"),
    ("Figure S2", "Cross-site validation", "fig_s02_cross_site_validation"),
    ("Figure S3", "Spatial block CV validation", "fig_s03_spatial_block_cv_validation"),
    ("Figure S4", "Integrated validation comparison", "fig_s04_integrated_comparison"),
    ("Figure S5", "Ranking stability", "fig_s05_ranking_stability"),
    ("Figure S6", "Final candidate analysis", "fig_s06_final_candidate_analysis"),
    ("Figure S7", "Feature importance", "fig_s07_feature_importance"),
    ("Figure S8", "Representative validation performance: Hujeong", "fig_s08_validation_performance_hujeong"),
    ("Figure S9", "Representative validation performance: Bongpyeong", "fig_s09_validation_performance_bongpyeong"),
]

# One canonical publication relationship map feeds captions, metadata, and manifests.
FIGURE_PUBLICATION_DETAILS: dict[str, dict[str, str]] = {
    "fig_01_study_areas": {
        "paired_table_ids": "table_01_study_sites_dataset_summary",
        "paired_source_data_ids": "final_map_inventory",
        "paper_role": "Study-area spatial context",
        "source_lineage": "prepared survey rasters; final_map_inventory",
    },
    "fig_02_workflow_validation_framework": {
        "paired_table_ids": "table_01_study_sites_dataset_summary; table_s02_model_configuration",
        "paired_source_data_ids": "final_model_provenance",
        "paper_role": "Methods workflow and validation design",
        "source_lineage": "project configuration; experiment matrix; final_model_provenance",
    },
    "fig_03_mbes_feature_definitions": {
        "paired_table_ids": "table_s01_feature_set_composition",
        "paired_source_data_ids": "NA",
        "paper_role": "Methods feature concepts",
        "source_lineage": "feature-set configuration; Stage 02 and Stage 03 feature definitions",
    },
    "fig_04_feature_set_comparison": {
        "paired_table_ids": "table_02_key_validation_performance",
        "paired_source_data_ids": "integrated_model_feature_marginals",
        "paper_role": "Results feature-set comparison",
        "source_lineage": "integrated_model_feature_marginals",
    },
    "fig_05_model_comparison": {
        "paired_table_ids": "table_02_key_validation_performance",
        "paired_source_data_ids": "integrated_model_feature_marginals",
        "paper_role": "Results model comparison",
        "source_lineage": "integrated_model_feature_marginals",
    },
    "fig_06_feature_model_heatmap": {
        "paired_table_ids": "table_02_key_validation_performance",
        "paired_source_data_ids": "integrated_feature_model_matrix",
        "paper_role": "Results integrated feature-model comparison",
        "source_lineage": "integrated_feature_model_matrix",
    },
    "fig_07_final_candidate_comparison": {
        "paired_table_ids": "table_03_final_candidate_selection",
        "paired_source_data_ids": "integrated_feature_model_matrix",
        "paper_role": "Results final candidate selection",
        "source_lineage": "stage2_final_candidate_comparison; integrated_feature_model_matrix",
    },
    "fig_08_feature_importance": {
        "paired_table_ids": "table_s08_final_feature_importance",
        "paired_source_data_ids": "final_model_provenance",
        "paper_role": "Results final-model feature importance",
        "source_lineage": "final-map feature_importance artifacts; final_model_provenance",
    },
    "fig_09_predicted_habitat_maps": {
        "paired_table_ids": "table_04_primary_habitat_prediction_summary",
        "paired_source_data_ids": "final_map_inventory; probability_statistics; binary_statistics",
        "paper_role": "Results primary habitat predictions",
        "source_lineage": "Stage 12 primary_balanced probability rasters and valid masks",
    },
    "fig_s01_within_site_validation": {
        "paired_table_ids": "table_s03_full_within_site_results",
        "paired_source_data_ids": "within_site_detailed_summary; within_by_site_feature_model",
        "paper_role": "Supplementary within-site validation",
        "source_lineage": "within-site five-seed summaries",
    },
    "fig_s02_cross_site_validation": {
        "paired_table_ids": "table_s04_full_cross_site_results",
        "paired_source_data_ids": "cross_site_detailed_summary; cross_by_direction_feature_model",
        "paper_role": "Supplementary directional transfer validation",
        "source_lineage": "cross-site five-seed directional summaries",
    },
    "fig_s03_spatial_block_cv_validation": {
        "paired_table_ids": "table_s05_full_spatial_cv_results",
        "paired_source_data_ids": "spatial_block_cv_detailed_summary; spatial_by_site_feature_model",
        "paper_role": "Supplementary spatial generalization validation",
        "source_lineage": "spatial-block five-seed by five-fold summaries",
    },
    "fig_s04_integrated_comparison": {
        "paired_table_ids": "NA",
        "paired_source_data_ids": "integrated_model_feature_marginals; integrated_feature_model_matrix",
        "paper_role": "Supplementary integrated marginal comparison",
        "source_lineage": "integrated_model_feature_marginals; integrated_feature_model_matrix",
    },
    "fig_s05_ranking_stability": {
        "paired_table_ids": "table_s06_rank_stability_shift",
        "paired_source_data_ids": "rank_stability_all_entities; feature_rank_stability",
        "paper_role": "Supplementary rank stability",
        "source_lineage": "rank_stability_all_entities",
    },
    "fig_s06_final_candidate_analysis": {
        "paired_table_ids": "table_s06_rank_stability_shift; table_s07_final_candidate_map_statistics",
        "paired_source_data_ids": "rank_stability_all_entities; probability_statistics; binary_statistics",
        "paper_role": "Supplementary final-candidate robustness and map sensitivity",
        "source_lineage": "rank_stability_all_entities; Stage 12 final-map statistics",
    },
    "fig_s07_feature_importance": {
        "paired_table_ids": "table_s08_final_feature_importance",
        "paired_source_data_ids": "final_model_provenance",
        "paper_role": "Supplementary final-model feature importance",
        "source_lineage": "final-map feature_importance artifacts; final_model_provenance",
    },
    "fig_s08_validation_performance_hujeong": {
        "paired_table_ids": "table_s09_representative_validation_hujeong",
        "paired_source_data_ids": "NA",
        "paper_role": "Supplementary representative validation for Hujeong",
        "source_lineage": "RF full_multi seed-42 stored predictions for Hujeong",
    },
    "fig_s09_validation_performance_bongpyeong": {
        "paired_table_ids": "table_s10_representative_validation_bongpyeong",
        "paired_source_data_ids": "NA",
        "paper_role": "Supplementary representative validation for Bongpyeong",
        "source_lineage": "RF full_multi seed-42 stored predictions for Bongpyeong",
    },
}

# Official paper numbers and canonical filenames share the same numeric prefix.
MAIN_TABLE_SPECS = [
    ("Table 1", "Study sites and dataset summary", "table_01_study_sites_dataset_summary", "_publication_site_summary"),
    ("Table 2", "Top-performing model-feature combinations by validation scope", "table_02_key_validation_performance", "_key_validation_performance"),
    ("Table 3", "Final candidate comparison and selection", "table_03_final_candidate_selection", "_final_candidate_selection_summary"),
    ("Table 4", "Final habitat prediction summary", "table_04_primary_habitat_prediction_summary", "_primary_habitat_summary"),
]

SUPPLEMENTARY_TABLE_SPECS = [
    ("Table S1", "Feature-set definitions and composition", "table_s01_feature_set_composition", "_feature_set_definitions"),
    ("Table S2", "Model configuration and hyperparameters", "table_s02_model_configuration", "_model_config_summary"),
    ("Table S3", "Full within-site validation results", "table_s03_full_within_site_results", "_validation_publication_summary"),
    ("Table S4", "Full cross-site transfer results", "table_s04_full_cross_site_results", "_validation_publication_summary"),
    ("Table S5", "Full spatial block CV results", "table_s05_full_spatial_cv_results", "_validation_publication_summary"),
    ("Table S6", "Rank stability and rank shift", "table_s06_rank_stability_shift", "_rank_stability_shift_summary"),
    ("Table S7", "Final candidate map statistics", "table_s07_final_candidate_map_statistics", "_final_candidate_map_statistics"),
    ("Table S8", "Final-model feature importance", "table_s08_final_feature_importance", "_portable_feature_importance"),
    ("Table S9", "Representative validation performance: Hujeong", "table_s09_representative_validation_hujeong", "_representative_validation_table"),
    ("Table S10", "Representative validation performance: Bongpyeong", "table_s10_representative_validation_bongpyeong", "_representative_validation_table"),
]

SOURCE_FIGURE_STEMS = [
    "within_by_site_model", "within_by_site_feature", "within_by_site_feature_model",
    "cross_by_direction_model", "cross_by_direction_feature", "cross_by_direction_feature_model",
    "spatial_by_site_model", "spatial_by_site_feature", "spatial_by_site_feature_model",
    "within_overall_model", "within_overall_feature", "within_overall_feature_model",
    "cross_overall_model", "cross_overall_feature", "cross_overall_feature_model",
    "spatial_overall_model", "spatial_overall_feature", "spatial_overall_feature_model",
    "model_rank_stability", "feature_rank_stability", "site_rank_correlation",
]

MAIN_PANEL_PARENT_DIRS = {
    "fig01": "fig_01_study_areas",
    "fig02": "fig_02_workflow_validation_framework",
    "fig03": "fig_03_mbes_feature_definitions",
    "fig04": "fig_04_feature_set_comparison",
    "fig05": "fig_05_model_comparison",
    "fig06": "fig_06_feature_model_heatmap",
    "fig07": "fig_07_final_candidate_comparison",
    "fig08": "fig_08_feature_importance",
    "fig09": "fig_09_predicted_habitat_maps",
}

PUBLICATION_CONFIG: Mapping[str, Any] = {}
PUBLICATION_FIGURES: dict[str, dict[str, Any]] = {}
PUBLICATION_TABLES: dict[str, dict[str, Any]] = {}
PUBLICATION_SOURCES: dict[str, dict[str, Any]] = {}
PUBLICATION_REPRESENTATIVE_RUNS: dict[str, dict[str, Any]] = {}
PRIMARY_PUBLICATION_CANDIDATE = ""
OFFICIAL_CAPTIONS: dict[str, str] = {}
TABLE_CAPTIONS: dict[str, str] = {}
CANDIDATE_TABLE_ROLES: dict[str, tuple[str, str]] = {}
CANDIDATE_COMPACT_LABELS: dict[str, str] = {}
MODEL_COMPACT_LABELS: dict[str, str] = {}
PUBLICATION_SITE_STYLES: dict[str, tuple[str, float]] = {}


def _configure_publication_policy(config: Mapping[str, Any]) -> None:
    """Install one validated publication specification for the current run."""

    global PUBLICATION_CONFIG, PUBLICATION_FIGURES, PUBLICATION_TABLES
    global PUBLICATION_SOURCES, PUBLICATION_REPRESENTATIVE_RUNS
    global PRIMARY_PUBLICATION_CANDIDATE, OFFICIAL_CAPTIONS, TABLE_CAPTIONS
    global CANDIDATE_TABLE_ROLES, CANDIDATE_ORDER, SUPPLEMENTARY_CANDIDATE_COLORS
    global CANDIDATE_COMPACT_LABELS, MODEL_COMPACT_LABELS, PUBLICATION_SITE_STYLES
    global SITE_ORDER, FEATURE_ORDER, MODEL_ORDER, FEATURE_GROUP_ORDER
    global FEATURE_GROUP_COLORS, DISPLAY_LABELS, VALIDATION_DISPLAY
    global MAIN_FIGURE_SPECS, SUPPLEMENTARY_FIGURE_SPECS, FIGURE_PUBLICATION_DETAILS
    global MAIN_TABLE_SPECS, SUPPLEMENTARY_TABLE_SPECS, SOURCE_FIGURE_STEMS
    global MAIN_PANEL_PARENT_DIRS

    publication = config["publication"]
    figures = list(configured_figure_specs(config))
    tables = list(configured_table_specs(config))
    sources = list(configured_source_data_specs(config))
    display_sections = {
        name: list(configured_publication_display(config, name))
        for name in ("sites", "candidates", "feature_sets", "models", "validations", "feature_groups")
    }
    labels = dict(publication["display"].get("labels", {}))
    for items in display_sections.values():
        labels.update({item["id"]: item["label"] for item in items})

    PUBLICATION_CONFIG = config
    PUBLICATION_FIGURES = {item["id"]: item for item in figures}
    PUBLICATION_TABLES = {item["id"]: item for item in tables}
    PUBLICATION_SOURCES = {item["id"]: item for item in sources}
    PUBLICATION_REPRESENTATIVE_RUNS = {
        item["target_site"]: item for item in configured_representative_runs(config)
    }
    PRIMARY_PUBLICATION_CANDIDATE = str(publication["primary_candidate_id"])
    OFFICIAL_CAPTIONS = {item["stem"]: item["caption"] + "\n" for item in figures}
    TABLE_CAPTIONS = {item["stem"]: item["caption"] + "\n" for item in tables}

    CANDIDATE_ORDER = [item["id"] for item in display_sections["candidates"]]
    SUPPLEMENTARY_CANDIDATE_COLORS = {
        item["id"]: item["color"] for item in display_sections["candidates"]
    }
    CANDIDATE_TABLE_ROLES = {
        item["id"]: (item["final_map_role"], item["selection_basis"])
        for item in display_sections["candidates"]
    }
    CANDIDATE_COMPACT_LABELS = {
        item["id"]: str(item.get("compact_label", item["label"]))
        for item in display_sections["candidates"]
    }
    MODEL_COMPACT_LABELS = {
        item["id"]: str(item.get("compact_label", item["label"]))
        for item in display_sections["models"]
    }
    PUBLICATION_SITE_STYLES = {
        item["id"]: (str(item.get("hatch", "")), float(item.get("alpha", 1.0)))
        for item in display_sections["sites"]
    }
    SITE_ORDER = [item["id"] for item in display_sections["sites"]]
    FEATURE_ORDER = [item["id"] for item in display_sections["feature_sets"]]
    MODEL_ORDER = [item["id"] for item in display_sections["models"]]
    FEATURE_GROUP_ORDER = [item["id"] for item in display_sections["feature_groups"]]
    FEATURE_GROUP_COLORS = {
        item["id"]: item["color"] for item in display_sections["feature_groups"]
    }
    DISPLAY_LABELS = labels
    VALIDATION_DISPLAY = {
        item["id"]: item["label"] for item in display_sections["validations"]
    }
    MAIN_FIGURE_SPECS = [
        (item["number"], item["title"], item["stem"])
        for item in figures if item["role"] == "main"
    ]
    SUPPLEMENTARY_FIGURE_SPECS = [
        (item["number"], item["title"], item["stem"])
        for item in figures if item["role"] == "supplementary"
    ]
    FIGURE_PUBLICATION_DETAILS = {
        item["stem"]: {
            "paired_table_ids": "; ".join(item.get("paired_table_ids", [])) or "NA",
            "paired_source_data_ids": "; ".join(item.get("paired_source_data_ids", [])) or "NA",
            "paper_role": item["manuscript_role"],
            "source_lineage": item["source_lineage"],
        }
        for item in figures
    }
    MAIN_TABLE_SPECS = [
        (item["number"], item["title"], item["stem"], "_" + item["builder"])
        for item in tables if item["role"] == "main"
    ]
    SUPPLEMENTARY_TABLE_SPECS = [
        (item["number"], item["title"], item["stem"], "_" + item["builder"])
        for item in tables if item["role"] == "supplementary"
    ]
    SOURCE_FIGURE_STEMS = [
        item["stem"] for item in sources if item["category"] == "figure"
    ]
    MAIN_PANEL_PARENT_DIRS = {
        panel["stem"].split("_panel_", 1)[0]: figure["stem"]
        for figure in figures if figure["role"] == "main"
        for panel in figure.get("panels", [])
    }
    configured_publication_contract(config)


def _publication_figure(figure_id: str) -> Mapping[str, Any]:
    try:
        return PUBLICATION_FIGURES[figure_id]
    except KeyError as exc:
        raise ValueError(f"Publication figure {figure_id!r} is not configured.") from exc


def _publication_panels(figure_id: str) -> list[Mapping[str, Any]]:
    return list(_publication_figure(figure_id).get("panels", []))


def _representative_run(target_site: str) -> Mapping[str, Any]:
    try:
        return PUBLICATION_REPRESENTATIVE_RUNS[target_site]
    except KeyError as exc:
        raise ValueError(f"No publication representative run configured for {target_site!r}.") from exc


@dataclass(frozen=True)
class ReportOptions:
    """Runtime options for report generation."""

    project_id: str
    site_filter: str | None
    candidate_label_filter: str | None
    threshold: float
    output_dir: Path | None
    overwrite: bool
    dry_run: bool
    dpi: int
    no_figures: bool
    no_tables: bool
    no_maps: bool
    no_asc: bool
    no_diagnostics: bool
    no_grids: bool
    max_preview_size: int
    allow_full_xyz: bool
    xyz_stride: int | None
    save_vector: bool
    map_cmap: str
    difference_cmap: str
    hist_log_y: bool
    skip_native_asc: bool
    skip_0p5m_asc: bool
    generate_paper_manifest: bool
    generate_captions: bool
    generate_source_data: bool
    generate_metadata: bool
    main_panels: bool
    main_panels_only: bool
    panel_label: bool
    figures_only: bool
    tables_only: bool


@dataclass(frozen=True)
class ReportDirs:
    """Output directories for stage 13."""

    root: Path
    tables: Path
    table_main: Path
    table_supplementary: Path
    table_source_data: Path
    table_source_validation: Path
    table_source_figures: Path
    table_source_maps: Path
    table_source_inventories: Path
    table_internal: Path
    table_internal_supporting: Path
    table_internal_qc: Path
    figures: Path
    main_composites: Path
    main_panels: Path
    supplementary_composites: Path
    supplementary_panels: Path
    maps: Path
    diagnostics: Path
    asciis: Path
    grids: Path
    paper_manifest: Path


@dataclass(frozen=True)
class PanelMetadata:
    """Metadata for one Illustrator-ready main panel."""

    figure_number: str
    panel_letter: str
    panel_title: str
    relative_png: str
    recommended_usage: str


@dataclass(frozen=True)
class PanelCanvas:
    """Fixed panel canvas geometry."""

    kind: str
    figsize: tuple[float, float]
    margins: Mapping[str, float]


def _require_rasterio() -> Any:
    """Import rasterio with a clear dependency message."""

    try:
        import rasterio  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "rasterio is required for report map exports. Run this stage in "
            "the mbes_seaweed environment."
        ) from exc
    return rasterio


def _require_matplotlib_pyplot() -> Any:
    """Import matplotlib.pyplot with a clear dependency message."""

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "matplotlib is required for report figures. Use --no-figures to "
            "skip figure generation."
        ) from exc
    return plt


def _save_figure_outputs(
    fig: Any,
    path: Path,
    *,
    dpi: int,
    save_vector: bool | None = None,
) -> None:
    """Save PNG and optional vector PDF for one figure."""

    use_vector = SAVE_VECTOR_OUTPUTS if save_vector is None else bool(save_vector)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if use_vector:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def _save_main_figure_outputs(
    fig: Any,
    path: Path,
    *,
    dpi: int,
    save_vector: bool | None = None,
) -> None:
    """Save a main figure on the fixed canvas so centered titles stay centered."""

    use_vector = SAVE_VECTOR_OUTPUTS if save_vector is None else bool(save_vector)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    if use_vector:
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".svg"))


def _add_centered_main_title(
    fig: Any,
    title: str,
    *,
    fontsize: int = MAIN_TITLE_FONTSIZE,
    y: float = 0.975,
) -> None:
    """Place a main figure title at the saved canvas center."""

    fig.text(
        0.5,
        y,
        title,
        ha="center",
        va="top",
        transform=fig.transFigure,
        fontsize=fontsize,
        fontweight="bold",
    )


def _remove_stale_figure_outputs(path: Path) -> None:
    """Remove stale derived figure files when a main figure must be skipped."""

    for candidate in [path, path.with_suffix(".pdf")]:
        if candidate.exists():
            candidate.unlink()


def _short_label(label: Any, max_len: int = 18) -> str:
    """Return a compact, wrapped label for plot axes."""

    text = str(DISPLAY_LABELS.get(str(label), label))
    if len(text) <= max_len:
        return text
    for sep in ["_", "-", " "]:
        if sep in text:
            parts = text.split(sep)
            midpoint = max(1, len(parts) // 2)
            return sep.join(parts[:midpoint]) + "\n" + sep.join(parts[midpoint:])
    return text[: max_len - 1] + "."


def _display_label(value: Any) -> str:
    """Return publication-friendly label for an internal id."""

    text = str(value)
    if text in DISPLAY_LABELS:
        return DISPLAY_LABELS[text]
    if text.startswith("train_") and "_test_" in text:
        body = text.removeprefix("train_")
        train_site, test_site = body.split("_test_", 1)
        return f"Train {_display_label(train_site)} -> Test {_display_label(test_site)}"
    return text.replace("_", " ")


def _display_column_name(value: str) -> str:
    """Return publication-friendly metric/column label."""

    return DISPLAY_LABELS.get(value, value.replace("_", " "))


def _feature_group(feature: Any) -> str:
    """Return the report-level feature class for feature-importance plots."""

    text = str(feature)
    if text.startswith("dpth_"):
        return "Bathymetric features"
    if text in {"a200_norm", "a300_norm", "a400_norm"}:
        return "Acoustic intensity"
    if text in {"a300_low", "a300_high", "a300_log1p"} or (
        text.startswith("a") and text.endswith(("_low", "_high", "_log1p"))
    ):
        return "Texture features"
    if text.startswith("NDI_"):
        return "Frequency-difference indices"
    return "Other"


def _publication_model_label(value: Any) -> str:
    """Return the shared publication label for a model id."""

    text = str(value)
    if text not in MODEL_ORDER:
        raise ValueError(f"Unknown model id for publication label: {text}")
    return _display_label(text)


def _publication_feature_set_label(value: Any) -> str:
    """Return the shared publication label for a feature-set id."""

    text = str(value)
    if text not in FEATURE_ORDER:
        raise ValueError(f"Unknown feature-set id for publication label: {text}")
    return _display_label(text)


def _publication_candidate_label(value: Any) -> str:
    """Return the shared publication label for a final-candidate id."""

    text = str(value)
    if text not in CANDIDATE_ORDER:
        raise ValueError(f"Unknown candidate id for publication label: {text}")
    return _display_label(text)


def _publication_entity_label(entity_type: str, model_id: Any, feature_set: Any) -> str:
    """Return one publication-ready model, feature-set, or combination label."""

    if entity_type == "model":
        return _publication_model_label(model_id)
    if entity_type == "feature_set":
        return _publication_feature_set_label(feature_set)
    if entity_type == "model_feature":
        return (
            f"{_publication_model_label(model_id)} + "
            f"{_publication_feature_set_label(feature_set)}"
        )
    raise ValueError(f"Unknown rank entity type: {entity_type}")


def _nice_ylim(values: Sequence[float], lower_floor: float | None = None) -> tuple[float, float]:
    """Return a y-range with 15-20 percent headroom."""

    arr = np.asarray([value for value in values if np.isfinite(value)], dtype="float64")
    if arr.size == 0:
        return (0.0, 1.0)
    ymin = float(arr.min())
    ymax = float(arr.max())
    span = max(ymax - ymin, 0.05)
    low = ymin - span * 0.10
    high = ymax + span * 0.20
    if lower_floor is not None:
        low = max(lower_floor, low)
    return (low, high)


def _as_float(value: Any, default: float = np.nan) -> float:
    """Return ``value`` as float, preserving missing values as ``default``."""

    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _ordered(values: Iterable[str], order: Sequence[str]) -> list[str]:
    """Return values in preferred order, keeping unknown values at the end."""

    items = [str(value) for value in values]
    known = [value for value in order if value in items]
    rest = sorted(value for value in items if value not in order)
    return known + rest


def _require_existing_file(path: Path, label: str) -> Path:
    """Validate that a path exists and is a file."""

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    """Read a required non-empty CSV."""

    _require_existing_file(path, label)
    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} is empty: {path}") from exc
    if data.empty:
        raise ValueError(f"{label} contains zero rows: {path}")
    return data


def _write_csv(
    df: pd.DataFrame,
    path: Path,
    overwrite: bool,
    *,
    float_format: str | None = None,
) -> None:
    """Write a CSV file with overwrite protection."""

    if path.exists() and not overwrite:
        print(f"Skipped existing CSV due to --no-overwrite: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    content = df.to_csv(index=False, float_format=float_format)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    """Copy a file, respecting overwrite protection."""

    _require_existing_file(src, "Input file")
    if dst.exists() and not overwrite:
        print(f"Skipped existing file due to --no-overwrite: {dst}", flush=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_text(path: Path, text: str, overwrite: bool) -> None:
    """Write text with overwrite protection."""

    if path.exists() and not overwrite:
        print(f"Skipped existing text due to --no-overwrite: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _make_report_dirs(report_root: Path, options: ReportOptions) -> ReportDirs:
    """Return report output directories, creating them unless dry-running."""

    dirs = ReportDirs(
        root=report_root,
        tables=report_root / "tables",
        table_main=report_root / "tables" / "main",
        table_supplementary=report_root / "tables" / "supplementary",
        table_source_data=report_root / "tables" / "source_data",
        table_source_validation=report_root / "tables" / "source_data" / "validation",
        table_source_figures=report_root / "tables" / "source_data" / "figures",
        table_source_maps=report_root / "tables" / "source_data" / "maps",
        table_source_inventories=report_root / "tables" / "source_data" / "inventories",
        table_internal=report_root / "tables" / "internal",
        table_internal_supporting=report_root / "tables" / "internal" / "supporting",
        table_internal_qc=report_root / "tables" / "internal" / "qc",
        figures=report_root / "figures",
        main_composites=report_root / "figures" / "main" / "composites",
        main_panels=report_root / "figures" / "main" / "panels",
        supplementary_composites=report_root / "figures" / "supplementary" / "composites",
        supplementary_panels=report_root / "figures" / "supplementary" / "panels",
        maps=report_root / "maps",
        diagnostics=report_root / "diagnostics",
        asciis=report_root / "asciis",
        grids=report_root / "grids",
        paper_manifest=report_root / "paper_manifest",
    )
    if not options.dry_run:
        directory_names = (
            [
                "tables", "table_main", "table_supplementary", "table_source_data",
                "table_source_validation", "table_source_figures", "table_source_maps",
                "table_source_inventories", "table_internal", "table_internal_supporting",
                "table_internal_qc", "paper_manifest",
            ]
            if options.tables_only
            else REPORT_SUBDIRS
        )
        for name in directory_names:
            path = getattr(dirs, name)
            path.mkdir(parents=True, exist_ok=True)
        if (options.allow_full_xyz or options.xyz_stride is not None) and not options.no_grids:
            dirs.grids.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_typography_report(dirs: ReportDirs, options: ReportOptions) -> None:
    """Record the requested font policy and Matplotlib's actual resolution."""

    text = (
        "requested_font_order: Arial; Helvetica; DejaVu Sans\n"
        f"resolved_font_family: {RESOLVED_FONT_FAMILY}\n"
        "svg_fonttype: none\n"
        "pdf_fonttype: 42\n"
    )
    _write_text(dirs.paper_manifest / "typography.txt", text, options.overwrite)


def _current_git_commit() -> str:
    """Return the current commit without modifying the repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "NA"
    return result.stdout.strip() or "NA"


def _source_paths_from_frame(data: pd.DataFrame) -> str:
    """Return unique path-like values represented by a source-data frame."""

    values: list[str] = []
    for column in data.columns:
        if "path" not in str(column).lower():
            continue
        for value in data[column].dropna().astype(str):
            portable = _project_relative_path(value)
            if portable and portable not in values:
                values.append(portable)
    return "; ".join(values) if values else "NA"


def _figure_stem_from_path(path: Path) -> str:
    """Return the canonical parent Figure stem for a composite or panel path."""

    if path.stem in FIGURE_PUBLICATION_DETAILS:
        return path.stem
    if path.parent.name in FIGURE_PUBLICATION_DETAILS:
        return path.parent.name
    return path.stem


def _figure_number_from_stem(stem: str) -> str:
    """Return the publication Figure number for a canonical stem."""

    for number, _, candidate in MAIN_FIGURE_SPECS + SUPPLEMENTARY_FIGURE_SPECS:
        if candidate == stem:
            return number
    return stem


def _figure_publication_details(path: Path, panel_letter: str = "composite") -> dict[str, str]:
    """Return canonical pairing and lineage metadata for one Figure unit."""

    stem = _figure_stem_from_path(path)
    details = dict(FIGURE_PUBLICATION_DETAILS.get(stem, {}))
    if stem == "fig_s06_final_candidate_analysis":
        if panel_letter == "a":
            details["paired_table_ids"] = "table_s06_rank_stability_shift"
            details["paired_source_data_ids"] = "rank_stability_all_entities"
        elif panel_letter in {"b", "c"}:
            details["paired_table_ids"] = "table_s07_final_candidate_map_statistics"
            details["paired_source_data_ids"] = (
                "probability_statistics" if panel_letter == "b" else "binary_statistics"
            )
    details["parent_figure_stem"] = stem
    return details


def _write_figure_metadata(
    path: Path,
    *,
    parent_figure: str,
    panel_letter: str,
    panel_title: str,
    official_or_auxiliary: str,
    plot_function: str,
    source_data: pd.DataFrame,
    options: ReportOptions,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write the shared composite/panel metadata contract."""

    publication = _figure_publication_details(path, panel_letter)
    canonical_parent = _figure_number_from_stem(publication.get("parent_figure_stem", ""))
    values: dict[str, Any] = {
        "parent_figure": canonical_parent if canonical_parent else parent_figure,
        "parent_figure_stem": publication.get("parent_figure_stem", "NA"),
        "panel_letter": panel_letter,
        "panel_title": panel_title,
        "official_or_auxiliary": official_or_auxiliary,
        "paired_table_ids": publication.get("paired_table_ids", "NA"),
        "paired_source_data_ids": publication.get("paired_source_data_ids", "NA"),
        "paper_role": publication.get("paper_role", "NA"),
        "source_lineage": publication.get("source_lineage", "NA"),
        "script_path": FIGURE_SCRIPT_PATH,
        "plot_function": plot_function,
        "source_data_paths": _source_paths_from_frame(source_data),
        "model": "NA",
        "feature_set": "NA",
        "seed": "NA",
        "site": "NA",
        "target_site": "NA",
        "validation_strategy": "NA",
        "threshold": options.threshold,
        "generated_at": FIGURE_GENERATED_AT,
        "git_commit": FIGURE_GIT_COMMIT,
        "dpi": options.dpi,
    }
    if extra:
        values.update(extra)
    text = "".join(f"{key}: {value}\n" for key, value in values.items())
    _write_sidecar_text(
        path.with_name(path.stem + "_metadata.txt"),
        text,
        options.overwrite,
    )


def _analysis_path(output_root: Path, name: str) -> Path:
    """Return one analysis_compare CSV path."""

    return output_root / "analysis_compare" / name


def _expected_report_path(base: Path, relative: str) -> Path:
    """Return an expected report output path."""

    return base / relative


def _focused_analysis_files(analysis_config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return Stage-11 focused-comparison files declared by analysis policy."""

    return tuple(
        f"{item['output_stem']}.csv"
        for item in configured_focused_model_comparisons(analysis_config)
    )


def _load_analysis_inputs(
    output_root: Path,
    analysis_config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """Load required and frequently used analysis CSV inputs."""

    analysis: dict[str, pd.DataFrame] = {}
    names = set(CORE_ANALYSIS_FILES)
    names.update(_focused_analysis_files(analysis_config))
    names.update(REPORT_ANALYSIS_FILES)

    for name in sorted(names):
        path = _analysis_path(output_root, name)
        if path.exists():
            analysis[name] = _read_csv(path, name)

    required = [*CORE_ANALYSIS_FILES, *_focused_analysis_files(analysis_config)]
    missing_core = [name for name in required if name not in analysis]
    if missing_core:
        raise FileNotFoundError(
            "Missing core analysis_compare CSV(s): " + ", ".join(missing_core)
        )
    return analysis


def _selected_sites(
    experiment_matrix: Mapping[str, Any],
    site_filter: str | None,
) -> list[str]:
    """Return selected site ids in requested report order."""

    sites = experiment_matrix.get("sites")
    if not isinstance(sites, list) or not all(isinstance(site, str) for site in sites):
        raise ValueError("experiment_matrix.sites must be a list of strings.")
    ordered_sites = _ordered(sites, SITE_ORDER)
    if site_filter is None:
        return ordered_sites
    if site_filter not in ordered_sites:
        raise ValueError(f"--site={site_filter!r} is not listed in experiment_matrix.sites.")
    return [site_filter]


def _selected_mapping_sites(
    analysis_config: Mapping[str, Any],
    experiment_matrix: Mapping[str, Any],
    site_filter: str | None,
) -> list[str]:
    """Return final-map sites from the Stage 11/12 analysis contract."""

    mapping_policy = configured_final_mapping_policy(analysis_config, experiment_matrix)
    sites = list(mapping_policy["sites"])
    if site_filter is None:
        return sites
    if site_filter not in sites:
        raise ValueError(
            f"--site={site_filter!r} is not selected by analysis.final_mapping.sites."
        )
    return [site_filter]


def _selected_candidates(
    candidates: pd.DataFrame,
    candidate_label_filter: str | None,
) -> pd.DataFrame:
    """Return selected final candidate rows in publication order."""

    required = ["candidate_label", "feature_set", "model_id"]
    missing = [column for column in required if column not in candidates.columns]
    if missing:
        raise ValueError(
            "stage2_final_candidate_comparison.csv missing required column(s): "
            + ", ".join(missing)
        )
    out = candidates.copy()
    if candidate_label_filter is not None:
        out = out[out["candidate_label"].astype(str) == candidate_label_filter].copy()
        if out.empty:
            raise ValueError(
                f"--candidate-label={candidate_label_filter!r} is not listed in "
                "stage2_final_candidate_comparison.csv."
            )
    out["candidate_label"] = out["candidate_label"].astype(str)
    out["_candidate_order"] = out["candidate_label"].map(
        {label: index for index, label in enumerate(CANDIDATE_ORDER)}
    )
    out["_candidate_order"] = out["_candidate_order"].fillna(9999)
    out = out.sort_values(["_candidate_order", "candidate_label"]).drop(columns=["_candidate_order"])
    return out.reset_index(drop=True)


def _site_name(site_id: str, sites_config: Mapping[str, Any]) -> str:
    """Return a human-readable site name."""

    site_config = get_site_config(sites_config, site_id)
    return str(site_config.get("site_name", site_id))


def _load_final_map_records(
    *,
    output_root: Path,
    sites_config: Mapping[str, Any],
    sites: Sequence[str],
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read final map train summaries and feature importance CSVs."""

    summary_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    expected_records: list[dict[str, Any]] = []

    for site_id in sites:
        site_name = _site_name(site_id, sites_config)
        for _, candidate in candidates.iterrows():
            candidate_label = str(candidate["candidate_label"])
            final_dir = output_root / "final_maps" / site_id / candidate_label
            for file_name in MAP_REQUIRED_FILES:
                expected_records.append(
                    {
                        "site_id": site_id,
                        "site_name": site_name,
                        "candidate_label": candidate_label,
                        "path": str(final_dir / file_name),
                        "exists": (final_dir / file_name).exists(),
                        "required": True,
                    }
                )
            for file_name in MAP_OPTIONAL_FILES:
                expected_records.append(
                    {
                        "site_id": site_id,
                        "site_name": site_name,
                        "candidate_label": candidate_label,
                        "path": str(final_dir / file_name),
                        "exists": (final_dir / file_name).exists(),
                        "required": False,
                    }
                )
            train_summary_path = final_dir / "train_summary.csv"
            feature_importance_path = final_dir / "feature_importance.csv"
            probability_path = final_dir / "prediction_probability.tif"
            mask_path = final_dir / "prediction_valid_mask.tif"
            _require_existing_file(train_summary_path, "Final map train_summary.csv")
            _require_existing_file(feature_importance_path, "Final map feature_importance.csv")
            _require_existing_file(probability_path, "Final probability raster")
            _require_existing_file(mask_path, "Final valid-mask raster")

            summary = _read_csv(train_summary_path, "Final map train_summary.csv")
            summary["candidate_label"] = candidate_label
            summary["site_id"] = site_id
            summary["site_name"] = site_name
            summary["train_summary_path"] = str(train_summary_path)
            summary_frames.append(summary)

            importance = _read_csv(feature_importance_path, "Final map feature_importance.csv")
            importance["candidate_label"] = candidate_label
            importance["site_id"] = site_id
            importance["site_name"] = site_name
            importance["feature_importance_path"] = str(feature_importance_path)
            importance_frames.append(importance)

    train_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    feature_importance = (
        pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    )
    expected = pd.DataFrame(expected_records)
    missing = expected[expected["required"] & ~expected["exists"]]
    if not missing.empty:
        raise FileNotFoundError(
            "Missing final map input(s): " + "; ".join(missing["path"].astype(str).tolist())
        )
    return train_summary, feature_importance


def _map_record_iter(train_summary: pd.DataFrame) -> Iterable[dict[str, Any]]:
    """Yield one normalized final map record per train summary row."""

    for _, row in train_summary.iterrows():
        yield {
            "site_id": str(row["site_id"]),
            "site_name": str(row.get("site_name", row["site_id"])),
            "candidate_label": str(row["candidate_label"]),
            "feature_set": str(row.get("feature_set", "")),
            "model_id": str(row.get("model_id", "")),
            "probability_src": Path(str(row["prediction_probability_path"])),
            "valid_mask_src": Path(str(row["prediction_valid_mask_path"])),
            "train_summary_path": Path(str(row["train_summary_path"])),
        }


def _raster_profile_record(path: Path) -> dict[str, Any]:
    """Return core raster metadata."""

    rasterio = _require_rasterio()
    with rasterio.open(path) as src:
        return {
            "width": int(src.width),
            "height": int(src.height),
            "count": int(src.count),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "crs": str(src.crs),
            "transform": src.transform,
            "pixel_size_x": float(src.res[0]),
            "pixel_size_y": float(src.res[1]),
            "bounds": src.bounds,
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }


def _binary_output_path(dirs: ReportDirs, site_id: str, candidate_label: str) -> Path:
    """Return the report binary map path."""

    return dirs.maps / site_id / candidate_label / "binary_05.tif"


def _probability_output_path(dirs: ReportDirs, site_id: str, candidate_label: str) -> Path:
    """Return the report probability map path."""

    return dirs.maps / site_id / candidate_label / "probability.tif"


def _png_output_path(
    dirs: ReportDirs,
    site_id: str,
    candidate_label: str,
    kind: str,
) -> Path:
    """Return one report map PNG path."""

    return dirs.maps / site_id / candidate_label / f"{kind}.png"


def _write_binary_map(
    probability_path: Path,
    valid_mask_path: Path,
    binary_path: Path,
    threshold: float,
    overwrite: bool,
) -> dict[str, Any]:
    """Write binary habitat map with uint8 nodata=255 using raster windows."""

    rasterio = _require_rasterio()
    if binary_path.exists() and not overwrite:
        print(f"Skipped existing binary map due to --no-overwrite: {binary_path}", flush=True)
        return _binary_counts_from_probability(probability_path, valid_mask_path, threshold)

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    positive = 0
    negative = 0
    invalid = 0

    with rasterio.open(probability_path) as prob_src, rasterio.open(valid_mask_path) as mask_src:
        if (
            prob_src.width != mask_src.width
            or prob_src.height != mask_src.height
            or prob_src.transform != mask_src.transform
            or str(prob_src.crs) != str(mask_src.crs)
        ):
            raise ValueError(
                "Probability raster and valid mask grid mismatch: "
                f"{probability_path}, {valid_mask_path}"
            )

        profile = prob_src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=BINARY_NODATA,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(binary_path, "w", **profile) as dst:
            for _, window in prob_src.block_windows(1):
                probability = prob_src.read(1, window=window, masked=False)
                valid_mask = mask_src.read(1, window=window, masked=False) == 1
                valid = valid_mask & np.isfinite(probability)
                out = np.full(probability.shape, BINARY_NODATA, dtype="uint8")
                positives = valid & (probability >= threshold)
                negatives = valid & (probability < threshold)
                out[positives] = 1
                out[negatives] = 0
                positive += int(np.count_nonzero(positives))
                negative += int(np.count_nonzero(negatives))
                invalid += int(out.size - np.count_nonzero(valid))
                dst.write(out, 1, window=window)

    return {
        "positive_pixel_count": positive,
        "negative_pixel_count": negative,
        "invalid_pixel_count": invalid,
    }


def _binary_counts_from_probability(
    probability_path: Path,
    valid_mask_path: Path,
    threshold: float,
) -> dict[str, Any]:
    """Count binary classes without writing a binary raster."""

    rasterio = _require_rasterio()
    positive = 0
    negative = 0
    invalid = 0

    with rasterio.open(probability_path) as prob_src, rasterio.open(valid_mask_path) as mask_src:
        for _, window in prob_src.block_windows(1):
            probability = prob_src.read(1, window=window, masked=False)
            valid_mask = mask_src.read(1, window=window, masked=False) == 1
            valid = valid_mask & np.isfinite(probability)
            positives = valid & (probability >= threshold)
            negatives = valid & (probability < threshold)
            positive += int(np.count_nonzero(positives))
            negative += int(np.count_nonzero(negatives))
            invalid += int(probability.size - np.count_nonzero(valid))

    return {
        "positive_pixel_count": positive,
        "negative_pixel_count": negative,
        "invalid_pixel_count": invalid,
    }


def _histogram_percentiles(hist: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    """Approximate percentiles from a probability histogram."""

    total = int(hist.sum())
    if total == 0:
        return {key: np.nan for key in ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]}
    cumulative = np.cumsum(hist)
    result: dict[str, float] = {}
    for label, quantile in [
        ("p01", 0.01),
        ("p05", 0.05),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p95", 0.95),
        ("p99", 0.99),
    ]:
        index = int(np.searchsorted(cumulative, quantile * total, side="left"))
        index = max(0, min(index, len(edges) - 2))
        result[label] = float((edges[index] + edges[index + 1]) / 2.0)
    return result


def _probability_stats(
    probability_path: Path,
    valid_mask_path: Path,
    metadata: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Compute windowed all-valid and positive-prediction probability statistics."""

    rasterio = _require_rasterio()
    valid_count = 0
    invalid_count = 0
    min_value = math.inf
    max_value = -math.inf
    sum_value = 0.0
    sum_sq_value = 0.0
    hist = np.zeros(HISTOGRAM_BINS, dtype="int64")
    edges = np.linspace(0.0, 1.0, HISTOGRAM_BINS + 1, dtype="float64")
    positive_count = 0
    positive_sum = 0.0
    positive_hist = np.zeros(HISTOGRAM_BINS, dtype="int64")
    positive_edges = np.linspace(threshold, 1.0, HISTOGRAM_BINS + 1, dtype="float64")

    with rasterio.open(probability_path) as prob_src, rasterio.open(valid_mask_path) as mask_src:
        for _, window in prob_src.block_windows(1):
            probability = prob_src.read(1, window=window, masked=False)
            valid_mask = mask_src.read(1, window=window, masked=False) == 1
            valid = valid_mask & np.isfinite(probability)
            values = probability[valid].astype("float64", copy=False)
            count = int(values.size)
            valid_count += count
            invalid_count += int(probability.size - count)
            if count:
                min_value = min(min_value, float(values.min()))
                max_value = max(max_value, float(values.max()))
                sum_value += float(values.sum())
                sum_sq_value += float(np.square(values).sum())
                clipped = np.clip(values, 0.0, 1.0)
                hist += np.histogram(clipped, bins=edges)[0]
                positive_values = values[values >= threshold]
                if positive_values.size:
                    positive_count += int(positive_values.size)
                    positive_sum += float(positive_values.sum())
                    positive_hist += np.histogram(
                        np.clip(positive_values, threshold, 1.0),
                        bins=positive_edges,
                    )[0]

    mean = sum_value / valid_count if valid_count else np.nan
    variance = (sum_sq_value / valid_count - mean * mean) if valid_count else np.nan
    std = math.sqrt(max(variance, 0.0)) if valid_count else np.nan
    percentiles = _histogram_percentiles(hist, edges)
    positive_percentiles = _histogram_percentiles(positive_hist, positive_edges)
    total_pixels = int(metadata["width"]) * int(metadata["height"])

    return {
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
        "pixel_size_x": float(metadata["pixel_size_x"]),
        "pixel_size_y": float(metadata["pixel_size_y"]),
        "crs": str(metadata["crs"]),
        "valid_pixel_count": valid_count,
        "invalid_pixel_count": invalid_count,
        "valid_ratio": valid_count / total_pixels if total_pixels else np.nan,
        "min": min_value if valid_count else np.nan,
        "max": max_value if valid_count else np.nan,
        "mean": mean,
        "median": percentiles["p50"],
        "std": std,
        **percentiles,
        "positive_probability_pixel_count": positive_count,
        "positive_probability_mean": positive_sum / positive_count if positive_count else np.nan,
        "positive_probability_median": positive_percentiles["p50"],
        "positive_probability_p25": positive_percentiles["p25"],
        "positive_probability_p75": positive_percentiles["p75"],
    }


def _binary_stats_from_counts(
    counts: Mapping[str, Any],
    metadata: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Return binary map area statistics from pixel counts."""

    positive = int(counts.get("positive_pixel_count", 0))
    negative = int(counts.get("negative_pixel_count", 0))
    invalid = int(counts.get("invalid_pixel_count", 0))
    valid = positive + negative
    pixel_area = abs(float(metadata["pixel_size_x"]) * float(metadata["pixel_size_y"]))
    return {
        "threshold": threshold,
        "positive_pixel_count": positive,
        "negative_pixel_count": negative,
        "invalid_pixel_count": invalid,
        "positive_ratio_valid": positive / valid if valid else np.nan,
        "negative_ratio_valid": negative / valid if valid else np.nan,
        "positive_area_m2": positive * pixel_area,
        "positive_area_ha": positive * pixel_area / 10000.0,
        "negative_area_m2": negative * pixel_area,
        "negative_area_ha": negative * pixel_area / 10000.0,
        "valid_pixel_count": valid,
        "valid_area_m2": valid * pixel_area,
        "valid_area_ha": valid * pixel_area / 10000.0,
    }


def _ascii_header(width: int, height: int, transform: Any, nodata: int | float) -> str:
    """Return an ESRI ASCII Grid header."""

    cellsize_x = float(transform.a)
    cellsize_y = abs(float(transform.e))
    if not math.isclose(cellsize_x, cellsize_y, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"ASCII export requires square pixels, got {cellsize_x} x {cellsize_y}.")
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


def _format_ascii_array(array: np.ndarray, nodata: int | float, integer: bool) -> np.ndarray:
    """Convert an array chunk to ASC-ready nodata values."""

    if np.ma.isMaskedArray(array):
        data = np.ma.asarray(array).filled(nodata)
    else:
        data = np.asarray(array)
    if integer:
        out = np.where(np.isfinite(data), data, nodata).astype("int32", copy=False)
    else:
        out = np.where(np.isfinite(data), data, nodata).astype("float32", copy=False)
    return out


def _write_ascii_grid(
    *,
    src_path: Path,
    asc_path: Path,
    txt_path: Path,
    resolution: float | None,
    resampling: str,
    integer: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Write ESRI ASCII Grid and identical .txt copy using chunked reads."""

    rasterio = _require_rasterio()
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    if resolution is None:
        return write_esri_ascii_pair(
            src_path,
            asc_path,
            txt_path,
            nodata=BINARY_NODATA_ASC if integer else PROBABILITY_NODATA_ASC,
            integer=integer,
            precision=0 if integer else 6,
            chunk_rows=ASCII_CHUNK_ROWS,
            overwrite=overwrite,
        ) | {
            "resolution_m": "native",
            "resampling": "none_native_grid",
            "size_bytes_asc": asc_path.stat().st_size,
            "size_bytes_txt": txt_path.stat().st_size,
        }

    if asc_path.exists() and not overwrite:
        print(f"Skipped existing ASC due to --no-overwrite: {asc_path}", flush=True)
        return {
            "asc_path": str(asc_path),
            "txt_path": str(txt_path),
            "resolution_m": resolution if resolution is not None else "native",
            "resampling": resampling,
            "size_bytes_asc": asc_path.stat().st_size if asc_path.exists() else 0,
            "size_bytes_txt": txt_path.stat().st_size if txt_path.exists() else 0,
        }
    if txt_path.exists() and not overwrite:
        print(f"Skipped existing TXT due to --no-overwrite: {txt_path}", flush=True)
        return {
            "asc_path": str(asc_path),
            "txt_path": str(txt_path),
            "resolution_m": resolution if resolution is not None else "native",
            "resampling": resampling,
            "size_bytes_asc": asc_path.stat().st_size if asc_path.exists() else 0,
            "size_bytes_txt": txt_path.stat().st_size if txt_path.exists() else 0,
        }
    asc_path.parent.mkdir(parents=True, exist_ok=True)

    nodata = BINARY_NODATA_ASC if integer else PROBABILITY_NODATA_ASC
    resampling_enum = getattr(Resampling, resampling)

    with rasterio.open(src_path) as src:
        if resolution is None:
            dataset = src
            close_dataset = False
        else:
            transform = rasterio.transform.from_origin(
                src.bounds.left,
                src.bounds.top,
                float(resolution),
                float(resolution),
            )
            width = int(math.ceil((src.bounds.right - src.bounds.left) / float(resolution)))
            height = int(math.ceil((src.bounds.top - src.bounds.bottom) / float(resolution)))
            dataset = WarpedVRT(
                src,
                crs=src.crs,
                transform=transform,
                width=width,
                height=height,
                resampling=resampling_enum,
                nodata=nodata,
            )
            close_dataset = True

        try:
            header = _ascii_header(dataset.width, dataset.height, dataset.transform, nodata)
            with asc_path.open("w", encoding="utf-8") as stream:
                stream.write(header)
                for row_off in range(0, dataset.height, ASCII_CHUNK_ROWS):
                    height = min(ASCII_CHUNK_ROWS, dataset.height - row_off)
                    window = rasterio.windows.Window(0, row_off, dataset.width, height)
                    chunk = dataset.read(1, window=window, masked=True)
                    out = _format_ascii_array(chunk, nodata, integer=integer)
                    fmt = "%d" if integer else "%.6f"
                    np.savetxt(stream, out, fmt=fmt, delimiter=" ")
        finally:
            if close_dataset:
                dataset.close()

    shutil.copy2(asc_path, txt_path)
    return {
        "asc_path": str(asc_path),
        "txt_path": str(txt_path),
        "resolution_m": resolution if resolution is not None else "native",
        "resampling": resampling,
        "size_bytes_asc": asc_path.stat().st_size,
        "size_bytes_txt": txt_path.stat().st_size,
    }


def _write_xyz_grid(
    *,
    src_path: Path,
    xyz_path: Path,
    stride: int,
    integer: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Write a stride-sampled x,y,value grid CSV using chunked reads."""

    rasterio = _require_rasterio()
    if stride <= 0:
        raise ValueError("XYZ stride must be a positive integer.")
    if xyz_path.exists() and not overwrite:
        print(f"Skipped existing XYZ due to --no-overwrite: {xyz_path}", flush=True)
        return {
            "xyz_path": str(xyz_path),
            "stride": stride,
            "rows_written": 0,
            "size_bytes": xyz_path.stat().st_size if xyz_path.exists() else 0,
        }
    xyz_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with rasterio.open(src_path) as src, xyz_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x", "y", "value"])
        col_indices = np.arange(0, src.width, stride, dtype=int)
        xs = src.transform.c + (col_indices + 0.5) * src.transform.a
        for row_off in range(0, src.height, ASCII_CHUNK_ROWS):
            height = min(ASCII_CHUNK_ROWS, src.height - row_off)
            window = rasterio.windows.Window(0, row_off, src.width, height)
            chunk = src.read(1, window=window, masked=True)
            data = np.ma.asarray(chunk)
            for local_row in range(0, height, stride):
                row_index = row_off + local_row
                y = src.transform.f + (row_index + 0.5) * src.transform.e
                values = data[local_row, col_indices]
                mask = np.ma.getmaskarray(values)
                for x, value, is_masked in zip(xs, values, mask):
                    if is_masked:
                        continue
                    numeric = float(value)
                    if not np.isfinite(numeric):
                        continue
                    writer.writerow([f"{float(x):.6f}", f"{float(y):.6f}", int(numeric) if integer else f"{numeric:.6f}"])
                    rows_written += 1

    return {
        "xyz_path": str(xyz_path),
        "stride": stride,
        "rows_written": rows_written,
        "size_bytes": xyz_path.stat().st_size,
    }


def _write_map_png(
    raster_path: Path,
    png_path: Path,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    max_preview_size: int,
    dpi: int,
    overwrite: bool,
    save_vector: bool = True,
    colorbar_label: str = "",
    discrete_binary: bool = False,
) -> None:
    """Write a downsampled raster PNG quicklook."""

    rasterio = _require_rasterio()
    plt = _require_matplotlib_pyplot()
    if png_path.exists() and not overwrite:
        print(f"Skipped existing map PNG due to --no-overwrite: {png_path}", flush=True)
        return
    png_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as src:
        scale = max(src.width / max_preview_size, src.height / max_preview_size, 1.0)
        out_width = max(1, int(src.width / scale))
        out_height = max(1, int(src.height / scale))
        preview = src.read(1, out_shape=(out_height, out_width), masked=True)
        bounds = src.bounds
    preview = np.ma.masked_invalid(preview)
    if discrete_binary:
        preview = np.ma.masked_where(preview == BINARY_NODATA, preview)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#d9d9d9", alpha=1.0)
    fig, ax = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    image = ax.imshow(preview, cmap=cmap_obj, vmin=vmin, vmax=vmax, extent=extent, origin="upper")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    width_m = float(bounds.right - bounds.left)
    scale_m = max(1.0, round(width_m / 5.0 / 10.0) * 10.0)
    x0 = bounds.left + width_m * 0.08
    y0 = bounds.bottom + (bounds.top - bounds.bottom) * 0.08
    ax.plot([x0, x0 + scale_m], [y0, y0], color="black", linewidth=2.5)
    ax.text(x0 + scale_m / 2, y0 + (bounds.top - bounds.bottom) * 0.025, f"{scale_m:g} m", ha="center", va="bottom", fontsize=8)
    ax.annotate("N", xy=(0.94, 0.90), xytext=(0.94, 0.78), xycoords="axes fraction", ha="center", va="center", arrowprops={"arrowstyle": "-|>", "lw": 1.4}, fontsize=9)
    if discrete_binary:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=plt.get_cmap(cmap)(0.05), edgecolor="black", label="0 non-habitat"),
            Patch(facecolor=plt.get_cmap(cmap)(0.95), edgecolor="black", label="1 habitat"),
            Patch(facecolor="#d9d9d9", edgecolor="black", label="invalid"),
        ]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    else:
        fig.colorbar(image, ax=ax, fraction=0.036, pad=0.025, label=colorbar_label or "Value")
    _save_figure_outputs(fig, png_path, dpi=dpi, save_vector=save_vector)
    plt.close(fig)


def _write_stats_text(path: Path, row: Mapping[str, Any], overwrite: bool) -> None:
    """Write one plain-text statistics report."""

    lines = [f"{key}: {value}" for key, value in row.items()]
    _write_text(path, "\n".join(lines) + "\n", overwrite=overwrite)


def _process_final_maps(
    *,
    train_summary: pd.DataFrame,
    dirs: ReportDirs,
    options: ReportOptions,
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create report map products and return map statistics tables."""

    probability_rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    area_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []

    write_xyz = (options.allow_full_xyz or options.xyz_stride is not None) and not options.no_grids
    xyz_stride = 1 if options.allow_full_xyz else int(options.xyz_stride or 10)

    for record in _map_record_iter(train_summary):
        site_id = record["site_id"]
        site_name = record["site_name"]
        candidate_label = record["candidate_label"]
        feature_set = record["feature_set"]
        model_id = record["model_id"]
        probability_src = record["probability_src"]
        mask_src = record["valid_mask_src"]
        _require_existing_file(probability_src, "Final probability raster")
        _require_existing_file(mask_src, "Final valid mask raster")

        print(
            f"Reporting map: {site_id} / {candidate_label} / threshold_{options.threshold:g}",
            flush=True,
        )

        metadata = _raster_profile_record(probability_src)
        mask_metadata = _raster_profile_record(mask_src)
        alignment_ok = (
            metadata["width"] == mask_metadata["width"]
            and metadata["height"] == mask_metadata["height"]
            and str(metadata["crs"]) == str(mask_metadata["crs"])
            and metadata["transform"] == mask_metadata["transform"]
        )
        alignment_rows.append(
            {
                "site_id": site_id,
                "candidate_label": candidate_label,
                "probability_path": str(probability_src),
                "valid_mask_path": str(mask_src),
                "width": metadata["width"],
                "height": metadata["height"],
                "crs": metadata["crs"],
                "pixel_size_x": metadata["pixel_size_x"],
                "pixel_size_y": metadata["pixel_size_y"],
                "grid_match": alignment_ok,
                "status": "ok" if alignment_ok else "error",
                "message": "" if alignment_ok else "Probability raster and valid mask grids differ.",
            }
        )
        if not alignment_ok:
            raise ValueError(f"Probability raster and mask grid mismatch for {site_id}/{candidate_label}.")

        probability_out = _probability_output_path(dirs, site_id, candidate_label)
        binary_out = _binary_output_path(dirs, site_id, candidate_label)
        if not options.no_maps:
            _copy_file(probability_src, probability_out, options.overwrite)
            binary_counts = _write_binary_map(
                probability_src,
                mask_src,
                binary_out,
                options.threshold,
                options.overwrite,
            )
        else:
            binary_counts = _binary_counts_from_probability(
                probability_src,
                mask_src,
                options.threshold,
            )
        probability_stats = _probability_stats(
            probability_src,
            mask_src,
            metadata,
            options.threshold,
        )
        binary_stats = _binary_stats_from_counts(binary_counts, metadata, options.threshold)

        base = {
            "site_id": site_id,
            "site_name": site_name,
            "candidate_label": candidate_label,
            "feature_set": feature_set,
            "model_id": model_id,
        }
        probability_row = {
            **base,
            "raster_path": str(probability_src),
            **probability_stats,
        }
        binary_row = {
            **base,
            **binary_stats,
        }
        area_row = {
            **base,
            "threshold": options.threshold,
            "valid_pixel_count": binary_stats["valid_pixel_count"],
            "valid_area_m2": binary_stats["valid_area_m2"],
            "valid_area_ha": binary_stats["valid_area_ha"],
            "predicted_positive_pixel_count": binary_stats["positive_pixel_count"],
            "predicted_positive_area_m2": binary_stats["positive_area_m2"],
            "predicted_positive_area_ha": binary_stats["positive_area_ha"],
            "predicted_positive_ratio_valid": binary_stats["positive_ratio_valid"],
            "mean_probability": probability_stats["mean"],
            "median_probability": probability_stats["median"],
        }
        probability_rows.append(probability_row)
        binary_rows.append(binary_row)
        area_rows.append(area_row)

        if not options.no_maps:
            _write_map_png(
                probability_out,
                _png_output_path(dirs, site_id, candidate_label, "probability"),
                title=f"{site_name} | {candidate_label} | {feature_set} | {model_id}",
                cmap=options.map_cmap,
                vmin=0.0,
                vmax=1.0,
                max_preview_size=options.max_preview_size,
                dpi=options.dpi,
                overwrite=options.overwrite,
                save_vector=options.save_vector,
                colorbar_label="Habitat probability",
            )
            _write_map_png(
                binary_out,
                _png_output_path(dirs, site_id, candidate_label, "binary_05"),
                title=f"{site_name} | {candidate_label} | threshold={options.threshold:g}",
                cmap="gray_r",
                vmin=0.0,
                vmax=1.0,
                max_preview_size=options.max_preview_size,
                dpi=options.dpi,
                overwrite=options.overwrite,
                save_vector=options.save_vector,
                discrete_binary=True,
            )

        if not options.no_diagnostics:
            diag_base = dirs.diagnostics / f"{site_id}_{candidate_label}"
            _write_stats_text(
                diag_base.with_name(diag_base.name + "_probability_stats.txt"),
                probability_row,
                options.overwrite,
            )
            _write_stats_text(
                diag_base.with_name(diag_base.name + "_binary_05_stats.txt"),
                binary_row,
                options.overwrite,
            )

        if not options.no_asc:
            ascii_dir = dirs.asciis / site_id / candidate_label
            native_dir = ascii_dir / "native"
            resampled_dir = ascii_dir / "resampled_0p5m"
            exports = []
            if not options.skip_native_asc:
                exports.append(
                    (
                        probability_src,
                        native_dir / "probability.asc",
                        native_dir / "probability.txt",
                        None,
                        "nearest",
                        False,
                    )
                )
            if not options.skip_0p5m_asc:
                exports.append(
                    (
                        probability_src,
                        resampled_dir / "probability.asc",
                        resampled_dir / "probability.txt",
                        ASCII_RESAMPLED_RESOLUTION,
                        "bilinear",
                        False,
                    )
                )
            if binary_out.exists():
                if not options.skip_native_asc:
                    exports.append(
                        (
                            binary_out,
                            native_dir / "binary_05.asc",
                            native_dir / "binary_05.txt",
                            None,
                            "nearest",
                            True,
                        )
                    )
                if not options.skip_0p5m_asc:
                    exports.append(
                        (
                            binary_out,
                            resampled_dir / "binary_05.asc",
                            resampled_dir / "binary_05.txt",
                            ASCII_RESAMPLED_RESOLUTION,
                            "nearest",
                            True,
                        )
                    )
            else:
                warnings.append(
                    f"Binary ASC/TXT skipped for {site_id}/{candidate_label} because --no-maps "
                    "prevented binary_05.tif creation."
                )
            ascii_records: list[dict[str, Any]] = []
            for src_path, asc_path, txt_path, resolution, resampling, integer in exports:
                export_record = _write_ascii_grid(
                    src_path=src_path,
                    asc_path=asc_path,
                    txt_path=txt_path,
                    resolution=resolution,
                    resampling=resampling,
                    integer=integer,
                    overwrite=options.overwrite,
                )
                export_record["asc_path"] = Path(export_record["asc_path"]).relative_to(dirs.root).as_posix()
                export_record["txt_path"] = Path(export_record["txt_path"]).relative_to(dirs.root).as_posix()
                export_record.update(
                    {
                        **base,
                        "export_type": "binary_05" if integer else "probability",
                        "format": "ASC_TXT",
                    }
                )
                export_rows.append(export_record)
                ascii_records.append(export_record)
            if not options.no_diagnostics:
                text_lines = [
                    f"{row['export_type']} {row['resolution_m']}: "
                    f"{row['asc_path']} ({row['size_bytes_asc']} bytes), "
                    f"{row['txt_path']} ({row['size_bytes_txt']} bytes)"
                    for row in ascii_records
                ]
                _write_text(
                    dirs.diagnostics / f"{site_id}_{candidate_label}_ascii_export_summary.txt",
                    "\n".join(text_lines) + "\n",
                    options.overwrite,
                )

        if write_xyz:
            grid_dir = dirs.grids / site_id / candidate_label
            xyz_sources = [(probability_src, "probability_xyz_stride.xyz", False)]
            if binary_out.exists():
                xyz_sources.append((binary_out, "binary_05_xyz_stride.xyz", True))
            else:
                warnings.append(
                    f"Binary XYZ skipped for {site_id}/{candidate_label} because --no-maps "
                    "prevented binary_05.tif creation."
                )
            for src_path, name, integer in xyz_sources:
                export_record = _write_xyz_grid(
                    src_path=src_path,
                    xyz_path=grid_dir / name,
                    stride=xyz_stride,
                    integer=integer,
                    overwrite=options.overwrite,
                )
                export_record.update(
                    {
                        **base,
                        "export_type": "binary_05" if integer else "probability",
                        "format": "XYZ",
                    }
                )
                export_rows.append(export_record)
        elif not options.no_grids:
            warnings.append(
                "XYZ grid export skipped by default. Use --xyz-stride or --allow-full-xyz to enable it."
            )

    return (
        pd.DataFrame(probability_rows),
        pd.DataFrame(binary_rows),
        pd.DataFrame(area_rows),
        pd.DataFrame(export_rows),
        pd.DataFrame(alignment_rows),
    )


def _feature_set_definitions(feature_configs: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Build feature-set definition table from YAML configs."""

    rows: list[dict[str, Any]] = []
    for feature_set_id in _ordered(feature_configs.keys(), FEATURE_ORDER):
        config = feature_configs[feature_set_id]
        meta = config.get("feature_set", {})
        features = config.get("features", {})
        flat_features: list[str] = []
        group_counts: dict[str, int] = {}
        if isinstance(features, Mapping):
            for group in ["acoustic", "frequency_difference", "terrain", "texture"]:
                values = features.get(group, [])
                if isinstance(values, list):
                    flat_features.extend(str(value) for value in values)
                    group_counts[group] = len(values)
                else:
                    group_counts[group] = 0
        rows.append(
            {
                "feature_set": _publication_feature_set_label(feature_set_id),
                "feature_set_type": str(meta.get("type", "")).replace("_", " ").title(),
                "frequencies_khz": ", ".join(str(value) for value in meta.get("frequencies", [])),
                "reference_frequency_khz": meta.get("reference_frequency", ""),
                "feature_count": meta.get("feature_count", len(flat_features)),
                "acoustic_count": group_counts.get("acoustic", 0),
                "frequency_difference_count": group_counts.get("frequency_difference", 0),
                "terrain_count": group_counts.get("terrain", 0),
                "texture_count": group_counts.get("texture", 0),
                "features": "; ".join(_display_label(value) for value in flat_features),
            }
        )
    return pd.DataFrame(rows)


def _model_config_summary(model_configs: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Build model configuration summary table from YAML configs."""

    rows: list[dict[str, Any]] = []
    for model_id in _ordered(model_configs.keys(), MODEL_ORDER):
        config = model_configs[model_id]
        params = config.get("params", {})
        key_parts = {
            "random_forest": [
                f"{params.get('n_estimators')} trees",
                "unlimited depth" if params.get("max_depth") is None else f"maximum depth {params.get('max_depth')}",
                f"minimum leaf size {params.get('min_samples_leaf')}",
                f"{params.get('class_weight')} class weight",
            ],
            "lightgbm": [
                f"{params.get('n_estimators')} trees",
                f"learning rate {params.get('learning_rate')}",
                f"{params.get('num_leaves')} leaves",
                f"{params.get('class_weight')} class weight",
            ],
            "catboost": [
                f"{params.get('iterations')} iterations",
                f"learning rate {params.get('learning_rate')}",
                f"depth {params.get('depth')}",
            ],
            "xgboost": [
                f"{params.get('n_estimators')} trees",
                f"learning rate {params.get('learning_rate')}",
                f"maximum depth {params.get('max_depth')}",
            ],
        }
        complete_params = dict(params)
        seed_key = "random_seed" if model_id == "catboost" else "random_state"
        complete_params[seed_key] = "experiment seed"
        rows.append(
            {
                "model": _publication_model_label(model_id),
                "key_configuration": ", ".join(key_parts[model_id]),
                "additional_parameters_json": json.dumps(
                    complete_params,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def _site_sample_summary(
    sites_config: Mapping[str, Any],
    sites: Sequence[str],
    train_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build study-site and final-sample summary table."""

    rows: list[dict[str, Any]] = []
    for site_id in sites:
        site_config = get_site_config(sites_config, site_id)
        env = site_config.get("environment", {})
        habitat = site_config.get("habitat", {})
        site_rows = train_summary[train_summary["site_id"].astype(str) == site_id]
        rows.append(
            {
                "site_id": site_id,
                "site_name": site_config.get("site_name", site_id),
                "short_name": site_config.get("short_name", ""),
                "survey_date": site_config.get("survey_date", ""),
                "coast": env.get("coast", ""),
                "habitat_type": habitat.get("dominant_type", ""),
                "vegetation_group": habitat.get("vegetation_group", ""),
                "morphology": env.get("morphology", ""),
                "depth_min_m": env.get("depth_range_m", {}).get("min", ""),
                "depth_max_m": env.get("depth_range_m", {}).get("max", ""),
                "candidate_count": int(site_rows["candidate_label"].nunique()) if not site_rows.empty else 0,
                "sample_count_min": pd.to_numeric(site_rows.get("sample_count"), errors="coerce").min()
                if not site_rows.empty
                else np.nan,
                "sample_count_max": pd.to_numeric(site_rows.get("sample_count"), errors="coerce").max()
                if not site_rows.empty
                else np.nan,
                "positive_count_min": pd.to_numeric(site_rows.get("positive_count"), errors="coerce").min()
                if not site_rows.empty
                else np.nan,
                "positive_count_max": pd.to_numeric(site_rows.get("positive_count"), errors="coerce").max()
                if not site_rows.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _rank_stability_summary(analysis: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine model, feature, and feature-model rank stability tables."""

    frames = []
    for label, name in [
        ("model", "stage2_model_rank_stability.csv"),
        ("feature", "stage2_feature_rank_stability.csv"),
        ("feature_model", "stage2_feature_model_rank_stability.csv"),
    ]:
        data = analysis[name].copy()
        data.insert(0, "rank_entity_type", label)
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def _final_map_inventory(
    train_summary: pd.DataFrame,
    probability_stats: pd.DataFrame,
    binary_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Build final map inventory table."""

    rows = []
    for _, row in train_summary.iterrows():
        site_id = str(row["site_id"])
        candidate_label = str(row["candidate_label"])
        prob = probability_stats[
            (probability_stats["site_id"].astype(str) == site_id)
            & (probability_stats["candidate_label"].astype(str) == candidate_label)
        ]
        binary = binary_stats[
            (binary_stats["site_id"].astype(str) == site_id)
            & (binary_stats["candidate_label"].astype(str) == candidate_label)
        ]
        rows.append(
            {
                "site_id": site_id,
                "site_name": row.get("site_name", site_id),
                "candidate_label": candidate_label,
                "feature_set": row.get("feature_set", ""),
                "model_id": row.get("model_id", ""),
                "probability_path": row.get("prediction_probability_path", ""),
                "valid_mask_path": row.get("prediction_valid_mask_path", ""),
                "feature_importance_path": row.get("feature_importance_path", ""),
                "width": prob["width"].iloc[0] if not prob.empty else np.nan,
                "height": prob["height"].iloc[0] if not prob.empty else np.nan,
                "crs": prob["crs"].iloc[0] if not prob.empty else "",
                "valid_pixel_count": prob["valid_pixel_count"].iloc[0] if not prob.empty else np.nan,
                "positive_pixel_count": binary["positive_pixel_count"].iloc[0]
                if not binary.empty
                else np.nan,
                "status": row.get("status", ""),
            }
        )
    return pd.DataFrame(rows)


def _project_relative_path(value: Any) -> str:
    """Return a portable project-relative path when possible."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        return text
    framework_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(framework_root))
    except ValueError:
        return path.name


def _portable_cell(value: Any) -> Any:
    """Normalize absolute path values while preserving non-path scalar content."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    text = str(value)
    if "; " in text:
        parts = text.split("; ")
        if any(Path(part).is_absolute() for part in parts if part):
            return "; ".join(_project_relative_path(part) for part in parts)
    return _project_relative_path(text) if Path(text).is_absolute() else value


def _portable_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Replace machine-specific path columns with portable identifiers."""

    portable = data.copy()
    path_columns = [
        column
        for column in portable.columns
        if "path" in str(column).lower()
        or str(column).lower().endswith(("_dir", "_src"))
    ]
    for column in path_columns:
        portable[column] = portable[column].map(_project_relative_path)
        if column.endswith("_path"):
            portable[f"{column[:-5]}_artifact_id"] = portable[column].map(
                lambda value: Path(str(value)).stem if str(value) else ""
            )
    for column in portable.select_dtypes(include=["object"]).columns:
        portable[column] = portable[column].map(_portable_cell)
    return portable


def _normalize_figure_source_frame(path: Path, data: pd.DataFrame) -> pd.DataFrame:
    """Return the portable, plotted-only source schema for an official Figure."""

    out = _portable_dataframe(data)
    parent_stem = path.parent.name
    if path.stem.startswith("fig_s06b_positive_pixel_probability_distribution"):
        if "candidate_label" in out.columns:
            out["candidate_order"] = out["candidate_label"].map(
                {value: index + 1 for index, value in enumerate(CANDIDATE_ORDER)}
            )
        if "site_id" in out.columns:
            out["site_order"] = out["site_id"].map(
                {value: index + 1 for index, value in enumerate(SITE_ORDER)}
            )
        columns = [
            "site_id",
            "site_name",
            "candidate_label",
            "feature_set",
            "model_id",
            "candidate_order",
            "site_order",
            "positive_probability_mean",
            "positive_probability_median",
            "positive_probability_p25",
            "positive_probability_p75",
            "data_semantics",
            "supplementary_figure",
            "panel",
        ]
        out = out[[column for column in columns if column in out.columns]].copy()
    elif path.stem == "fig_s06_final_candidate_analysis":
        unused_probability_columns = {
            "raster_path",
            "width",
            "height",
            "pixel_size_x",
            "pixel_size_y",
            "crs",
            "valid_pixel_count",
            "invalid_pixel_count",
            "valid_ratio",
            "mean",
            "median",
            "min",
            "max",
            "std",
            "p01",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "probability_mean",
            "probability_median",
            "probability_p05",
            "probability_p25",
            "probability_p75",
            "probability_p95",
            "raster_artifact_id",
        }
        out = out.drop(columns=[column for column in unused_probability_columns if column in out.columns])
    if path.stem.startswith("fig_s07") or parent_stem == "fig_s07_feature_importance":
        if "feature" in out.columns:
            out["feature_label"] = out["feature"].map(_display_label)
            out["feature_group"] = out["feature"].map(_feature_group)
            feature_order = out["feature"].dropna().astype(str).drop_duplicates().tolist()[:12]
            selection_order = {feature: index + 1 for index, feature in enumerate(feature_order)}
            out["selection_order"] = out["feature"].astype(str).map(selection_order).astype("Int64")
            out["selection_rule"] = "Top 12 by combined mean importance; common order across panels"
    return out


def _publication_site_summary(
    sites_config: Mapping[str, Any],
    sites: Sequence[str],
    train_summary: pd.DataFrame,
    target_resolution_m: float,
) -> pd.DataFrame:
    """Build the compact publication site and dataset summary."""

    rows: list[dict[str, Any]] = []
    for site_id in _ordered(sites, SITE_ORDER):
        site = get_site_config(sites_config, site_id)
        environment = site.get("environment", {})
        habitat = site.get("habitat", {})
        rows.append(
            {
                "site_id": site_id,
                "site_name": site.get("site_name", site_id),
                "survey_date": site.get("survey_date", ""),
                "habitat_type": habitat.get("dominant_type", ""),
                "morphology": environment.get("morphology", ""),
                "depth_min_m": environment.get("depth_range_m", {}).get("min", np.nan),
                "depth_max_m": environment.get("depth_range_m", {}).get("max", np.nan),
                "frequencies_khz": ";".join(str(value) for value in site.get("frequencies", [])),
                "target_resolution_m": target_resolution_m,
            }
        )
    return pd.DataFrame(rows)


def _sampling_summary(
    output_root: Path,
    sites: Sequence[str],
    feature_set_id: str,
) -> pd.DataFrame:
    """Preserve Stage-04 modeling-sample counts outside the study-site table."""

    rows: list[dict[str, Any]] = []
    for site_id in _ordered(sites, SITE_ORDER):
        path = output_root / "samples" / site_id / f"{feature_set_id}_sample_summary.csv"
        source = _read_csv(path, f"Stage-04 sample summary for {site_id}")
        if source.shape[0] != 1:
            raise ValueError(f"Expected one Stage-04 sample-summary row: {path}")
        row = source.iloc[0]
        total = int(row["total_samples"])
        positive = int(row["positive_samples"])
        negative = int(row["negative_samples"])
        if total != positive + negative:
            raise ValueError(f"Stage-04 sample counts do not sum for {site_id}")
        rows.append(
            {
                "site_id": site_id,
                "feature_set_id": feature_set_id,
                "sample_unit": "Stage-04 labeled raster pixel retained for model training/evaluation",
                "positive_label": "Habitat/vegetated label (class 1)",
                "negative_label": "Non-habitat/non-vegetated label (class 0)",
                "valid_sample_count": total,
                "positive_count": positive,
                "negative_count": negative,
                "positive_ratio": positive / total,
                "negative_to_positive_sampling_ratio": float(row["n_neg_per_pos"]),
                "random_seed": int(row["random_seed"]),
                "source_path": _project_relative_path(path),
                "interpretation_caution": (
                    "The 0.25 positive ratio reflects the Stage-04 3:1 negative-to-positive "
                    "sampling design and is not an estimate of natural habitat coverage."
                ),
            }
        )
    return pd.DataFrame(rows)


def _primary_candidate_feature_set(analysis_config: Mapping[str, Any]) -> str:
    """Return the configured scientific feature set for the publication primary candidate."""

    candidates = {
        item["candidate_label"]: item
        for item in configured_analysis_candidates(analysis_config)
    }
    try:
        return str(candidates[PRIMARY_PUBLICATION_CANDIDATE]["feature_set"])
    except KeyError as exc:
        raise ValueError(
            "publication.primary_candidate_id must reference an analysis candidate: "
            f"{PRIMARY_PUBLICATION_CANDIDATE!r}."
        ) from exc


def _read_run_metrics(output_root: Path, validation_type: str) -> pd.DataFrame:
    """Read existing run-level evaluation metrics without recomputation."""

    files = sorted((output_root / validation_type).rglob("evaluation_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No evaluation_metrics.csv files found for {validation_type}")
    frames = []
    for path in files:
        frame = _read_csv(path, f"{validation_type} evaluation metrics")
        frame["evaluation_metrics_path"] = _project_relative_path(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _threshold_counts(metrics: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Count existing precision-target and fallback outcomes by summary unit."""

    values = metrics.copy()
    met = values["threshold_target_met"].astype(str).str.lower().map({"true": True, "false": False})
    if met.isna().any():
        raise ValueError("Unable to parse threshold_target_met in run-level evaluation metrics")
    values["target_met_runs"] = met.astype(int)
    values["fallback_runs"] = (~met).astype(int)
    return (
        values.groupby(list(keys), as_index=False, dropna=False)[["target_met_runs", "fallback_runs"]]
        .sum()
    )


def _validation_detailed_summary(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    validation_type: str,
) -> pd.DataFrame:
    """Attach threshold provenance and aggregation semantics to the full stored summary."""

    if validation_type == "cross_site":
        keys = ["train_site_id", "test_site_id", "site_id", "feature_set", "model_id"]
        aggregation = "mean across 5 seeds; each seed predicts the complete target-site dataset"
    elif validation_type == "spatial_block_cv":
        keys = ["site_id", "feature_set", "model_id"]
        aggregation = "mean across 25 fold-runs (5 seeds x 5 folds); not pooled OOF"
    else:
        keys = ["site_id", "feature_set", "model_id"]
        aggregation = "mean across 5 seed-specific held-out test subsets"
    counts = _threshold_counts(metrics, keys)
    result = summary.merge(counts, on=keys, how="left", validate="one_to_one")
    if result[["target_met_runs", "fallback_runs"]].isna().any().any():
        raise ValueError(f"Threshold outcome coverage is incomplete for {validation_type}")
    result["aggregation_unit"] = aggregation
    result["threshold_policy"] = (
        "precision-target threshold (target=0.90); fixed threshold 0.5 used only when target was not met"
    )
    return _portable_dataframe(result)


def _validation_publication_summary(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    validation_type: str,
) -> pd.DataFrame:
    """Return the compact reader-facing validation table from stored aggregates."""

    detailed = _validation_detailed_summary(summary, metrics, validation_type)
    common = pd.DataFrame(
        {
            "model": detailed["model_id"].map(_publication_model_label),
            "feature_set": detailed["feature_set"].map(_publication_feature_set_label),
            "AP_mean": detailed["AP_mean"],
            "ROC_AUC_mean": detailed["AUC_mean"],
            "precision_mean": detailed["Precision_mean"],
            "recall_mean": detailed["Recall_mean"],
            "F1_mean": detailed["F1_mean"],
        }
    )
    if validation_type == "cross_site":
        common.insert(
            0,
            "transfer_direction",
            detailed.apply(
                lambda row: (
                    f"{_display_label(row['train_site_id'])} → "
                    f"{_display_label(row['test_site_id'])}"
                ),
                axis=1,
            ),
        )
    else:
        common.insert(0, "site", detailed["site_id"].map(_display_label))
    return common


def _top_scope_rows(summary: pd.DataFrame, validation_type: str, top_n: int = 5) -> pd.DataFrame:
    """Select the stored Top-N model-feature rows within each evaluation scope."""

    group_columns = ["train_site_id", "test_site_id"] if validation_type == "cross_site" else ["site_id"]
    rows: list[dict[str, Any]] = []
    for group_values, group in summary.groupby(group_columns, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        ranked = group.sort_values(["AP_mean", "AUC_mean"], ascending=False).reset_index(drop=True)
        ranked["scope_rank"] = np.arange(1, ranked.shape[0] + 1)
        selected = ranked.head(top_n).copy()
        for row in selected.sort_values("scope_rank").itertuples(index=False):
            train_site = str(getattr(row, "train_site_id", getattr(row, "site_id", "")))
            test_site = str(getattr(row, "test_site_id", getattr(row, "site_id", "")))
            if validation_type == "cross_site":
                scope = f"{_display_label(train_site)} → {_display_label(test_site)}"
            else:
                scope = _display_label(str(getattr(row, "site_id")))
            rows.append(
                {
                    "validation_strategy": VALIDATION_DISPLAY[validation_type],
                    "evaluation_scope": scope,
                    "scope_rank": int(row.scope_rank),
                    "model": _publication_model_label(row.model_id),
                    "feature_set": _publication_feature_set_label(row.feature_set),
                    "AP": float(row.AP_mean),
                    "ROC_AUC": float(row.AUC_mean),
                }
            )
    return pd.DataFrame(rows)


def _key_validation_performance(
    within_summary: pd.DataFrame,
    cross_summary: pd.DataFrame,
    spatial_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the Top-5 model-feature results for each validation scope."""

    frames = [
        _top_scope_rows(within_summary, "within_site"),
        _top_scope_rows(cross_summary, "cross_site"),
        _top_scope_rows(spatial_summary, "spatial_block_cv"),
    ]
    result = pd.concat(frames, ignore_index=True)
    scope_order = {
        ("Within-site", "Bongpyeong"): 0,
        ("Within-site", "Hujeong"): 1,
        ("Cross-site", "Bongpyeong → Hujeong"): 2,
        ("Cross-site", "Hujeong → Bongpyeong"): 3,
        ("Spatial block CV", "Bongpyeong"): 4,
        ("Spatial block CV", "Hujeong"): 5,
    }
    result["_publication_order"] = [
        scope_order[(str(row.validation_strategy), str(row.evaluation_scope))]
        for row in result.itertuples()
    ]
    return (
        result.sort_values(["_publication_order", "scope_rank"], kind="stable")
        .drop(columns="_publication_order")
        .reset_index(drop=True)
    )


def _final_candidate_selection_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Transform Stage-11 candidate statistics into the main selection table."""

    roles = CANDIDATE_TABLE_ROLES
    result = pd.DataFrame(
        {
            "candidate": data["candidate_label"].map(_publication_candidate_label),
            "model": data["model_id"].map(_publication_model_label),
            "feature_set": data["feature_set"].map(_publication_feature_set_label),
            "within_AP": data["within_AP"],
            "cross_AP": data["cross_AP"],
            "spatial_AP": data["spatial_AP"],
            "integrated_AP": data["mean_AP"],
            "minimum_AP": data["min_AP"],
            "AP_range": data["range_AP"],
            "final_map_role": data["candidate_label"].map(lambda value: roles[str(value)][0]),
            "selection_basis": data["candidate_label"].map(lambda value: roles[str(value)][1]),
            "_candidate_id": data["candidate_label"],
        }
    )
    result["_candidate_order"] = result["_candidate_id"].map(
        {value: index for index, value in enumerate(CANDIDATE_ORDER)}
    )
    return (
        result.sort_values("_candidate_order", kind="stable")
        .drop(columns=["_candidate_id", "_candidate_order"])
        .reset_index(drop=True)
    )


def _primary_habitat_summary(
    probability_stats: pd.DataFrame,
    binary_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Build the two-row primary-candidate habitat prediction summary."""

    probability = probability_stats[
        probability_stats["candidate_label"] == PRIMARY_PUBLICATION_CANDIDATE
    ].copy()
    binary = binary_stats[
        binary_stats["candidate_label"] == PRIMARY_PUBLICATION_CANDIDATE
    ].copy()
    merged = probability.merge(
        binary,
        on=["site_id", "site_name", "candidate_label", "feature_set", "model_id", "valid_pixel_count"],
        suffixes=("_probability", "_binary"),
        validate="one_to_one",
    )
    return pd.DataFrame(
        {
            "Site": merged["site_id"].map(_display_label),
            "Candidate": merged["candidate_label"].map(_publication_candidate_label),
            "Feature set": merged["feature_set"].map(_publication_feature_set_label),
            "Model": merged["model_id"].map(_publication_model_label),
            "Valid mapped area (ha)": merged["valid_area_ha"],
            "Predicted habitat extent (ha)": merged["positive_area_ha"],
            "Predicted habitat ratio of valid area (%)": merged["positive_ratio_valid"] * 100.0,
            "Threshold": merged["threshold"],
            "Resolution (m)": merged["pixel_size_x"],
        }
    )


def _integrated_entity_summary(analysis: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine integrated model and feature summaries without duplicating feature-model data."""

    frames = []
    for entity_type, filename, id_column in [
        ("model", "stage2_integrated_model_stats.csv", "model_id"),
        ("feature", "stage2_integrated_feature_stats.csv", "feature_set"),
    ]:
        frame = analysis[filename].copy()
        frame.insert(0, "entity_id", frame[id_column])
        frame.insert(0, "entity_type", entity_type)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _rank_stability_source_summary(analysis: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine canonical rank entities and preserve source identifiers."""

    shift_columns = [
        "within_to_cross_rank_shift", "within_to_spatial_rank_shift", "cross_to_spatial_rank_shift"
    ]
    frames = []
    for entity_type, filename, id_columns in [
        ("model", "stage2_model_rank_stability.csv", ["model_id"]),
        ("feature_set", "stage2_feature_rank_stability.csv", ["feature_set"]),
        ("model_feature", "stage2_feature_model_rank_shift.csv", ["feature_set", "model_id"]),
    ]:
        frame = analysis[filename].copy()
        frame.insert(0, "entity_id", frame[id_columns].astype(str).agg(" + ".join, axis=1))
        frame.insert(0, "entity_type", entity_type)
        for column in shift_columns:
            if column not in frame:
                frame[column] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _rank_stability_shift_summary(analysis: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return reader-facing rank stability and explicitly directed rank shifts."""

    source = _rank_stability_source_summary(analysis)
    rows: list[dict[str, Any]] = []
    for row in source.itertuples(index=False):
        entity_type = str(row.entity_type)
        model_id = getattr(row, "model_id", np.nan)
        feature_set = getattr(row, "feature_set", np.nan)
        rows.append(
            {
                "entity_type": entity_type,
                "entity_label": _publication_entity_label(entity_type, model_id, feature_set),
                "within_AP": row.within_AP,
                "cross_AP": row.cross_AP,
                "spatial_AP": row.spatial_AP,
                "integrated_AP": row.mean_AP,
                "within_rank": row.within_rank,
                "cross_rank": row.cross_rank,
                "spatial_rank": row.spatial_rank,
                "mean_rank": row.mean_rank,
                "median_rank": row.median_rank,
                "best_rank": row.best_rank,
                "worst_rank": row.worst_rank,
                "rank_range": row.rank_range,
                "rank_stability_order": row.rank_stability_order,
                "cross_minus_within_rank": getattr(row, "within_to_cross_rank_shift"),
                "spatial_minus_within_rank": getattr(row, "within_to_spatial_rank_shift"),
                "spatial_minus_cross_rank": getattr(row, "cross_to_spatial_rank_shift"),
            }
        )
    return pd.DataFrame(rows)


def _final_candidate_map_statistics(
    probability_stats: pd.DataFrame,
    binary_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Return one reader-facing row per site and final-candidate map."""

    merged = probability_stats.merge(
        binary_stats,
        on=["site_id", "site_name", "candidate_label", "feature_set", "model_id", "valid_pixel_count"],
        suffixes=("_probability", "_binary"),
        validate="one_to_one",
    )
    result = pd.DataFrame(
        {
            "Site ID": merged["site_id"],
            "Site": merged["site_name"],
            "Candidate": merged["candidate_label"].map(_publication_candidate_label),
            "Feature set": merged["feature_set"].map(_publication_feature_set_label),
            "Valid mapped area (ha)": merged["valid_area_ha"],
            "Predicted habitat extent (ha)": merged["positive_area_ha"],
            "Predicted habitat ratio of valid area (%)": merged["positive_ratio_valid"] * 100.0,
            "Mean predicted probability among predicted habitat pixels": merged["positive_probability_mean"],
            "Median predicted probability among predicted habitat pixels": merged["positive_probability_median"],
            "Predicted probability 25th percentile": merged["positive_probability_p25"],
            "Predicted probability 75th percentile": merged["positive_probability_p75"],
            "_candidate_id": merged["candidate_label"],
        }
    )
    result["_site_order"] = result["Site ID"].map({value: index for index, value in enumerate(SITE_ORDER)})
    result["_candidate_order"] = result["_candidate_id"].map(
        {value: index for index, value in enumerate(CANDIDATE_ORDER)}
    )
    return (
        result.sort_values(["_site_order", "_candidate_order"], kind="stable")
        .drop(columns=["_candidate_id", "_site_order", "_candidate_order"])
        .reset_index(drop=True)
    )


def _portable_feature_importance(data: pd.DataFrame) -> pd.DataFrame:
    """Return compact publication labels for all final RF feature importances."""

    result = pd.DataFrame(
        {
            "site_id": data["site_id"],
            "candidate_label": data["candidate_label"].map(_publication_candidate_label),
            "feature_label": data["feature"].map(_display_label),
            "feature_group": data["feature"].map(_feature_group),
            "importance": data["importance"],
            "importance_rank_within_model": data["rank"],
            "_candidate_id": data["candidate_label"],
        }
    )
    result["_site_order"] = result["site_id"].map({value: index for index, value in enumerate(SITE_ORDER)})
    result["_candidate_order"] = result["_candidate_id"].map(
        {value: index for index, value in enumerate(CANDIDATE_ORDER)}
    )
    return (
        result.sort_values(
            ["_site_order", "_candidate_order", "importance_rank_within_model"],
            kind="stable",
        )
        .drop(columns=["_candidate_id", "_site_order", "_candidate_order"])
        .reset_index(drop=True)
    )


def _representative_validation_table(output_root: Path, target_site: str, threshold: float) -> pd.DataFrame:
    """Build S9/S10 from the same fixed stored predictions used by Figures S8/S9."""

    rows = []
    representative = _representative_run(target_site)
    representative_seed = int(representative["seed"])
    for spec in _supp_representative_paths(output_root, target_site):
        folds = []
        for fold_index, prediction_path in enumerate(spec["prediction_paths"]):
            fold_id = fold_index if spec["validation_strategy"] == "spatial_block_cv" else None
            folds.append(
                _read_supp_representative_predictions(
                    prediction_path,
                    site_id=target_site,
                    validation_strategy=spec["validation_strategy"],
                    train_site_id=spec["train_site_id"],
                    seed=representative_seed,
                    fold_id=fold_id,
                )
            )
        predictions = pd.concat(folds, ignore_index=True)
        if predictions["sample_index"].duplicated().any():
            raise ValueError(f"Duplicate pooled OOF sample_index values for {target_site}")
        _, _, average_precision, roc_auc, _, _ = _supp_curves_from_predictions(predictions)
        counts, _, _ = _supp_confusion_from_predictions(predictions, threshold=threshold)
        tn, fp, fn, tp = (int(counts[0, 0]), int(counts[0, 1]), int(counts[1, 0]), int(counts[1, 1]))
        _supp_validate_stored_probability_metrics(
            spec["evaluation_path"], average_precision=average_precision, roc_auc=roc_auc
        )
        rows.append(
            {
                "target_site": _display_label(target_site),
                "validation_strategy": VALIDATION_DISPLAY[spec["validation_strategy"]],
                "train_site": _display_label(spec["train_site_id"]),
                "test_site": _display_label(target_site),
                "sample_count": int(predictions.shape[0]),
                "Background reference count": int((predictions["y_true"] == 0).sum()),
                "Vegetation reference count": int((predictions["y_true"] == 1).sum()),
                "AP": average_precision,
                "ROC_AUC": roc_auc,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )
    return pd.DataFrame(rows)


def _write_tables(
    *,
    dirs: ReportDirs,
    options: ReportOptions,
    output_root: Path,
    analysis: Mapping[str, pd.DataFrame],
    analysis_config: Mapping[str, Any],
    sites_config: Mapping[str, Any],
    sites: Sequence[str],
    feature_configs: Mapping[str, Mapping[str, Any]],
    model_configs: Mapping[str, Mapping[str, Any]],
    train_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    probability_stats: pd.DataFrame,
    binary_stats: pd.DataFrame,
    area_summary: pd.DataFrame,
    export_inventory: pd.DataFrame,
    file_inventory: pd.DataFrame,
    target_resolution_m: float,
) -> None:
    """Write publication tables, canonical source data, and internal QC tables."""

    print("Writing publication-centered report tables", flush=True)
    validation_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def validation_input(source_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if source_id not in validation_cache:
            summary_path = output_root / source_id / f"{source_id}_summary.csv"
            validation_cache[source_id] = (
                _read_csv(summary_path, summary_path.name),
                _read_run_metrics(output_root, source_id),
            )
        return validation_cache[source_id]

    representative_by_id = {
        item["id"]: item for item in configured_representative_runs(PUBLICATION_CONFIG)
    }

    def build_official_table(spec: Mapping[str, Any]) -> pd.DataFrame:
        builder = spec["builder"]
        if builder == "publication_site_summary":
            return _publication_site_summary(sites_config, sites, train_summary, target_resolution_m)
        if builder == "key_validation_performance":
            within_summary, _ = validation_input("within_site")
            cross_summary, _ = validation_input("cross_site")
            spatial_summary, _ = validation_input("spatial_block_cv")
            return _key_validation_performance(within_summary, cross_summary, spatial_summary)
        if builder == "final_candidate_selection_summary":
            return _final_candidate_selection_summary(analysis["stage2_final_candidate_comparison.csv"])
        if builder == "primary_habitat_summary":
            return _primary_habitat_summary(probability_stats, binary_stats)
        if builder == "feature_set_definitions":
            return _feature_set_definitions(feature_configs)
        if builder == "model_config_summary":
            return _model_config_summary(model_configs)
        if builder == "validation_publication_summary":
            source_id = spec["parameters"]["validation_source"]
            summary, metrics = validation_input(source_id)
            return _validation_publication_summary(summary, metrics, source_id)
        if builder == "rank_stability_shift_summary":
            return _rank_stability_shift_summary(analysis)
        if builder == "final_candidate_map_statistics":
            return _final_candidate_map_statistics(probability_stats, binary_stats)
        if builder == "portable_feature_importance":
            return _portable_feature_importance(feature_importance)
        if builder == "representative_validation_table":
            run = representative_by_id[spec["parameters"]["representative_run"]]
            return _representative_validation_table(output_root, run["target_site"], options.threshold)
        raise ValueError(f"Unsupported configured Table builder: {builder!r}")

    for spec in configured_table_specs(PUBLICATION_CONFIG):
        root = dirs.table_main if spec["role"] == "main" else dirs.table_supplementary
        _write_csv(
            build_official_table(spec),
            root / f"{spec['stem']}.csv",
            options.overwrite,
            float_format="%.4f",
        )

    def validation_source(stem: str) -> pd.DataFrame:
        source_map = {
            "within_site_seed_metrics": ("within_site", False),
            "cross_site_seed_metrics": ("cross_site", False),
            "spatial_block_cv_fold_metrics": ("spatial_block_cv", False),
            "within_site_detailed_summary": ("within_site", True),
            "cross_site_detailed_summary": ("cross_site", True),
            "spatial_block_cv_detailed_summary": ("spatial_block_cv", True),
        }
        if stem in source_map:
            source_id, detailed = source_map[stem]
            summary, metrics = validation_input(source_id)
            return (
                _validation_detailed_summary(summary, metrics, source_id)
                if detailed
                else _portable_dataframe(metrics)
            )
        if stem == "integrated_model_feature_marginals":
            return _integrated_entity_summary(analysis)
        if stem == "integrated_feature_model_matrix":
            return analysis["stage2_integrated_feature_model_stats.csv"]
        if stem == "rank_stability_all_entities":
            return _rank_stability_source_summary(analysis)
        raise ValueError(f"Unsupported configured validation Source Data stem: {stem!r}")

    def figure_source(stem: str) -> pd.DataFrame:
        if stem == "model_rank_stability":
            return analysis["stage2_model_rank_stability.csv"]
        if stem == "feature_rank_stability":
            return analysis["stage2_feature_rank_stability.csv"]
        if stem == "site_rank_correlation":
            return pd.concat(
                [
                    analysis["stage2_site_feature_rank_correlation.csv"].assign(rank_group="feature"),
                    analysis["stage2_site_model_rank_correlation.csv"].assign(rank_group="model"),
                ],
                ignore_index=True,
            )
        return analysis[f"{stem}.csv"]

    def map_source(stem: str) -> pd.DataFrame:
        if stem == "final_map_inventory":
            return _final_map_inventory(train_summary, probability_stats, binary_stats)
        if stem == "probability_statistics":
            return probability_stats
        if stem == "binary_statistics":
            return binary_stats
        raise ValueError(f"Unsupported configured map Source Data stem: {stem!r}")

    def inventory_source(stem: str) -> pd.DataFrame:
        if stem == "final_model_provenance":
            return train_summary
        if stem == "sampling_summary":
            return _sampling_summary(
                output_root,
                sites,
                _primary_candidate_feature_set(analysis_config),
            )
        raise ValueError(f"Unsupported configured inventory Source Data stem: {stem!r}")

    source_roots = {
        "validation": dirs.table_source_validation,
        "figure": dirs.table_source_figures,
        "map": dirs.table_source_maps,
        "inventory": dirs.table_source_inventories,
    }
    source_builders = {
        "validation_source": validation_source,
        "figure_source": figure_source,
        "map_source": map_source,
        "inventory_source": inventory_source,
    }
    for spec in configured_source_data_specs(PUBLICATION_CONFIG):
        data = source_builders[spec["builder"]](spec["stem"])
        root = source_roots[spec["category"]]
        _write_csv(_portable_dataframe(data), root / f"{spec['stem']}.csv", options.overwrite)

    compatibility_names = {
        "A_full_multi_model_comparison": "full_multi_model_comparison",
        "B_single_200_model_comparison": "single200_model_comparison",
    }
    for comparison in configured_focused_model_comparisons(analysis_config):
        stem = comparison["output_stem"]
        output_stem = compatibility_names.get(stem, stem)
        _write_csv(
            analysis[f"{stem}.csv"],
            dirs.table_internal_supporting / f"{output_stem}.csv",
            options.overwrite,
        )
    if export_inventory.empty:
        for inventory_path in [
            output_root / "report_outputs" / "tables" / "internal" / "supporting" / "ascii_export_inventory.csv",
            output_root / "report_outputs" / "tables" / "supporting" / "ascii_export_inventory.csv",
        ]:
            if inventory_path.is_file() and inventory_path.stat().st_size > 1:
                export_inventory = _read_csv(inventory_path, "Existing ASCII export inventory")
                break
    _write_csv(
        export_inventory,
        dirs.table_internal_supporting / "ascii_export_inventory.csv",
        options.overwrite,
    )
    _write_csv(
        file_inventory,
        dirs.table_internal_qc / "qc_file_inventory.csv",
        options.overwrite,
    )


def _save_text_figure(
    path: Path,
    title: str,
    lines: Sequence[str],
    dpi: int,
    overwrite: bool,
    save_vector: bool | None = None,
) -> None:
    """Save a simple text-based figure."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)
    ax.text(
        0.02,
        0.92,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.5,
        transform=ax.transAxes,
    )
    _save_figure_outputs(fig, path, dpi=dpi, save_vector=save_vector)
    plt.close(fig)


def _metric_value_columns(df: pd.DataFrame) -> list[str]:
    """Return validation AP columns available in a comparison table."""

    return [column for column in ["within_AP", "cross_AP", "spatial_AP", "mean_AP"] if column in df.columns]


def _save_grouped_bar(
    df: pd.DataFrame,
    *,
    category: str,
    values: Sequence[str],
    path: Path,
    title: str,
    ylabel: str,
    dpi: int,
    overwrite: bool,
    order: Sequence[str] | None = None,
    top: int | None = None,
) -> None:
    """Save a grouped bar chart."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    if df.empty or category not in df.columns or not values:
        _save_text_figure(path, title, ["No data available."], dpi, overwrite)
        return
    plot_df = df.copy()
    if order is not None:
        plot_df[category] = pd.Categorical(plot_df[category].astype(str), categories=list(order), ordered=True)
        plot_df = plot_df.sort_values(category)
    elif "mean_AP" in plot_df.columns:
        plot_df = plot_df.sort_values("mean_AP", ascending=False)
    if top is not None:
        plot_df = plot_df.head(top)
    labels = [_short_label(_display_label(value)) for value in plot_df[category].astype(str).tolist()]
    x = np.arange(len(labels))
    width = 0.8 / max(len(values), 1)
    fig_width = max(9.0, len(labels) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8), constrained_layout=True)
    all_values: list[float] = []
    for index, column in enumerate(values):
        y = pd.to_numeric(plot_df[column], errors="coerce").to_numpy(dtype="float64")
        all_values.extend([float(value) for value in y if np.isfinite(value)])
        ax.bar(
            x + (index - (len(values) - 1) / 2) * width,
            y,
            width=width,
            label=_display_column_name(column),
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(*_nice_ylim(all_values, lower_floor=0.0))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=min(len(values), 4))
    ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, path, dpi=dpi)
    plt.close(fig)


def _save_heatmap(
    df: pd.DataFrame,
    *,
    row: str,
    column: str,
    value: str,
    path: Path,
    title: str,
    dpi: int,
    overwrite: bool,
    row_order: Sequence[str] | None = None,
    column_order: Sequence[str] | None = None,
    main_suptitle: bool = False,
    options: ReportOptions | None = None,
) -> None:
    """Save an imshow heatmap from a tidy table."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    if df.empty or row not in df.columns or column not in df.columns or value not in df.columns:
        _save_text_figure(path, title, ["No data available."], dpi, overwrite)
        return
    pivot = df.pivot_table(index=row, columns=column, values=value, aggfunc="mean")
    if row_order is not None:
        pivot = pivot.reindex([item for item in row_order if item in pivot.index])
    if column_order is not None:
        pivot = pivot.reindex(columns=[item for item in column_order if item in pivot.columns])
    values = pivot.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if math.isclose(vmin, vmax):
        vmin -= 0.01
        vmax += 0.01
    fig_width = max(8.5, pivot.shape[1] * 1.35)
    fig_height = max(6.0, pivot.shape[0] * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    if main_suptitle:
        fig.suptitle(title, fontsize=MAIN_TITLE_FONTSIZE, fontweight="bold", x=0.5, y=1.04)
    else:
        ax.set_title(title)
    ax.set_xticks(np.arange(pivot.shape[1]))
    x_rotation = 0 if main_suptitle else 35
    x_ha = "center" if main_suptitle else "right"
    ax.set_xticklabels([_short_label(_display_label(item), 16) for item in pivot.columns], rotation=x_rotation, ha=x_ha)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([_short_label(_display_label(item), 22) for item in pivot.index])
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.iloc[y, x]
            if pd.notna(val):
                norm = (float(val) - vmin) / (vmax - vmin)
                text_color = "black" if norm > 0.62 else "white"
                ax.text(x, y, f"{float(val):.3f}", ha="center", va="center", fontsize=8, color=text_color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Integrated AP" if value == "mean_AP" else _display_column_name(value))
    if main_suptitle and path.stem == "fig_06_feature_model_heatmap" and options is not None:
        panel = _publication_panels("fig_06_feature_model_heatmap")[0]
        _export_main_axes(
            fig=fig,
            axes=[ax],
            composite_path=path,
            panels=[(
                str(panel["letter"]),
                str(panel["title"]),
                str(panel["stem"]),
                df.copy(),
            )],
            options=options,
            plot_function="_save_heatmap",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, path, dpi=dpi)
    plt.close(fig)


def _read_quicklook(path: Path, max_preview_size: int) -> np.ma.MaskedArray:
    """Read a downsampled raster quicklook."""

    rasterio = _require_rasterio()
    with rasterio.open(path) as src:
        scale = max(src.width / max_preview_size, src.height / max_preview_size, 1.0)
        out_width = max(1, int(src.width / scale))
        out_height = max(1, int(src.height / scale))
        data = src.read(1, out_shape=(out_height, out_width), masked=True)
    return np.ma.masked_invalid(data)


def _read_quicklook_with_bounds(
    path: Path,
    max_preview_size: int,
    *,
    apply_scale_offset: bool = False,
) -> tuple[np.ma.MaskedArray, Any]:
    """Read a downsampled raster quicklook and return its map bounds."""

    rasterio = _require_rasterio()
    with rasterio.open(path) as src:
        scale = max(src.width / max_preview_size, src.height / max_preview_size, 1.0)
        out_width = max(1, int(src.width / scale))
        out_height = max(1, int(src.height / scale))
        data = src.read(1, out_shape=(out_height, out_width), masked=True)
        if apply_scale_offset:
            data = data.astype("float64") * float(src.scales[0]) + float(src.offsets[0])
        bounds = src.bounds
    return np.ma.masked_invalid(data), bounds


def _read_probability_binary_quicklook(
    probability_path: Path,
    valid_mask_path: Path,
    max_preview_size: int,
    threshold: float,
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray, Any]:
    """Read probability quicklook and derive binary quicklook from a valid mask."""

    rasterio = _require_rasterio()
    with rasterio.open(probability_path) as prob_src, rasterio.open(valid_mask_path) as mask_src:
        if (
            prob_src.width != mask_src.width
            or prob_src.height != mask_src.height
            or prob_src.transform != mask_src.transform
            or str(prob_src.crs) != str(mask_src.crs)
        ):
            raise ValueError(
                "Probability raster and valid mask grid mismatch: "
                f"{probability_path}, {valid_mask_path}"
            )
        scale = max(prob_src.width / max_preview_size, prob_src.height / max_preview_size, 1.0)
        out_width = max(1, int(prob_src.width / scale))
        out_height = max(1, int(prob_src.height / scale))
        probability = prob_src.read(1, out_shape=(out_height, out_width), masked=False)
        valid_mask = mask_src.read(1, out_shape=(out_height, out_width), masked=False) == 1
        bounds = prob_src.bounds
    valid = valid_mask & np.isfinite(probability)
    probability_ma = np.ma.masked_where(~valid, probability)
    binary = np.full(probability.shape, np.nan, dtype="float32")
    binary[valid & (probability >= threshold)] = 1.0
    binary[valid & (probability < threshold)] = 0.0
    binary_ma = np.ma.masked_invalid(binary)
    return probability_ma, binary_ma, bounds


def _add_map_scale_bar(ax: Any, bounds: Any) -> None:
    """Backward-compatible wrapper around the shared scale-bar helper."""

    shared_add_scale_bar(ax, bounds)


def _add_north_arrow(ax: Any) -> None:
    """Backward-compatible wrapper around the shared north-arrow helper."""

    shared_add_north_arrow(ax)


def _save_map_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    raster_key: str,
    path: Path,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    options: ReportOptions,
) -> None:
    """Save a grid of map quicklooks."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    selected = [record for record in records if Path(str(record[raster_key])).exists()]
    if not selected:
        _save_text_figure(path, title, ["No map data available."], options.dpi, options.overwrite)
        return
    cols = min(4, len(selected))
    rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.2 * cols, 4.0 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    is_binary = "binary" in title.lower()
    for ax in axes.ravel():
        ax.set_axis_off()
    for panel_index, (ax, record) in enumerate(zip(axes.ravel(), selected)):
        data = _read_quicklook(Path(str(record[raster_key])), options.max_preview_size)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#d9d9d9", alpha=1.0)
        image = ax.imshow(data, cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.set_title(
            f"{_display_label(record['site_id'])}\n{_short_label(_display_label(record['candidate_label']), 18)}",
            fontsize=10,
        )
        ax.text(
            0.02,
            0.96,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )
        ax.set_axis_off()
    if image is not None and not is_binary:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Habitat probability")
    if is_binary:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=plt.get_cmap(cmap)(0.05), edgecolor="black", label="non-habitat"),
            Patch(facecolor=plt.get_cmap(cmap)(0.95), edgecolor="black", label="habitat"),
            Patch(facecolor="#d9d9d9", edgecolor="black", label="invalid"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)


def _save_probability_histograms(
    probability_stats: pd.DataFrame,
    path: Path,
    options: ReportOptions,
) -> None:
    """Save a compact histogram proxy using percentile summaries."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    if probability_stats.empty:
        _save_text_figure(path, "Probability histograms", ["No data available."], options.dpi, options.overwrite)
        return
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    labels = [
        f"{row.site_id}\n{row.candidate_label}"
        for row in probability_stats.itertuples(index=False)
    ]
    x = np.arange(len(labels))
    med = pd.to_numeric(probability_stats["median"], errors="coerce")
    p25 = pd.to_numeric(probability_stats["p25"], errors="coerce")
    p75 = pd.to_numeric(probability_stats["p75"], errors="coerce")
    ax.errorbar(x, med, yerr=[med - p25, p75 - med], fmt="o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Probability distributions by final map")
    ax.set_ylim(0, 1)
    if options.hist_log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)


def _save_feature_importance_boxplot(
    feature_importance: pd.DataFrame,
    path: Path,
    options: ReportOptions,
) -> None:
    """Save feature-importance distribution boxplot across final models."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    if feature_importance.empty or "feature" not in feature_importance.columns or "importance" not in feature_importance.columns:
        _save_text_figure(path, "Feature importance boxplot", ["No feature importance data available."], options.dpi, options.overwrite)
        return
    data = feature_importance.copy()
    data["importance"] = pd.to_numeric(data["importance"], errors="coerce")
    order = (
        data.groupby("feature", dropna=False)["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .index.astype(str)
        .tolist()
    )
    grouped = [data.loc[data["feature"].astype(str) == feature, "importance"].dropna().to_numpy() for feature in order]
    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.5), 6.2), constrained_layout=True)
    ax.boxplot(grouped, labels=[_short_label(item, 18) for item in order], showfliers=False)
    ax.set_title("Feature importance distributions across final models")
    ax.set_ylabel("Importance")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)


def _save_difference_map(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_label: str,
    dirs: ReportDirs,
    options: ReportOptions,
    path: Path,
    kind: str = "probability",
) -> None:
    """Save candidate-minus-primary probability or binary difference maps by site."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    sites = _ordered({str(record["site_id"]) for record in records}, SITE_ORDER)
    fig, axes = plt.subplots(
        1,
        len(sites),
        figsize=(5 * max(len(sites), 1), 4.5),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    diffs: list[tuple[Any, str]] = []
    for site_id in sites:
        raster_name = "binary_05.tif" if kind == "binary" else "probability.tif"
        primary_path = dirs.maps / site_id / PRIMARY_PUBLICATION_CANDIDATE / raster_name
        other_path = dirs.maps / site_id / candidate_label / raster_name
        if not primary_path.exists() or not other_path.exists():
            diffs.append((None, site_id))
            continue
        primary = _read_quicklook(primary_path, options.max_preview_size)
        other = _read_quicklook(other_path, options.max_preview_size)
        shape = (
            min(primary.shape[0], other.shape[0]),
            min(primary.shape[1], other.shape[1]),
        )
        diff = other[: shape[0], : shape[1]] - primary[: shape[0], : shape[1]]
        diffs.append((diff, site_id))
    finite_abs = [
        float(np.nanmax(np.abs(np.ma.filled(diff, np.nan))))
        for diff, _ in diffs
        if diff is not None and np.isfinite(np.ma.filled(diff, np.nan)).any()
    ]
    vmax_abs = max(finite_abs) if finite_abs else 1.0
    vmax_abs = max(vmax_abs, 0.01)
    for ax, (diff, site_id) in zip(axes.ravel(), diffs):
        if diff is None:
            ax.set_title(f"{site_id}\nmissing")
            ax.set_axis_off()
            continue
        image = ax.imshow(diff, cmap=options.difference_cmap, vmin=-vmax_abs, vmax=vmax_abs)
        ax.set_title(site_id)
        ax.set_axis_off()
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, label=f"{kind.capitalize()} difference")
    fig.suptitle(f"{candidate_label} minus {PRIMARY_PUBLICATION_CANDIDATE} ({kind})")
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)


def _sample_curve_frame(path: Path, max_points: int = 600) -> pd.DataFrame:
    """Read and evenly sample one curve CSV for plotting/source data."""

    data = _read_csv(path, path.name)
    if data.shape[0] > max_points:
        indices = np.linspace(0, data.shape[0] - 1, max_points).round().astype(int)
        data = data.iloc[indices].copy()
    data["source_path"] = str(path)
    return data


def _collect_curve_frames(root: Path, file_name: str, max_files: int = 300) -> pd.DataFrame:
    """Collect sampled curve CSVs under a validation output root."""

    paths = sorted(root.rglob(file_name))
    frames = []
    for path in paths[:max_files]:
        try:
            frames.append(_sample_curve_frame(path))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _collect_confusion_frames(root: Path, max_files: int = 1000) -> pd.DataFrame:
    """Collect confusion matrix CSVs under a validation output root."""

    frames = []
    for path in sorted(root.rglob("confusion_matrix.csv"))[:max_files]:
        try:
            data = _read_csv(path, path.name)
        except Exception:
            continue
        data["source_path"] = str(path)
        frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _save_validation_curve_figure(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    options: ReportOptions,
) -> None:
    """Save sampled ROC/PR curve figure without recomputing metrics."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        raise FileExistsError(f"Output exists and --no-overwrite was set: {path}")
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        _save_text_figure(path, title, ["No curve CSV files found."], options.dpi, options.overwrite)
        return

    group_cols = [
        column
        for column in ["site_id", "train_site_id", "test_site_id", "feature_set", "model_id", "seed", "fold_id"]
        if column in data.columns
    ]
    fig, ax = plt.subplots(figsize=(6.8, 6.0), constrained_layout=True)
    grouped = data.groupby(group_cols, dropna=False) if group_cols else [(None, data)]
    for _, group in grouped:
        x = pd.to_numeric(group[x_col], errors="coerce")
        y = pd.to_numeric(group[y_col], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.any():
            ax.plot(x[valid], y[valid], color="#2563eb", alpha=0.08, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)


def _raw_study_raster_candidates(project_root: Path, site_id: str, layer: str) -> list[Path]:
    """Return sorted raw bathymetry or sonar GeoTIFF candidates for Fig.1."""

    if layer == "bathy":
        search_dir = project_root / "01_data" / site_id / "bathy"
        keywords = ["300", "bathy", "bathymetry", "depth", "dtm", "dem"]
    elif layer == "sonar":
        search_dir = project_root / "01_data" / site_id / "sonar"
        keywords = ["300", "backscatter", "bs", "sonar", "mosaic"]
    else:
        raise ValueError(f"Unsupported Fig.1 raw layer: {layer}")
    if not search_dir.exists():
        return []
    candidates = sorted(
        [
            path
            for path in search_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ]
    )
    if not candidates:
        return []

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        value = 0
        for index, keyword in enumerate(keywords):
            if keyword in name:
                value += (len(keywords) - index) * 10
        if "300" in name:
            value += 100
        return (-value, path.name)

    return sorted(candidates, key=score)


def _select_raw_study_raster(project_root: Path, site_id: str, layer: str) -> Path | None:
    """Select a raw bathymetry or sonar GeoTIFF for Fig.1."""

    candidates = _raw_study_raster_candidates(project_root, site_id, layer)
    return candidates[0] if candidates else None


def _validate_raster_mask_grid(raster: Any, mask: Any, raster_path: Path, mask_path: Path) -> None:
    """Require a common-valid mask on the exact plotted raster grid."""

    if (
        raster.width != mask.width
        or raster.height != mask.height
        or raster.transform != mask.transform
        or str(raster.crs) != str(mask.crs)
    ):
        raise ValueError(
            "Figure 1 raster and common-valid mask grid mismatch: "
            f"{raster_path}; {mask_path}"
        )


def _study_area_raster_sources(
    output_root: Path,
    selection: Mapping[str, Any],
) -> tuple[Path, Path | None]:
    """Resolve one configured Figure 1 raster and its optional valid-area mask."""

    site_id = str(selection["site"])
    layer = str(selection["layer"])
    file_name = str(selection["file_name"])
    source_domain = str(selection.get("source_domain", "project_input"))
    if source_domain == "project_input":
        return output_root.parent.parent / "01_data" / site_id / layer / file_name, None
    if source_domain == "prepared_common_valid":
        prepared = output_root / "prepared" / site_id
        mask_name = str(selection.get("valid_mask_file_name", "common_valid_mask.tif"))
        return prepared / file_name, prepared / mask_name
    raise ValueError(f"Unsupported Figure 1 source domain: {source_domain!r}")


def _raster_value_summary(
    path: Path,
    valid_mask_path: Path | None = None,
) -> dict[str, Any]:
    """Return exact single-band raster value metadata using windowed reads."""

    rasterio = _require_rasterio()
    min_value = math.inf
    max_value = -math.inf
    valid_count = 0
    mean_value = 0.0
    sum_squared_deviations = 0.0
    with rasterio.open(path) as src:
        mask_src = rasterio.open(valid_mask_path) if valid_mask_path is not None else None
        if mask_src is not None:
            _validate_raster_mask_grid(src, mask_src, path, valid_mask_path)
        scale = float(src.scales[0])
        offset = float(src.offsets[0])
        try:
            for _, window in src.block_windows(1):
                chunk = np.ma.asarray(src.read(1, window=window, masked=True))
                if mask_src is not None:
                    valid = mask_src.read(1, window=window, masked=False) > 0
                    chunk = np.ma.masked_where(~valid, chunk)
                values = chunk.compressed() if np.ma.is_masked(chunk) else np.asarray(chunk).ravel()
                if values.size == 0:
                    continue
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                values = values.astype("float64") * scale + offset
                chunk_count = int(values.size)
                chunk_mean = float(np.mean(values))
                chunk_squared_deviations = float(
                    np.sum((values - chunk_mean) ** 2, dtype="float64")
                )
                if valid_count == 0:
                    mean_value = chunk_mean
                    sum_squared_deviations = chunk_squared_deviations
                else:
                    combined_count = valid_count + chunk_count
                    mean_delta = chunk_mean - mean_value
                    sum_squared_deviations += (
                        chunk_squared_deviations
                        + mean_delta**2 * valid_count * chunk_count / combined_count
                    )
                    mean_value += mean_delta * chunk_count / combined_count
                valid_count += chunk_count
                min_value = min(min_value, float(np.min(values)))
                max_value = max(max_value, float(np.max(values)))
        finally:
            if mask_src is not None:
                mask_src.close()
        return {
            "path": str(path),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "min": min_value if valid_count else np.nan,
            "max": max_value if valid_count else np.nan,
            "valid_count": valid_count,
            "mean": mean_value if valid_count else np.nan,
            "population_std": (
                math.sqrt(max(sum_squared_deviations, 0.0) / valid_count)
                if valid_count
                else np.nan
            ),
            "crs": str(src.crs),
            "scale": float(src.scales[0]),
            "offset": float(src.offsets[0]),
        }


def _read_study_area_quicklook(
    raster_path: Path,
    valid_mask_path: Path | None,
    max_preview_size: int,
    *,
    apply_scale_offset: bool = False,
) -> tuple[np.ma.MaskedArray, Any]:
    """Read a Figure 1 quicklook clipped to configured common-valid coverage."""

    rasterio = _require_rasterio()
    with rasterio.open(raster_path) as src:
        scale = max(src.width / max_preview_size, src.height / max_preview_size, 1.0)
        out_width = max(1, int(src.width / scale))
        out_height = max(1, int(src.height / scale))
        data = np.ma.asarray(src.read(1, out_shape=(out_height, out_width), masked=True))
        if apply_scale_offset:
            data = data.astype("float64") * float(src.scales[0]) + float(src.offsets[0])
        bounds = src.bounds
        if valid_mask_path is not None:
            with rasterio.open(valid_mask_path) as mask_src:
                _validate_raster_mask_grid(src, mask_src, raster_path, valid_mask_path)
                valid = mask_src.read(1, out_shape=(out_height, out_width), masked=False) > 0
            data = np.ma.masked_where(~valid, data)
    return np.ma.masked_invalid(data), bounds


def _save_study_area_figure(
    train_summary: pd.DataFrame,
    probability_stats: pd.DataFrame,
    output_root: Path,
    path: Path,
    options: ReportOptions,
    warnings: list[str],
) -> None:
    """Create Fig.1 from site bathymetry and 300 kHz backscatter rasters."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    panel_specs = _publication_panels("fig_01_study_areas")
    source_rows: list[dict[str, Any]] = []
    selected_rasters: dict[tuple[str, str], Path] = {}
    selected_masks: dict[tuple[str, str], Path | None] = {}
    selected_stats: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[str] = []
    for panel in panel_specs:
        selection = panel["selection"]
        site_id = str(selection["site"])
        layer = str(selection["layer"])
        raster_path, mask_path = _study_area_raster_sources(output_root, selection)
        if not raster_path.exists():
            missing.append(str(raster_path))
        elif mask_path is not None and not mask_path.exists():
            missing.append(str(mask_path))
        else:
            selected_rasters[(site_id, layer)] = raster_path
            selected_masks[(site_id, layer)] = mask_path
            selected_stats[(site_id, layer)] = _raster_value_summary(raster_path, mask_path)
    if missing:
        message = "Fig.1 skipped: required configured bathymetry/backscatter raster source(s) missing: " + "; ".join(missing)
        warnings.append(message)
        print(f"Warning: {message}", flush=True)
        _remove_stale_figure_outputs(path)
        if source_rows:
            _write_figure_source(path, pd.DataFrame(source_rows), options)
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.08, hspace=0.34, wspace=0.30)
    for ax, panel in zip(axes.ravel(), panel_specs):
        selection = panel["selection"]
        panel_label = f"({panel['letter']})"
        site_id = str(selection["site"])
        layer = str(selection["layer"])
        map_type = str(selection["map_type"])
        cmap = str(selection["cmap"])
        colorbar_label = str(selection["colorbar_label"])
        raster_path = selected_rasters[(site_id, layer)]
        mask_path = selected_masks[(site_id, layer)]
        data, bounds = _read_study_area_quicklook(
            raster_path,
            mask_path,
            options.max_preview_size,
            apply_scale_offset=(layer == "sonar"),
        )
        metadata = _raster_profile_record(raster_path)
        stats = selected_stats[(site_id, layer)]
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("#d9d9d9")
        image_limits: dict[str, float] = {}
        if layer == "sonar":
            population_std = float(stats["population_std"])
            image_limits = {
                "vmin": float(stats["mean"]) - 3.0 * population_std,
                "vmax": float(stats["mean"]) + 3.0 * population_std,
            }
        image = ax.imshow(
            data,
            cmap=cmap_obj,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            origin="upper",
            **image_limits,
        )
        ax.set_title(f"{panel_label} {_display_label(site_id)} {map_type}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m, UTM Zone 52N)", fontsize=8)
        ax.set_ylabel("Northing (m, UTM Zone 52N)", fontsize=8)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=7, rotation=0)
        add_map_cartography(ax, bounds)
        cbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.045)
        cbar.set_label(colorbar_label, fontsize=8)
        source_rows.append(
            {
                "panel": panel_label,
                "site_id": site_id,
                "site_label": _display_label(site_id),
                "source_layer": layer,
                "map_type": map_type,
                "raster_path": str(raster_path),
                "valid_mask_path": str(mask_path) if mask_path is not None else "",
                "source_domain": str(selection.get("source_domain", "project_input")),
                "dtype": stats["dtype"],
                "nodata": stats["nodata"],
                "min": stats["min"],
                "max": stats["max"],
                "crs": str(metadata["crs"]),
                "status": "used",
            }
        )
    _export_main_axes(
        fig=fig,
        axes=list(axes.ravel()),
        composite_path=path,
        panels=[
            (
                str(panel["letter"]),
                str(panel["title"]),
                str(panel["stem"]),
                pd.DataFrame([source_rows[index]]),
            )
            for index, panel in enumerate(panel_specs)
        ],
        options=options,
        plot_function="_save_study_area_figure",
    )
    _add_centered_main_title(fig, "Study areas and MBES raster layers", y=0.975)
    _save_main_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    _write_figure_source(path, pd.DataFrame(source_rows), options)


def _save_workflow_validation_figure(path: Path, options: ReportOptions) -> None:
    """Create a vertical workflow and validation framework flowchart."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(8.2, 11.0), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    steps = [
        ("MBES data acquisition", "200 / 300 / 400 kHz\nbathymetry + backscatter", "#dbeafe"),
        ("Preprocessing", "co-registration / 0.07 m resampling\ncommon area", "#e0f2fe"),
        ("Feature generation", "acoustic 3 + NDI 2\nbathymetric 2 + texture 3", "#dcfce7"),
        ("Labeled samples", "vegetated / non-vegetated", "#fef9c3"),
        ("Model x feature matrix", "7 feature sets x 4 ML models", "#fef3c7"),
        ("Validation framework", "within-site / cross-site\nspatial block CV", "#fee2e2"),
        ("Integrated candidate selection", "validation synthesis", "#f3e8ff"),
        ("Final retraining and habitat mapping", "site-specific final maps", "#e5e7eb"),
    ]
    box_w = 0.72
    box_h = 0.075
    x0 = 0.14
    y_top = 0.90
    gap = 0.032
    box_axes: list[Any] = []
    source_rows: list[dict[str, Any]] = []
    for index, (heading, detail, color) in enumerate(steps):
        y = y_top - index * (box_h + gap)
        box_ax = ax.inset_axes([x0, y - box_h / 2, box_w, box_h])
        box_ax.set_facecolor(color)
        box_ax.set_xticks([])
        box_ax.set_yticks([])
        for spine in box_ax.spines.values():
            spine.set_color("#374151")
            spine.set_linewidth(1.2)
        letter = chr(ord("a") + index)
        box_ax.text(0.04, 0.67, f"({letter}) {heading}", ha="left", va="center", fontsize=10.5, fontweight="bold")
        box_ax.text(0.04, 0.26, detail, ha="left", va="center", fontsize=8.8, color="#374151")
        box_axes.append(box_ax)
        source_rows.append({"panel_letter": letter, "title": heading, "text": detail})
        if index < len(steps) - 1:
            y_next = y_top - (index + 1) * (box_h + gap)
            ax.annotate(
                "",
                xy=(0.50, y_next + box_h / 2 + 0.004),
                xytext=(0.50, y - box_h / 2 - 0.004),
                arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#374151"},
            )
    panel_specs = _publication_panels("fig_02_workflow_validation_framework")
    _export_main_axes(
        fig=fig,
        axes=box_axes,
        composite_path=path,
        panels=[
            (
                str(panel["letter"]),
                str(panel["title"]),
                str(panel["stem"]),
                pd.DataFrame([source_rows[index]]),
            )
            for index, panel in enumerate(panel_specs)
        ],
        options=options,
        plot_function="_save_workflow_validation_figure",
    )
    fig.suptitle("Workflow and validation framework", fontsize=MAIN_TITLE_FONTSIZE, fontweight="bold", y=0.985)
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    _write_figure_source(path, pd.DataFrame(source_rows), options)


def _save_feature_definition_figure(
    feature_configs: Mapping[str, Mapping[str, Any]],
    path: Path,
    options: ReportOptions,
) -> None:
    """Create Fig.3 10-band feature-stack definition infographic."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    groups = [
        (
            "(a) Acoustic intensity",
            ["a200_norm", "a300_norm", "a400_norm"],
            "x_norm = (x - depth-bin median) / max(P90 - P10, epsilon)",
            "#e7f3e8",
        ),
        (
            "(b) Frequency-difference indices",
            ["NDI_4_2_norm", "NDI_4_3_norm"],
            "NDI 400-200 / 400-300 = (A - B) / (|A| + |B|)",
            "#f1eaf8",
        ),
        (
            "(c) Bathymetric features",
            ["dpth_lite_300", "dpth_rel_300"],
            "dpth_rel = (z - median_local) / max(IQR_local, epsilon)",
            "#e5f0fa",
        ),
        (
            "(d) Texture features",
            ["a300_low", "a300_high", "a300_log1p"],
            "low-pass; high = a300 - low; log(1 + |LoG|)",
            "#faefe1",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.10, hspace=0.24, wspace=0.18)
    source_rows: list[dict[str, Any]] = []
    for ax, (title, names, formula, color) in zip(axes.ravel(), groups):
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0.02, 0.10), 0.96, 0.82, facecolor=color, edgecolor="#4b5563", linewidth=1.0))
        ax.text(0.06, 0.82, title, fontsize=12.5, fontweight="bold", transform=ax.transAxes)
        y0 = 0.66
        for index, feature in enumerate(names):
            ax.text(0.08, y0 - index * 0.11, _display_label(feature), fontsize=10.5, transform=ax.transAxes)
            source_rows.append(
                {
                    "feature_group": title,
                    "feature_name": feature,
                    "feature_label": _display_label(feature),
                    "formula": formula,
                }
            )
        ax.text(0.06, 0.20, formula, fontsize=9.2, style="italic", transform=ax.transAxes)
    fig.text(
        0.5,
        0.035,
        "10-band feature stack = 3 acoustic + 2 NDI + 2 bathymetric + 3 texture | GLCM not used.",
        ha="center",
        va="center",
        fontsize=10,
    )
    panel_specs = _publication_panels("fig_03_mbes_feature_definitions")
    _export_main_axes(
        fig=fig,
        axes=list(axes.ravel()),
        composite_path=path,
        panels=[
            (
                str(panel["letter"]),
                str(panel["title"]),
                str(panel["stem"]),
                pd.DataFrame([row for row in source_rows if row["feature_group"].startswith(f"({panel['letter']})")]),
            )
            for panel in panel_specs
        ],
        options=options,
        plot_function="_save_feature_definition_figure",
    )
    fig.suptitle("MBES-derived 10-band feature stack", fontsize=MAIN_TITLE_FONTSIZE, fontweight="bold", x=0.5, y=0.965)
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    _write_figure_source(path, pd.DataFrame(source_rows), options)


def _save_confusion_summary_figure(
    data: pd.DataFrame,
    *,
    path: Path,
    title: str,
    options: ReportOptions,
) -> None:
    """Save aggregate confusion matrix figure from stored confusion CSVs."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    required = ["TP", "TN", "FP", "FN"]
    if data.empty or any(column not in data.columns for column in required):
        _save_text_figure(path, title, ["No confusion matrix CSV files found."], options.dpi, options.overwrite)
        return
    totals = {column: int(pd.to_numeric(data[column], errors="coerce").fillna(0).sum()) for column in required}
    matrix = np.array([[totals["TN"], totals["FP"]], [totals["FN"], totals["TP"]]], dtype="float64")
    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    for y in range(2):
        for x in range(2):
            ax.text(x, y, f"{int(matrix[y, x]):,}", ha="center", va="center", fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)


def _write_validation_curve_figures(
    *,
    output_root: Path,
    dirs: ReportDirs,
    options: ReportOptions,
    warnings: list[str],
) -> None:
    """Generate validation ROC, PR, and confusion figures from stored outputs."""

    specs = [
        ("within_site", output_root / "within_site", "Within-site"),
        ("cross_site", output_root / "cross_site", "Cross-site"),
        ("spatial_block_cv", output_root / "spatial_block_cv", "Spatial block CV"),
    ]
    for slug, root, label in specs:
        if not root.exists():
            warnings.append(f"Validation figure input missing: {root}")
            continue
        roc = _collect_curve_frames(root, "roc_curve.csv")
        pr = _collect_curve_frames(root, "pr_curve.csv")
        confusion = _collect_confusion_frames(root)
        roc_path = dirs.figures / f"fig_validation_{slug}_roc.png"
        pr_path = dirs.figures / f"fig_validation_{slug}_pr.png"
        confusion_path = dirs.figures / f"fig_validation_{slug}_confusion_matrix.png"
        _save_validation_curve_figure(
            roc,
            x_col="fpr",
            y_col="tpr",
            path=roc_path,
            title=f"{label} ROC curves",
            xlabel="False positive rate",
            ylabel="True positive rate",
            options=options,
        )
        _save_validation_curve_figure(
            pr,
            x_col="recall",
            y_col="precision",
            path=pr_path,
            title=f"{label} precision-recall curves",
            xlabel="Recall",
            ylabel="Precision",
            options=options,
        )
        _save_confusion_summary_figure(
            confusion,
            path=confusion_path,
            title=f"{label} aggregate confusion matrix",
            options=options,
        )
        if options.generate_source_data:
            for data, figure_path in [(roc, roc_path), (pr, pr_path), (confusion, confusion_path)]:
                source_path = figure_path.with_name(figure_path.stem + "_source.csv")
                if not data.empty:
                    _write_csv(data, source_path, options.overwrite)


def _caption_text(path: Path, kind: str) -> str:
    """Return a generic caption for one figure or table artifact."""

    title = path.stem.replace("_", " ")
    project_id = str(PUBLICATION_CONFIG.get("project_id", DEFAULT_PROJECT_ID))
    if kind == "figure":
        return (
            f"{title}.\n\n"
            f"Generated automatically from {project_id} validation summaries, final-map outputs, "
            "or report-stage raster statistics. Values are derived from existing pipeline "
            "outputs; no model retraining or validation recomputation is performed in stage 13.\n"
        )
    return (
        f"{title}.\n\n"
        f"Automatically generated {project_id} report table assembled from existing validation, "
        "analysis, final-map, or QC outputs. Stage 13 does not alter source results.\n"
    )


def _publication_table_caption(record: Mapping[str, Any]) -> str:
    """Return a submission-ready caption for an official publication table."""

    table_id = str(record["table_id"])
    if table_id not in TABLE_CAPTIONS:
        raise ValueError(f"No publication Table caption configured for {table_id!r}.")
    return TABLE_CAPTIONS[table_id]

    heading = f"{record['table_number']}. {record['title']}."
    details = {
        "table_01_study_sites_dataset_summary": (
            "Survey context for the Bongpyeong and Hujeong study sites. Each row represents one survey site. "
            "Depth values are signed elevations in metres relative to the source vertical datum, so negative "
            "values indicate positions below that datum. Frequencies are in kilohertz (kHz), and target "
            "resolution is the common 0.07 m analysis-grid cell size rather than survey positional accuracy."
        ),
        "table_02_key_validation_performance": (
            "Top-performing model-feature combinations within each validation scope. Each scope contains its "
            "five highest stored mean average-precision (AP) results without forced inclusion of a reference "
            "combination. Random Forest with the Full 10-band feature set is therefore absent when it ranks below "
            "the top five. Scope rank is recalculated "
            "within one site or transfer direction; rank 1 is best. Receiver operating characteristic area under "
            "the curve (ROC AUC) and AP are threshold independent. Within-site and cross-site values are means "
            "across five seeds; spatial-block cross-validation (CV) values are means across 25 fold-runs "
            "(five seeds by five folds), not pooled out-of-fold estimates. Final integrated selection is reported "
            "separately in Table 3."
        ),
        "table_03_final_candidate_selection": (
            "Comparison of the four final Random Forest (RF) candidates in the same order used in Figure 7. "
            "Average precision (AP) is summarized separately for within-site, cross-site, and spatial-block "
            "cross-validation (CV). Integrated AP is the equal-weight arithmetic mean of those three AP values; "
            "it is neither sample weighted nor a pooled AP. Minimum AP and AP range summarize robustness across "
            "validation strategies and should not be interpreted as confidence intervals."
        ),
        "table_04_primary_habitat_prediction_summary": (
            "Mapped habitat extent for the primary balanced Random Forest candidate. Each row represents one "
            "site. Valid mapped area includes only finite prediction pixels accepted by the final feature mask; "
            "NoData and invalid feature cells are excluded. Predicted habitat uses the fixed pixel probability "
            "threshold p >= 0.5. Predicted habitat ratio equals predicted habitat extent divided by valid mapped area. Areas "
            "derive from the 0.07 m raster grid and are reported in hectares (ha)."
        ),
        "table_s01_feature_set_composition": (
            "Definitions and composition of the seven feature sets used throughout model training and validation. "
            "Each row is one feature set; frequencies and reference frequency are in kHz. Feature counts separate "
            "acoustic intensity, normalized difference index (NDI), bathymetric terrain, and texture predictors."
        ),
        "table_s02_model_configuration": (
            "YAML-defined classifier configurations for Random Forest, LightGBM, CatBoost, and XGBoost. Each row "
            "is one model. Key configuration provides a concise reader-facing summary. Complete model parameters "
            "are provided in the JSON configuration column, including the model-specific runtime seed key."
        ),
        "table_s03_full_within_site_results": (
            "Within-site performance for every site, model, and feature set. Each row is a mean across five "
            "seed-specific held-out test subsets. Average precision (AP) and receiver operating characteristic "
            "area under the curve (ROC AUC) are threshold independent. Precision, recall, and F1 use the stored "
            "precision-target threshold policy (target 0.90); all 280 runs met that target. Variability and full "
            "threshold provenance remain in the unnumbered validation source data."
        ),
        "table_s04_full_cross_site_results": (
            "Directional cross-site transfer performance for every model and feature set. Each row is a mean "
            "across five seeds evaluated on the complete target-site dataset; transfer directions are not pooled. "
            "AP and ROC AUC are threshold independent. Precision, recall, and F1 combine stored precision-target "
            "thresholds with fixed 0.5 fallbacks: 89 of 280 runs met the 0.90 target and 191 used fallback. The "
            "detailed threshold outcomes remain in the unnumbered validation source data."
        ),
        "table_s05_full_spatial_cv_results": (
            "Spatial-block cross-validation (CV) performance for every site, model, and feature set. Each row is "
            "the mean across 25 fold-runs (five seeds by five folds); these are not 25 independent experiments "
            "and are not pooled out-of-fold (OOF) predictions. AP and ROC AUC are threshold independent. Of "
            "1,400 fold-runs, 874 met the 0.90 precision target and 526 used threshold 0.5; threshold-dependent "
            "precision, recall, and F1 should therefore be interpreted with the detailed threshold provenance "
            "provided in the unnumbered validation source data."
        ),
        "table_s06_rank_stability_shift": (
            "Rank stability of model, feature-set, and model-feature entities across validation strategies. Rank "
            "1 is best. Integrated AP is the equal-weight mean of within-site, cross-site, and spatial block CV AP, not "
            "a pooled or sample-weighted metric. Rank shifts are defined as later minus earlier rank: positive "
            "values indicate deterioration, negative values improvement, and zero no change. Shifts are available "
            "only for model-feature entities; other entries are NA. rank_stability_order is the stored ordering "
            "from Stage 11 based on increasing mean rank, then rank range and Integrated AP."
        ),
        "table_s07_final_candidate_map_statistics": (
            "Site-level map statistics for all four final Random Forest candidates (two sites by four candidates). "
            "Probability statistics were calculated only from pixels classified as habitat (p >= 0.5), whereas "
            "habitat ratio was calculated relative to the full valid mapped area. Positive-probability quartiles "
            "are deterministic approximations from a 10,000-bin histogram. NoData and invalid feature cells are "
            "excluded from valid mapped area. positive_area_ratio is a 0-1 proportion equal to "
            "predicted_habitat_area_ha divided by valid_mapped_area_ha, not a percentage. Pixel probability is "
            "distinct from dataset-level average precision (AP)."
        ),
        "table_s08_final_feature_importance": (
            "Feature importance for all eight final Random Forest (RF) models (two sites by four candidates). Each "
            "row is one feature within one final model. Importance is the stored impurity-based RF importance and "
            "rank is within that model; values are neither permutation importance nor validation performance. "
            "Feature labels and groups match Figure S7."
        ),
        "table_s09_representative_validation_hujeong": (
            "Representative Hujeong validation for RF, Full 10-band, seed 42. Within-site uses the held-out "
            "test subset, spatial block CV pools five OOF folds, and cross-site uses Bongpyeong-to-Hujeong complete "
            "target-site predictions. sample_count is the number of prediction records; positive and negative "
            "counts are actual habitat and non-habitat labels. AP and ROC AUC use stored probabilities; true "
            "negative (TN), false positive (FP), false negative (FN), and true positive (TP) counts use p >= 0.5. "
            "No run averaging or confidence interval is applied."
        ),
        "table_s10_representative_validation_bongpyeong": (
            "Representative Bongpyeong validation for RF, Full 10-band, seed 42. Within-site uses the held-out "
            "test subset, spatial block CV pools five OOF folds, and cross-site uses Hujeong-to-Bongpyeong complete "
            "target-site predictions. sample_count is the number of prediction records; positive and negative "
            "counts are actual habitat and non-habitat labels. AP and ROC AUC use stored probabilities; true "
            "negative (TN), false positive (FP), false negative (FN), and true positive (TP) counts use p >= 0.5. "
            "No run averaging or confidence interval is applied."
        ),
    }
    return f"{heading}\n\n{details.get(table_id, record['description'])}\n"


TABLE_DOCUMENTATION = {
    "table_01_study_sites_dataset_summary": {
        "purpose": "Describe the survey setting and common spatial grid for the two study sites.",
        "paired": "Main Figures 1-2",
        "citation": "Methods - Study sites and survey data",
        "row_unit": "One surveyed study site.",
        "aggregation": "No statistical aggregation; site metadata are selected from the project/site configuration.",
        "threshold": "Not applicable.",
        "source": "config_sites.yaml and config_project.yaml; deterministic field selection by Stage 13.",
        "caution": "Signed depth values are not habitat-area estimates, and grid resolution is not positional accuracy.",
    },
    "table_02_key_validation_performance": {
        "purpose": "Show the five strongest model-feature combinations in each validation scope.",
        "paired": "Main Figures 4-6",
        "citation": "Results - Validation performance",
        "row_unit": "One model-feature combination within one site or directional transfer scope.",
        "aggregation": "Five-seed means for within/cross-site; 25 fold-run means for spatial block CV. The five highest AP rows are selected per scope without forced reference inclusion.",
        "threshold": "AP and ROC AUC are threshold independent; threshold-dependent metrics are intentionally excluded.",
        "source": "within_site_summary.csv, cross_site_summary.csv, and spatial_block_cv_summary.csv; deterministic scope ranking by stored AP and ROC AUC.",
        "caution": "Scope ranks cannot be compared as pooled sample-level performance across validation strategies.",
    },
    "table_03_final_candidate_selection": {
        "purpose": "Compare the four final mapping candidates and document why each was retained.",
        "paired": "Main Figure 7",
        "citation": "Results - Final candidate selection",
        "row_unit": "One final Random Forest candidate.",
        "aggregation": "Integrated AP is an equal-weight mean of within-site, cross-site, and spatial block CV AP.",
        "threshold": "AP is threshold independent.",
        "source": "analysis_compare/stage2_final_candidate_comparison.csv; deterministic selection, label mapping, and column ordering.",
        "caution": "Integrated AP is not pooled, sample weighted, or a confidence estimate.",
    },
    "table_04_primary_habitat_prediction_summary": {
        "purpose": "Report valid mapped area and predicted habitat extent for the primary candidate.",
        "paired": "Main Figure 9",
        "citation": "Results - Predicted habitat extent",
        "row_unit": "One primary-candidate probability map for one site.",
        "aggregation": "Pixel counts are converted to area with the stored 0.07 m raster resolution.",
        "threshold": "Predicted habitat is probability p >= 0.5; invalid and NoData pixels are excluded.",
        "source": "Stage-13 probability_statistics.csv and binary_statistics.csv derived window-by-window from Stage-12 final maps.",
        "caution": "Habitat ratio describes valid mapped pixels, not the full raster bounding box or entire study region.",
    },
    "table_s01_feature_set_composition": {
        "purpose": "Define the composition of each feature set used by the experiments.",
        "paired": "Main Figure 3",
        "citation": "Methods - Feature generation",
        "row_unit": "One YAML-defined feature set.",
        "aggregation": "Feature counts are deterministic counts of YAML feature lists.",
        "threshold": "Not applicable.",
        "source": "03_experiments/<PROJECT_ID>/feature_sets/*.yaml and Stage 02-03 feature naming.",
        "caution": "Feature membership does not measure feature importance or model performance.",
    },
    "table_s02_model_configuration": {
        "purpose": "Expose the stored classifier hyperparameters used in validation.",
        "paired": "Main Figure 2 and Methods",
        "citation": "Methods - Machine-learning models",
        "row_unit": "One YAML-defined classifier.",
        "aggregation": "No aggregation; key settings are summarized and the complete YAML parameter mapping is serialized as JSON.",
        "threshold": "Not applicable.",
        "source": "03_experiments/<PROJECT_ID>/models/*.yaml and utils/modeling.py seed injection.",
        "caution": "The concise configuration is not exhaustive; the JSON column is authoritative and experiment seeds are supplied at runtime.",
    },
    "table_s03_full_within_site_results": {
        "purpose": "Compare all model-feature combinations under within-site validation.",
        "paired": "Supplementary Figure S1",
        "citation": "Results - Within-site performance",
        "row_unit": "One site, model, and feature-set combination.",
        "aggregation": "Mean across five seed-specific held-out test subsets.",
        "threshold": "AP/AUC are threshold independent; precision/recall/F1 use the 0.90 precision-target policy. All runs met target.",
        "source": "within_site/within_site_summary.csv plus run-level threshold provenance; deterministic column selection.",
        "caution": "Means summarize different seeded held-out subsets and are not pooled predictions.",
    },
    "table_s04_full_cross_site_results": {
        "purpose": "Compare all model-feature combinations for both directional transfers.",
        "paired": "Supplementary Figure S2",
        "citation": "Results - Cross-site transferability",
        "row_unit": "One transfer direction, model, and feature-set combination.",
        "aggregation": "Mean across five seeds, each predicting the complete target-site dataset.",
        "threshold": "AP/AUC are threshold independent; precision/recall/F1 mix target-derived and 0.5 fallback thresholds. Detailed outcomes remain in source data.",
        "source": "cross_site/cross_site_summary.csv plus run-level threshold provenance; deterministic column selection.",
        "caution": "Transfer directions are distinct and are not pooled; threshold-dependent means require the status qualifier.",
    },
    "table_s05_full_spatial_cv_results": {
        "purpose": "Compare all model-feature combinations under spatial-block cross-validation.",
        "paired": "Supplementary Figure S3",
        "citation": "Results - Spatial generalization",
        "row_unit": "One site, model, and feature-set combination.",
        "aggregation": "Mean across 25 fold-runs (five seeds by five folds), not pooled OOF.",
        "threshold": "AP/AUC are threshold independent; precision/recall/F1 mix target-derived and 0.5 fallback thresholds. Detailed outcomes remain in source data.",
        "source": "spatial_block_cv/spatial_block_cv_summary.csv plus fold-level threshold provenance; deterministic column selection.",
        "caution": "The 25 fold-runs are not independent experiments and differ from seed-42 pooled OOF in Tables S9-S10.",
    },
    "table_s06_rank_stability_shift": {
        "purpose": "Quantify how model, feature-set, and combination ranks change across validation strategies.",
        "paired": "Supplementary Figures S5 and S6(a)",
        "citation": "Results - Ranking robustness",
        "row_unit": "One model, feature set, or model-feature entity.",
        "aggregation": "Ranks use stored validation-level AP summaries; integrated AP gives equal weight to three validation strategies.",
        "threshold": "AP and all derived ranks are threshold independent.",
        "source": "Stage-11 model, feature, and feature-model rank-stability/shift CSVs; deterministic union and label mapping.",
        "caution": "Positive shifts mean rank deterioration because larger rank numbers are worse; shifts are NA for non-combination entities.",
    },
    "table_s07_final_candidate_map_statistics": {
        "purpose": "Compare pixel-level probability distributions and mapped habitat extent for all final candidates.",
        "paired": "Supplementary Figure S6(b-c)",
        "citation": "Results - Final-map sensitivity",
        "row_unit": "One site and final-candidate map.",
        "aggregation": "Positive-probability summaries cover only valid pixels with p >= 0.5; quartiles use a deterministic 10,000-bin histogram.",
        "threshold": "Positive-probability summaries and predicted habitat extent use p >= 0.5; the ratio uses the full valid mapped area as denominator.",
        "source": "Stage-13 probability_statistics.csv and binary_statistics.csv from Stage-12 final maps; deterministic join.",
        "caution": "Conditional positive-pixel probability describes confidence among mapped habitat pixels and does not measure habitat prevalence; pixel probability is not AP.",
    },
    "table_s08_final_feature_importance": {
        "purpose": "Provide every final-model feature importance shown or summarized in Main Figure 8 and Supplementary Figure S7.",
        "paired": "Main Figure 8 and Supplementary Figure S7",
        "paired_heading": "Paired Figures",
        "citation": "Results - Feature importance",
        "row_unit": "One feature within one site-candidate final Random Forest model.",
        "aggregation": "No aggregation in the table; ranks are within each stored final model.",
        "threshold": "Not applicable.",
        "source": "final_maps/*/*/feature_importance.csv; deterministic label/group mapping and column selection.",
        "caution": "Impurity importance is model specific and does not establish causal or permutation importance.",
    },
    "table_s09_representative_validation_hujeong": {
        "purpose": "Provide exact Hujeong metrics and confusion counts corresponding to Figure S8.",
        "paired": "Supplementary Figure S8",
        "citation": "Results - Representative validation comparison",
        "row_unit": "One validation strategy for target site Hujeong.",
        "aggregation": "Within/cross rows use one seed-42 prediction set; spatial uses seed-42 five-fold pooled OOF predictions.",
        "threshold": "AP/AUC use probabilities; TN/FP/FN/TP use p >= 0.5.",
        "source": "Stored RF + Full 10-band seed-42 predictions used by Figure S8; no retraining or multi-run averaging.",
        "caution": "The three rows use different validation populations and should not be interpreted as identically sampled tests.",
    },
    "table_s10_representative_validation_bongpyeong": {
        "purpose": "Provide exact Bongpyeong metrics and confusion counts corresponding to Figure S9.",
        "paired": "Supplementary Figure S9",
        "citation": "Results - Representative validation comparison",
        "row_unit": "One validation strategy for target site Bongpyeong.",
        "aggregation": "Within/cross rows use one seed-42 prediction set; spatial uses seed-42 five-fold pooled OOF predictions.",
        "threshold": "AP/AUC use probabilities; TN/FP/FN/TP use p >= 0.5.",
        "source": "Stored RF + Full 10-band seed-42 predictions used by Figure S9; no retraining or multi-run averaging.",
        "caution": "The three rows use different validation populations and should not be interpreted as identically sampled tests.",
    },
}


def _table_documentation(record: Mapping[str, Any]) -> Mapping[str, str]:
    """Return table documentation, with a neutral fallback for configured IDs."""

    table_id = str(record["table_id"])
    title = str(record.get("title", table_id))
    generic = {
        "purpose": f"Provide the configured publication table: {title}.",
        "paired": "See publication.yaml pairings.",
        "citation": "Project publication specification.",
        "row_unit": "One row of the configured table builder output.",
        "aggregation": "Defined by the allowlisted Stage 13 table builder.",
        "threshold": "See table values and project analysis configuration.",
        "source": "Configured upstream artifacts and deterministic Stage 13 serialization.",
        "caution": "Interpret values according to the configured project analysis policy.",
    }
    mode = PUBLICATION_CONFIG.get("publication", {}).get("table_documentation_mode", "canonical")
    if mode == "generic":
        return generic
    if mode != "canonical":
        raise ValueError(f"Unsupported publication.table_documentation_mode: {mode!r}.")
    return TABLE_DOCUMENTATION.get(table_id, generic)


COLUMN_DEFINITIONS = {
    "site_id": "Stable project site identifier.",
    "site_name": "Publication site name.",
    "survey_date": "Date of the source multibeam survey.",
    "habitat_type": "Dominant labeled habitat context from the site configuration.",
    "morphology": "Configured seabed morphology description.",
    "depth_min_m": "Minimum signed depth/elevation value reported for the site.",
    "depth_max_m": "Maximum signed depth/elevation value reported for the site.",
    "frequencies_khz": "Acoustic frequencies included in the survey.",
    "target_resolution_m": "Common analysis-grid cell size.",
    "validation_strategy": "Validation design used for the row.",
    "evaluation_scope": "Site or directional transfer within which scope_rank is defined.",
    "scope_rank": "AP-based rank within one evaluation scope; 1 is best.",
    "model": "Publication model label.",
    "feature_set": "Publication feature-set label.",
    "AP": "Average precision, the area summary of the precision-recall relationship.",
    "ROC_AUC": "Area under the receiver operating characteristic curve.",
    "train_site": "Site used to train the directional cross-site model.",
    "test_site": "Site used for evaluation.",
    "candidate": "Publication label for a final mapping candidate.",
    "within_AP": "Average precision under within-site validation.",
    "cross_AP": "Average precision under cross-site validation.",
    "spatial_AP": "Average precision under spatial-block cross-validation.",
    "integrated_AP": "Equal-weight mean of within_AP, cross_AP, and spatial_AP.",
    "minimum_AP": "Minimum of the three validation-strategy AP values.",
    "AP_range": "Maximum minus minimum validation-strategy AP.",
    "final_map_role": "Planned analytical role of the candidate map.",
    "selection_basis": "Stored rationale for retaining the candidate.",
    "valid_mapped_area_ha": "Area represented by finite valid prediction pixels.",
    "predicted_habitat_area_ha": "Valid mapped area classified as habitat at the fixed threshold.",
    "habitat_ratio_of_valid_area": "Predicted habitat extent divided by valid mapped area.",
    "threshold": "Probability cutoff used for binary classification.",
    "resolution_m": "Raster cell size in metres.",
    "feature_set_type": "Single-, dual-, or multi-frequency feature-set class.",
    "reference_frequency_khz": "Frequency used as the reference for derived terrain/texture predictors.",
    "feature_count": "Number of predictors in the feature set.",
    "features": "Semicolon-separated publication feature labels.",
    "task": "Configured prediction task.",
    "random_state": "Source of the runtime estimator seed.",
    "key_configuration": "Concise reader-facing summary of influential configured hyperparameters.",
    "additional_parameters_json": "Complete configured parameter mapping, including the runtime experiment-seed key, serialized as JSON.",
    "transfer_direction": "Publication train-site to test-site direction.",
    "AP_mean": "Mean average precision across the stated repeat unit.",
    "ROC_AUC_mean": "Mean ROC AUC across the stated repeat unit.",
    "precision_mean": "Mean positive predictive value at stored evaluation thresholds.",
    "recall_mean": "Mean sensitivity at stored evaluation thresholds.",
    "F1_mean": "Mean harmonic mean of precision and recall at stored thresholds.",
    "entity_type": "Ranked entity level: model, feature_set, or model_feature.",
    "entity_label": "Publication-ready entity label.",
    "within_rank": "Rank under within-site validation; 1 is best.",
    "cross_rank": "Rank under cross-site validation; 1 is best.",
    "spatial_rank": "Rank under spatial block CV; 1 is best.",
    "mean_rank": "Arithmetic mean of the three validation ranks.",
    "median_rank": "Median of the three validation ranks.",
    "best_rank": "Smallest (best) rank across validation strategies.",
    "worst_rank": "Largest (worst) rank across validation strategies.",
    "rank_range": "worst_rank minus best_rank.",
    "rank_stability_order": "Stored Stage-11 stability ordering.",
    "cross_minus_within_rank": "cross_rank minus within_rank.",
    "spatial_minus_within_rank": "spatial_rank minus within_rank.",
    "spatial_minus_cross_rank": "spatial_rank minus cross_rank.",
    "candidate_label": "Publication label for the final candidate.",
    "probability_mean": "Mean predicted habitat probability across valid map pixels.",
    "probability_median": "Histogram-approximated median probability across valid map pixels.",
    "probability_p05": "Histogram-approximated 5th percentile of valid-pixel probability.",
    "probability_p25": "Histogram-approximated 25th percentile of valid-pixel probability.",
    "probability_p75": "Histogram-approximated 75th percentile of valid-pixel probability.",
    "probability_p95": "Histogram-approximated 95th percentile of valid-pixel probability.",
    "positive_probability_mean": "Mean probability among valid pixels classified as habitat at p >= 0.5.",
    "positive_probability_median": "Histogram-approximated median probability among valid pixels classified as habitat at p >= 0.5.",
    "positive_probability_p25": "Histogram-approximated 25th percentile among valid pixels classified as habitat at p >= 0.5.",
    "positive_probability_p75": "Histogram-approximated 75th percentile among valid pixels classified as habitat at p >= 0.5.",
    "positive_area_ratio": "Predicted habitat extent divided by valid mapped area, expressed as a 0-1 proportion.",
    "feature_label": "Publication feature label shared with Figure S7.",
    "feature_group": "Publication feature group shared with Main Figure 8 and Figure S7.",
    "importance": "Stored Random Forest impurity-based feature importance.",
    "importance_rank_within_model": "Descending importance rank within one final model.",
    "target_site": "Publication target-site name.",
    "sample_count": "Number of prediction records in the validation row.",
    "negative_count": "Number of actual non-habitat labels.",
    "positive_count": "Number of actual habitat labels.",
    "TN": "True-negative count at the fixed threshold.",
    "FP": "False-positive count at the fixed threshold.",
    "FN": "False-negative count at the fixed threshold.",
    "TP": "True-positive count at the fixed threshold.",
}


def _column_unit(column: str) -> str:
    """Return a concise publication unit for a table column."""

    if column.endswith("_ha"):
        return "ha"
    if column.endswith("_khz"):
        return "kHz"
    if column.endswith("_m"):
        return "m"
    if column in {"AP", "ROC_AUC", "AP_mean", "ROC_AUC_mean", "within_AP", "cross_AP", "spatial_AP", "integrated_AP", "minimum_AP", "AP_range", "probability_mean", "probability_median", "probability_p05", "probability_p25", "probability_p75", "probability_p95", "positive_probability_mean", "positive_probability_median", "positive_probability_p25", "positive_probability_p75", "positive_area_ratio", "habitat_ratio_of_valid_area", "importance", "precision_mean", "recall_mean", "F1_mean", "threshold"}:
        return "unitless"
    return "NA"


def _possible_values(series: pd.Series) -> str:
    """Summarize possible values without embedding a full table copy."""

    values = series.dropna()
    if values.empty:
        return "NA only"
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return f"{numeric.min():.6g} to {numeric.max():.6g}"
    unique = values.astype(str).drop_duplicates().tolist()
    shown = ", ".join(unique[:8])
    return shown + (f", ... ({len(unique)} values)" if len(unique) > 8 else "")


def _markdown_cell(value: Any) -> str:
    """Escape one value for a Markdown table cell."""

    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_table_dictionary(
    path: Path,
    data: pd.DataFrame,
    record: Mapping[str, Any],
    overwrite: bool,
) -> None:
    """Write a reader-facing dictionary for one official table."""

    context = _table_documentation(record)
    lines = [
        f"# {record['table_number']}. {record['title']}",
        "",
        "## Purpose",
        "",
        f"- **Paper question:** {context['purpose']}",
        f"- **{context.get('paired_heading', 'Paired Figure')}:** {context['paired']}",
        f"- **First manuscript citation:** {context['citation']}",
        "",
        "## Row unit",
        "",
        context["row_unit"],
        "",
        "## Column dictionary",
        "",
        "| Column name | Publication label | Definition | Data type | Unit | Possible values | Calculation/source | Interpretation | Important caution |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for column in data.columns:
        definition = COLUMN_DEFINITIONS.get(
            str(column),
            str(column).replace("_", " ").capitalize() + ".",
        )
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in [
                    column,
                    _display_column_name(str(column)),
                    definition,
                    str(data[column].dtype),
                    _column_unit(str(column)),
                    _possible_values(data[column]),
                    context["source"],
                    definition,
                    context["caution"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Aggregation",
            "",
            context["aggregation"],
            "",
            "## Threshold",
            "",
            context["threshold"],
            "",
            "## Source lineage",
            "",
            f"- **Upstream and transformation:** {context['source']}",
            "- **Stage:** Stage 13 deterministic reporting transformation; no model training or prediction.",
            f"- **{context.get('paired_heading', 'Paired Figure')} source:** {context['paired']} source CSV sidecar(s).",
            "",
            "## Interpretation notes",
            "",
            f"- {context['purpose']}",
            f"- **Do not over-interpret:** {context['caution']}",
            "",
        ]
    )
    _write_text(path, "\n".join(lines), overwrite)


def _write_table_index(
    dirs: ReportDirs,
    official_rows: Sequence[Mapping[str, Any]],
    options: ReportOptions,
) -> None:
    """Write the canonical index for all official publication tables."""

    lines = [
        f"# {options.project_id} Official Table Index",
        "",
        "Official tables use a four-file contract: CSV, caption, metadata, and dictionary.",
        "",
        "| Number | Title | Purpose | Paired Figure | CSV | Dictionary | Caption |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in official_rows:
        context = _table_documentation(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["table_number"]),
                    str(row["title"]),
                    context["purpose"],
                    context["paired"],
                    f"[{row['filename_csv']}]({Path(str(row['csv_path'])).relative_to('tables')})",
                    f"[{row['dictionary_file']}]({Path(str(row['dictionary_path'])).relative_to('tables')})",
                    f"[{row['caption_file']}]({Path(str(row['caption_path'])).relative_to('tables')})",
                ]
            )
            + " |"
        )
    lines.append("")
    _write_text(dirs.tables / "TABLE_INDEX.md", "\n".join(lines), options.overwrite)


def _write_sidecar_text(path: Path, text: str, overwrite: bool) -> None:
    """Write a sidecar text file unless protected by no-overwrite."""

    if path.exists() and not overwrite:
        return
    _write_text(path, text, overwrite=True)


def _configured_panel_paths(root: Path, *, role: str) -> list[Path]:
    """Return configured panel PNG paths without discovering stale files."""

    paths: list[Path] = []
    for figure in configured_figure_specs(PUBLICATION_CONFIG):
        if figure["role"] != role:
            continue
        for panel in figure.get("panels", []):
            parent = (
                MAIN_PANEL_PARENT_DIRS.get(
                    panel["stem"].split("_panel_", 1)[0],
                    figure["stem"],
                )
                if role == "main"
                else figure["stem"]
            )
            paths.append(root / parent / f"{panel['stem']}.png")
    return paths


def _configured_figure_paths(dirs: ReportDirs) -> list[Path]:
    """Return configured composite and panel PNG paths in specification order."""

    paths: list[Path] = []
    for figure in configured_figure_specs(PUBLICATION_CONFIG):
        root = dirs.main_composites if figure["role"] == "main" else dirs.supplementary_composites
        paths.append(root / f"{figure['stem']}.png")
    paths.extend(_configured_panel_paths(dirs.main_panels, role="main"))
    paths.extend(_configured_panel_paths(dirs.supplementary_panels, role="supplementary"))
    return [path for path in paths if path.exists()]


def _write_artifact_sidecars(
    *,
    dirs: ReportDirs,
    options: ReportOptions,
    output_root: Path,
) -> None:
    """Write captions, source-data manifests, and metadata for figures/tables."""

    generated_time = datetime.now().isoformat(timespec="seconds")
    figure_paths = _configured_figure_paths(dirs)
    main_rows, supplementary_rows, source_rows, internal_rows = _current_table_manifest_rows(dirs)
    active_table_rows = main_rows + supplementary_rows + source_rows + internal_rows
    table_records = {str(row["table_id"]): row for row in active_table_rows}
    table_paths = [dirs.root / str(row["csv_path"]) for row in active_table_rows]

    if options.generate_captions:
        for path in figure_paths:
            caption_path = path.with_name(path.stem + "_caption.txt")
            if caption_path.exists():
                continue
            _write_sidecar_text(
                caption_path,
                _caption_text(path, "figure"),
                options.overwrite,
            )
        for path in table_paths:
            record = table_records[path.stem]
            if bool(record["official"]):
                heading = f"{record['table_number']}. {record['title']}."
            else:
                heading = f"{record['title']}."
            _write_sidecar_text(
                path.with_name(path.stem + "_caption.txt"),
                _publication_table_caption(record)
                if bool(record["official"])
                else (
                    f"{heading}\n\n{record['description']} This unnumbered artifact is not an "
                    "official manuscript table.\n"
                ),
                options.overwrite,
            )

    if options.generate_source_data:
        for path in figure_paths:
            source_path = path.with_name(path.stem + "_source.csv")
            if source_path.exists():
                continue
            source = pd.DataFrame(
                [
                    {
                        "figure_file": str(path),
                        "source_root": str(output_root),
                        "source_description": (
                            "analysis_compare CSVs, validation curve/confusion CSVs, "
                            "final_maps rasters, or report-stage map statistics"
                        ),
                    }
                ]
            )
            _write_csv(_portable_dataframe(source), source_path, options.overwrite)

    if options.generate_metadata:
        for path in figure_paths:
            metadata_path = path.with_name(path.stem + "_metadata.txt")
            if metadata_path.exists():
                continue
            metadata = (
                f"artifact_type: figure\n"
                f"filename: {path.name}\n"
                f"path: {path.relative_to(dirs.root)}\n"
                f"project_id: {options.project_id}\n"
                f"threshold: {options.threshold}\n"
                f"dpi: {options.dpi}\n"
                f"generated_time: {generated_time}\n"
                f"input_root: {_project_relative_path(output_root)}\n"
            )
            _write_sidecar_text(metadata_path, metadata, options.overwrite)
        for path in table_paths:
            record = table_records[path.stem]
            metadata = (
                f"artifact_type: table\n"
                f"table_id: {record['table_id']}\n"
                f"table_number: {record['table_number']}\n"
                f"table_type: {record['table_type']}\n"
                f"official: {record['official']}\n"
                f"filename: {path.name}\n"
                f"path: {path.relative_to(dirs.root)}\n"
                f"source_function: {record['source_function']}\n"
                f"used_by: {record['used_by']}\n"
                f"duplicate_of: {record['duplicate_of'] or 'NA'}\n"
                f"project_id: {options.project_id}\n"
                f"generated_time: {generated_time}\n"
                f"input_root: {_project_relative_path(output_root)}\n"
            )
            _write_sidecar_text(path.with_name(path.stem + "_metadata.txt"), metadata, options.overwrite)

    official_rows = main_rows + supplementary_rows
    for record in official_rows:
        csv_path = dirs.root / str(record["csv_path"])
        dictionary_path = dirs.root / str(record["dictionary_path"])
        _write_table_dictionary(
            dictionary_path,
            _read_csv(csv_path, f"Official table {record['table_number']}"),
            record,
            options.overwrite,
        )
    _write_table_index(dirs, official_rows, options)


def _write_figure_source(path: Path, data: pd.DataFrame, options: ReportOptions) -> None:
    """Write source CSV for a figure when requested."""

    if options.generate_source_data:
        _write_csv(
            _normalize_figure_source_frame(path, data),
            path.with_name(path.stem + "_source.csv"),
            options.overwrite,
        )


def _main_panel_specs() -> list[dict[str, str]]:
    """Return the planned main-panel figure manifest rows."""

    configured_rows: list[dict[str, str]] = []
    for figure in configured_figure_specs(PUBLICATION_CONFIG):
        if figure["role"] != "main":
            continue
        for panel in figure.get("panels", []):
            png_path = Path(figure["stem"]) / f"{panel['stem']}.png"
            configured_rows.append(
                {
                    "figure_number": figure["number"],
                    "panel_letter": panel["letter"],
                    "panel_title": panel["title"],
                    "filename_png": str(png_path),
                    "filename_pdf": str(png_path.with_suffix(".pdf")),
                    "filename_svg": str(png_path.with_suffix(".svg")),
                    "source_data": str(png_path.with_name(png_path.stem + "_source.csv")),
                    "caption_file": str(png_path.with_name(png_path.stem + "_caption.txt")),
                    "metadata_file": str(png_path.with_name(png_path.stem + "_metadata.txt")),
                    "recommended_usage": f"Main {figure['number'].replace('Figure ', 'Fig.')}",
                }
            )
    return configured_rows

    specs = [
        ("Figure 1", "a", "Bongpyeong bathymetry", "fig01/fig01_panel_a_bongpyeong_bathymetry.png", "Main Fig.1"),
        ("Figure 1", "b", "Bongpyeong 300 kHz backscatter", "fig01/fig01_panel_b_bongpyeong_300khz_backscatter_dn.png", "Main Fig.1"),
        ("Figure 1", "c", "Hujeong bathymetry", "fig01/fig01_panel_c_hujeong_bathymetry.png", "Main Fig.1"),
        ("Figure 1", "d", "Hujeong 300 kHz backscatter", "fig01/fig01_panel_d_hujeong_300khz_backscatter_dn.png", "Main Fig.1"),
        ("Figure 2", "a", "MBES data acquisition", "fig02/fig02_panel_a_mbes_data_acquisition.png", "Main Fig.2"),
        ("Figure 2", "b", "Preprocessing", "fig02/fig02_panel_b_preprocessing.png", "Main Fig.2"),
        ("Figure 2", "c", "Feature generation", "fig02/fig02_panel_c_feature_generation.png", "Main Fig.2"),
        ("Figure 2", "d", "Labeled samples", "fig02/fig02_panel_d_labeled_samples.png", "Main Fig.2"),
        ("Figure 2", "e", "Model × feature matrix", "fig02/fig02_panel_e_model_feature_matrix.png", "Main Fig.2"),
        ("Figure 2", "f", "Validation framework", "fig02/fig02_panel_f_validation_framework.png", "Main Fig.2"),
        ("Figure 2", "g", "Integrated candidate selection", "fig02/fig02_panel_g_integrated_candidate_selection.png", "Main Fig.2"),
        ("Figure 2", "h", "Final retraining and habitat mapping", "fig02/fig02_panel_h_final_retraining_habitat_mapping.png", "Main Fig.2"),
        ("Figure 3", "a", "Acoustic intensity", "fig03/fig03_panel_a_acoustic_intensity.png", "Main Fig.3"),
        ("Figure 3", "b", "Frequency-difference indices", "fig03/fig03_panel_b_frequency_difference_indices.png", "Main Fig.3"),
        ("Figure 3", "c", "Bathymetric features", "fig03/fig03_panel_c_bathymetric_features.png", "Main Fig.3"),
        ("Figure 3", "d", "Texture features", "fig03/fig03_panel_d_texture_features.png", "Main Fig.3"),
        ("Figure 4", "a", "Feature-set validation-specific AP", "fig04/fig04_panel_a_feature_set_validation_specific_ap.png", "Main Fig.4"),
        ("Figure 4", "b", "Feature-set within-site AP ranking", "fig04/fig04_panel_b_feature_set_within_site_ap.png", "Main Fig.4"),
        ("Figure 4", "c", "Feature-set cross-site AP ranking", "fig04/fig04_panel_c_feature_set_cross_site_ap.png", "Main Fig.4"),
        ("Figure 4", "d", "Feature-set spatial block CV AP ranking", "fig04/fig04_panel_d_feature_set_spatial_cv_ap.png", "Main Fig.4"),
        ("Figure 4", "e", "Feature-set Integrated AP ranking", "fig04/fig04_panel_e_feature_set_mean_ap_ranking.png", "Main Fig.4"),
        ("Figure 5", "a", "Model validation-specific AP", "fig05/fig05_panel_a_model_validation_specific_ap.png", "Main Fig.5"),
        ("Figure 5", "b", "Model within-site AP ranking", "fig05/fig05_panel_b_model_within_site_ap.png", "Main Fig.5"),
        ("Figure 5", "c", "Model cross-site AP ranking", "fig05/fig05_panel_c_model_cross_site_ap.png", "Main Fig.5"),
        ("Figure 5", "d", "Model spatial block CV AP ranking", "fig05/fig05_panel_d_model_spatial_cv_ap.png", "Main Fig.5"),
        ("Figure 5", "e", "Model Integrated AP ranking", "fig05/fig05_panel_e_model_mean_ap_ranking.png", "Main Fig.5"),
        ("Figure 6", "a", "Feature x model integrated performance", "fig06/fig06_panel_a_feature_model_integrated_performance_heatmap.png", "Main Fig.6"),
        ("Figure 7", "a", "Final candidate validation-specific AP", "fig07/fig07_panel_a_final_candidate_validation_specific_ap.png", "Main Fig.7"),
        ("Figure 7", "b", "Final candidate Integrated AP ranking", "fig07/fig07_panel_b_final_candidate_mean_ap_ranking.png", "Main Fig.7"),
        ("Figure 8", "a", "Feature importance", "fig08/fig08_panel_a_all_feature_importance.png", "Main Fig.8"),
        ("Figure 8", "b", "Feature group contribution", "fig08/fig08_panel_b_feature_group_contribution.png", "Main Fig.8"),
        ("Figure 9", "a", "Bongpyeong probability", "fig09/fig09_panel_a_bongpyeong_probability_map.png", "Main Fig.9"),
        ("Figure 9", "b", "Bongpyeong binary habitat", "fig09/fig09_panel_b_bongpyeong_binary_habitat_p05.png", "Main Fig.9"),
        ("Figure 9", "c", "Hujeong probability", "fig09/fig09_panel_c_hujeong_probability_map.png", "Main Fig.9"),
        ("Figure 9", "d", "Hujeong binary habitat", "fig09/fig09_panel_d_hujeong_binary_habitat_p05.png", "Main Fig.9"),
    ]
    rows = []
    for figure_number, panel, panel_title, filename_png, recommended in specs:
        old_path = Path(filename_png)
        parent = MAIN_PANEL_PARENT_DIRS[old_path.parts[0]]
        png_path = Path(parent) / old_path.name
        rows.append(
            {
                "figure_number": figure_number,
                "panel_letter": panel,
                "panel_title": panel_title,
                "filename_png": str(png_path),
                "filename_pdf": str(png_path.with_suffix(".pdf")),
                "filename_svg": str(png_path.with_suffix(".svg")),
                "source_data": str(png_path.with_name(png_path.stem + "_source.csv")),
                "caption_file": str(png_path.with_name(png_path.stem + "_caption.txt")),
                "metadata_file": str(png_path.with_name(png_path.stem + "_metadata.txt")),
                "recommended_usage": recommended,
            }
        )
    return rows


def _panel_path(dirs: ReportDirs, relative_png: str) -> Path:
    """Return the absolute path for one main panel relative filename."""

    old_path = Path(relative_png)
    parent = MAIN_PANEL_PARENT_DIRS.get(old_path.parts[0], old_path.parts[0])
    return dirs.main_panels / parent / old_path.name


def _panel_title(letter: str, title: str, options: ReportOptions) -> str:
    """Return a panel title with optional panel letter."""

    return f"({letter}) {title}" if options.panel_label else title


def _panel_canvas(kind: str) -> PanelCanvas:
    """Return the fixed canvas spec for a panel kind."""

    spec = PANEL_CANVAS[kind]
    return PanelCanvas(kind=kind, figsize=spec["figsize"], margins=spec["margins"])


def _new_panel_figure(kind: str) -> tuple[Any, Any]:
    """Create one fixed-size panel figure and axis."""

    plt = _require_matplotlib_pyplot()
    canvas = _panel_canvas(kind)
    fig, ax = plt.subplots(figsize=canvas.figsize, constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(**dict(canvas.margins))
    return fig, ax


def _style_panel_axis(ax: Any) -> None:
    """Apply common axis and tick font sizes to one panel."""

    ax.xaxis.label.set_size(PANEL_STYLE["axis"])
    ax.yaxis.label.set_size(PANEL_STYLE["axis"])
    ax.tick_params(labelsize=PANEL_STYLE["tick"])


def _bounds_tuple(bounds: Any) -> tuple[float, float, float, float]:
    """Return bounds as left, right, bottom, top."""

    return (float(bounds.left), float(bounds.right), float(bounds.bottom), float(bounds.top))


def _union_bounds(bounds_values: Sequence[Any]) -> tuple[float, float, float, float]:
    """Return the union bounds from rasterio bounds-like objects."""

    tuples = [_bounds_tuple(bounds) for bounds in bounds_values]
    left = min(item[0] for item in tuples)
    right = max(item[1] for item in tuples)
    bottom = min(item[2] for item in tuples)
    top = max(item[3] for item in tuples)
    return (left, right, bottom, top)


def _map_axis_ticks(bounds: tuple[float, float, float, float], count: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed map ticks for a shared extent."""

    left, right, bottom, top = bounds
    return np.linspace(left, right, count), np.linspace(bottom, top, count)


def _apply_shared_map_extent(ax: Any, bounds: tuple[float, float, float, float]) -> None:
    """Apply common extent, ticks, labels, and map panel styling."""

    left, right, bottom, top = bounds
    xticks, yticks = _map_axis_ticks(bounds)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel("Easting (m, UTM Zone 52N)", fontsize=max(PANEL_STYLE["axis"] - 2, 9))
    ax.set_ylabel("Northing (m, UTM Zone 52N)", fontsize=max(PANEL_STYLE["axis"] - 2, 9))
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=max(PANEL_STYLE["tick"] - 2, 8), rotation=0)
    xlabels = ax.get_xticklabels()
    if len(xlabels) >= 2:
        xlabels[0].set_ha("left")
        xlabels[-1].set_ha("right")


def _save_panel_outputs(fig: Any, path: Path, options: ReportOptions) -> None:
    """Save one main-panel PNG and PDF for Illustrator assembly."""

    if path.exists() and not options.overwrite:
        print(f"Skipped existing panel due to --no-overwrite: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PANEL_DPI, facecolor="white", transparent=False)
    if options.save_vector:
        fig.savefig(path.with_suffix(".pdf"), facecolor="white", transparent=False)
        fig.savefig(path.with_suffix(".svg"), facecolor="white", transparent=False)


def _prepare_metric_panel_data(
    df: pd.DataFrame,
    *,
    category: str,
    order: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare validation-specific and Integrated AP columns for panel outputs."""

    data = df.copy()
    data = data[data[category].astype(str).isin([str(value) for value in order])].copy()
    data[category] = pd.Categorical(data[category].astype(str), categories=list(order), ordered=True)
    data = data.sort_values(category)
    metrics = [column for column in ["within_AP", "cross_AP", "spatial_AP"] if column in data.columns]
    if "mean_AP" not in data.columns and metrics:
        data["mean_AP"] = data[metrics].mean(axis=1)
    return data, metrics


def _save_validation_specific_panel(
    data: pd.DataFrame,
    *,
    category: str,
    metrics: Sequence[str],
    path: Path,
    title: str,
    options: ReportOptions,
) -> None:
    """Save one grouped validation-specific AP panel."""

    plt = _require_matplotlib_pyplot()
    if data.empty or not metrics:
        _save_text_figure(path, title, ["No validation-specific AP data available."], options.dpi, options.overwrite)
        return
    labels = [_short_label(_display_label(value), 18) for value in data[category].astype(str)]
    width = 0.72 / max(len(metrics), 1)
    colors = {"within_AP": "#4f83b8", "cross_AP": "#c96b4a", "spatial_AP": "#5a9f68"}
    fig, ax = _new_panel_figure("landscape_wide_left" if category == "feature_set" else "landscape")
    all_values: list[float] = []
    if category == "feature_set":
        y = np.arange(len(labels))
        for idx, metric in enumerate(metrics):
            values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype="float64")
            all_values.extend([float(value) for value in values if np.isfinite(value)])
            offset = (idx - (len(metrics) - 1) / 2) * width
            ax.barh(
                y + offset,
                values,
                height=width,
                color=colors.get(metric, None),
                label=_display_column_name(metric),
            )
            for yi, value in zip(y + offset, values):
                if np.isfinite(value):
                    ax.text(value, yi, f" {value:.3f}", va="center", ha="left", fontsize=PANEL_STYLE["value"])
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Average precision (AP)", fontsize=PANEL_STYLE["axis"])
        ax.set_ylabel("Feature set", fontsize=PANEL_STYLE["axis"])
        ax.set_xlim(0, max(1.02, max(all_values) * 1.16 if all_values else 1.02))
        ax.grid(axis="x", alpha=0.25)
    else:
        x = np.arange(len(labels))
        for idx, metric in enumerate(metrics):
            values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype="float64")
            all_values.extend([float(value) for value in values if np.isfinite(value)])
            ax.bar(
                x + (idx - (len(metrics) - 1) / 2) * width,
                values,
                width=width,
                color=colors.get(metric, None),
                label=_display_column_name(metric),
            )
        ax.set_ylabel("Average precision (AP)", fontsize=PANEL_STYLE["axis"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_ylim(0, max(1.0, max(all_values) * 1.10 if all_values else 1.0))
        ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=PANEL_STYLE["tick"])
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=min(len(metrics), 3),
        fontsize=max(PANEL_STYLE["legend"] - 1, 8),
        handlelength=1.2,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    _save_panel_outputs(fig, path, options)
    plt.close(fig)
    _write_figure_source(path, data, options)


def _save_metric_ranking_panel(
    data: pd.DataFrame,
    *,
    category: str,
    metric: str,
    path: Path,
    title: str,
    color: str,
    options: ReportOptions,
) -> None:
    """Save one horizontal AP ranking panel."""

    plt = _require_matplotlib_pyplot()
    if data.empty or metric not in data.columns:
        _save_text_figure(path, title, [f"No {metric} data available."], options.dpi, options.overwrite)
        return
    rank_df = data.copy()
    rank_df[metric] = pd.to_numeric(rank_df[metric], errors="coerce")
    rank_df = rank_df.sort_values(metric, ascending=True)
    values = rank_df[metric].to_numpy(dtype="float64")
    y = np.arange(rank_df.shape[0])
    fig, ax = _new_panel_figure("landscape_wide_left")
    ax.barh(y, values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(_display_label(value), 26) for value in rank_df[category].astype(str)], fontsize=PANEL_STYLE["tick"])
    ax.set_xlabel("Average precision (AP)", fontsize=PANEL_STYLE["axis"])
    ax.set_title("Validation-specific AP", fontsize=PANEL_STYLE["title"])
    ax.set_ylabel("Feature set" if category == "feature_set" else "Model", fontsize=PANEL_STYLE["axis"])
    ax.set_xlim(*_nice_ylim(values, lower_floor=0.0))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(axis="x", alpha=0.25)
    for yi, value in zip(y, values):
        if np.isfinite(value):
            ax.text(value, yi, f" {value:.3f}", va="center", ha="left", fontsize=PANEL_STYLE["value"])
    _save_panel_outputs(fig, path, options)
    plt.close(fig)
    _write_figure_source(path, rank_df, options)


def _save_metric_main_panels(
    df: pd.DataFrame,
    *,
    category: str,
    order: Sequence[str],
    prefix: str,
    titles: Mapping[str, str],
    dirs: ReportDirs,
    options: ReportOptions,
) -> None:
    """Save Fig.4/Fig.5 panel files."""

    data, metrics = _prepare_metric_panel_data(df, category=category, order=order)
    colors = {"within_AP": "#4f83b8", "cross_AP": "#c96b4a", "spatial_AP": "#5a9f68", "mean_AP": "#374151"}
    _save_validation_specific_panel(
        data,
        category=category,
        metrics=metrics,
        path=_panel_path(dirs, titles["validation_file"]),
        title=_panel_title("a", titles["validation_title"], options),
        options=options,
    )
    for metric, file_key, title_key, panel_letter in [
        ("within_AP", "within_file", "within_title", "b"),
        ("cross_AP", "cross_file", "cross_title", "c"),
        ("spatial_AP", "spatial_file", "spatial_title", "d"),
        ("mean_AP", "mean_file", "mean_title", "e"),
    ]:
        _save_metric_ranking_panel(
            data,
            category=category,
            metric=metric,
            path=_panel_path(dirs, titles[file_key]),
            title=_panel_title(panel_letter, titles[title_key], options),
            color=colors[metric],
            options=options,
        )


def _save_study_area_main_panels(
    *,
    output_root: Path,
    dirs: ReportDirs,
    options: ReportOptions,
    warnings: list[str],
) -> None:
    """Save Fig.1 as four independent raster panels."""

    plt = _require_matplotlib_pyplot()
    panel_specs = _publication_panels("fig_01_study_areas")
    site_bounds: dict[str, tuple[float, float, float, float]] = {}
    sources: dict[str, tuple[Path, Path | None]] = {
        str(panel["id"]): _study_area_raster_sources(output_root, panel["selection"])
        for panel in panel_specs
    }
    for site_id in SITE_ORDER:
        paths = [
            sources[str(panel["id"])][0]
            for panel in panel_specs
            if str(panel["selection"]["site"]) == site_id
        ]
        existing_bounds = [_raster_profile_record(path)["bounds"] for path in paths if path.exists()]
        if existing_bounds:
            site_bounds[site_id] = _union_bounds(existing_bounds)
    for panel in panel_specs:
        selection = panel["selection"]
        letter = str(panel["letter"])
        title = str(selection.get("render_title", panel["title"]))
        site_id = str(selection["site"])
        layer = str(selection["layer"])
        cmap_name = str(selection["cmap"])
        colorbar_label = str(selection["colorbar_label"])
        file_out = f"fig01/{panel['stem']}.png"
        raster_path, mask_path = sources[str(panel["id"])]
        if not raster_path.exists():
            message = f"Main panel skipped: required Fig.1 raster missing: {raster_path}"
            warnings.append(message)
            print(f"Warning: {message}", flush=True)
            continue
        if mask_path is not None and not mask_path.exists():
            message = f"Main panel skipped: required Fig.1 common-valid mask missing: {mask_path}"
            warnings.append(message)
            print(f"Warning: {message}", flush=True)
            continue
        data, bounds = _read_study_area_quicklook(
            raster_path,
            mask_path,
            options.max_preview_size,
            apply_scale_offset=(layer == "sonar"),
        )
        stats = _raster_value_summary(raster_path, mask_path)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#d9d9d9")
        fig, ax = _new_panel_figure("map")
        image = ax.imshow(
            data,
            cmap=cmap,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            origin="upper",
        )
        ax.set_aspect("equal")
        _apply_shared_map_extent(ax, site_bounds.get(site_id, _bounds_tuple(bounds)))
        add_map_cartography(ax, bounds)
        cbar = fig.colorbar(image, ax=ax, fraction=0.060, pad=0.055)
        cbar.set_label(colorbar_label, fontsize=max(PANEL_STYLE["cbar_label"] - 1, 9))
        cbar.ax.tick_params(labelsize=max(PANEL_STYLE["cbar_tick"] - 2, 8), pad=4)
        path = _panel_path(dirs, file_out)
        _save_panel_outputs(fig, path, options)
        plt.close(fig)
        _write_figure_source(
            path,
            pd.DataFrame(
                [
                    {
                        "site_id": site_id,
                        "site_label": _display_label(site_id),
                        "source_layer": layer,
                        "raster_path": str(raster_path),
                        "valid_mask_path": str(mask_path) if mask_path is not None else "",
                        "source_domain": str(selection.get("source_domain", "project_input")),
                        "dtype": stats["dtype"],
                        "nodata": stats["nodata"],
                        "min": stats["min"],
                        "max": stats["max"],
                        "scale": stats["scale"],
                        "offset": stats["offset"],
                        "crs": stats["crs"],
                        "status": "used",
                    }
                ]
            ),
            options,
        )


def _save_workflow_main_panels(
    *,
    dirs: ReportDirs,
    options: ReportOptions,
) -> None:
    """Save Fig.2 workflow boxes as independent panels."""

    plt = _require_matplotlib_pyplot()
    panels = [
        ("a", "MBES data acquisition", "200 / 300 / 400 kHz\nbathymetry + backscatter", "#dbeafe", "fig02/fig02_panel_a_mbes_data_acquisition.png"),
        ("b", "Preprocessing", "co-registration / 0.07 m resampling\ncommon area", "#e0f2fe", "fig02/fig02_panel_b_preprocessing.png"),
        ("c", "Feature generation", "acoustic 3 + NDI 2\nbathymetric 2 + texture 3", "#dcfce7", "fig02/fig02_panel_c_feature_generation.png"),
        ("d", "Labeled samples", "vegetated / non-vegetated", "#fef9c3", "fig02/fig02_panel_d_labeled_samples.png"),
        ("e", "Model × feature matrix", "7 feature sets × 4 ML models", "#fef3c7", "fig02/fig02_panel_e_model_feature_matrix.png"),
        ("f", "Validation framework", "within-site / cross-site\nspatial block CV", "#fee2e2", "fig02/fig02_panel_f_validation_framework.png"),
        ("g", "Integrated candidate selection", "validation synthesis", "#ede9fe", "fig02/fig02_panel_g_integrated_candidate_selection.png"),
        ("h", "Final retraining and habitat mapping", "site-specific final maps", "#e5e7eb", "fig02/fig02_panel_h_final_retraining_habitat_mapping.png"),
    ]
    for letter, title, body, color, relative_path in panels:
        fig, ax = _new_panel_figure("box")
        ax.set_axis_off()
        rect = plt.Rectangle((0.08, 0.22), 0.84, 0.54, facecolor=color, edgecolor="#374151", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(0.50, 0.58, title, ha="center", va="center", fontsize=PANEL_STYLE["title"], fontweight="bold")
        ax.text(0.50, 0.39, body, ha="center", va="center", fontsize=PANEL_STYLE["axis"], linespacing=1.25)
        path = _panel_path(dirs, relative_path)
        _save_panel_outputs(fig, path, options)
        plt.close(fig)
        _write_figure_source(
            path,
            pd.DataFrame([{"panel_letter": letter, "title": title, "text": body}]),
            options,
        )


def _save_feature_definition_main_panels(
    *,
    dirs: ReportDirs,
    options: ReportOptions,
) -> None:
    """Save Fig.3 feature-group boxes as independent panels."""

    plt = _require_matplotlib_pyplot()
    panels = [
        (
            "a",
            "Acoustic intensity",
            ["a200_norm", "a300_norm", "a400_norm"],
            "x_norm = (x - depth-bin median) / max(P90 - P10, epsilon)",
            "fig03/fig03_panel_a_acoustic_intensity.png",
            "#dcfce7",
        ),
        (
            "b",
            "Frequency-difference indices",
            ["Normalized Difference Index", "• 400–200", "• 400–300"],
            "NDI = (A - B) / (|A| + |B|)",
            "fig03/fig03_panel_b_frequency_difference_indices.png",
            "#ede9fe",
        ),
        (
            "c",
            "Bathymetric features",
            ["dpth_lite_300", "dpth_rel_300"],
            "dpth_rel = (z - median_local) / max(IQR_local, epsilon)",
            "fig03/fig03_panel_c_bathymetric_features.png",
            "#dbeafe",
        ),
        (
            "d",
            "Texture features",
            ["a300_low", "a300_high", "a300_log1p"],
            "low-pass / high-pass / log(1 + |LoG|)",
            "fig03/fig03_panel_d_texture_features.png",
            "#fef3c7",
        ),
    ]
    for letter, title, features, formula, relative_path, color in panels:
        fig, ax = _new_panel_figure("box")
        ax.set_axis_off()
        rect = plt.Rectangle((0.07, 0.14), 0.86, 0.68, facecolor=color, edgecolor="#374151", linewidth=1.3)
        ax.add_patch(rect)
        ax.text(0.50, 0.70, title, ha="center", va="center", fontsize=PANEL_STYLE["title"], fontweight="bold")
        feature_text = "\n".join(_display_label(feature) if feature in DISPLAY_LABELS else feature for feature in features)
        ax.text(0.50, 0.49, feature_text, ha="center", va="center", fontsize=PANEL_STYLE["axis"], linespacing=1.25)
        ax.text(0.50, 0.25, formula, ha="center", va="center", fontsize=max(9, PANEL_STYLE["tick"] - 1), linespacing=1.2)
        path = _panel_path(dirs, relative_path)
        _save_panel_outputs(fig, path, options)
        plt.close(fig)
        _write_figure_source(
            path,
            pd.DataFrame(
                [{"panel_letter": letter, "title": title, "features": "; ".join(features), "formula": formula}]
            ),
            options,
        )


def _save_feature_model_heatmap_panel(
    df: pd.DataFrame,
    *,
    dirs: ReportDirs,
    options: ReportOptions,
) -> None:
    """Save Fig.6 heatmap as a single independent panel."""

    plt = _require_matplotlib_pyplot()
    path = _panel_path(dirs, "fig06/fig06_panel_a_feature_model_integrated_performance_heatmap.png")
    if path.exists() and not options.overwrite:
        print(f"Skipped existing panel due to --no-overwrite: {path}", flush=True)
        return
    source = df.copy()
    pivot = source.pivot_table(index="feature_set", columns="model_id", values="mean_AP", aggfunc="mean")
    pivot = pivot.reindex([item for item in FEATURE_ORDER if item in pivot.index])
    pivot = pivot.reindex(columns=[item for item in MODEL_ORDER if item in pivot.columns])
    values = pivot.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if math.isclose(vmin, vmax):
        vmin -= 0.01
        vmax += 0.01
    fig, ax = _new_panel_figure("landscape_wide_left")
    image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    heatmap_xlabels = [
        "Random\nForest" if str(item) == "random_forest" else _short_label(_display_label(item), 16)
        for item in pivot.columns
    ]
    ax.set_xticklabels(heatmap_xlabels, rotation=0, ha="center", fontsize=max(PANEL_STYLE["tick"] - 1, 8))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([_short_label(_display_label(item), 24) for item in pivot.index], fontsize=PANEL_STYLE["tick"])
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.iloc[y, x]
            if pd.notna(val):
                norm = (float(val) - vmin) / (vmax - vmin)
                ax.text(x, y, f"{float(val):.3f}", ha="center", va="center", fontsize=PANEL_STYLE["value"], color="black" if norm > 0.62 else "white")
    cbar = fig.colorbar(image, ax=ax, fraction=0.065, pad=0.055)
    cbar.set_label("Integrated AP", fontsize=PANEL_STYLE["cbar_label"])
    cbar.ax.tick_params(labelsize=PANEL_STYLE["cbar_tick"])
    _save_panel_outputs(fig, path, options)
    plt.close(fig)
    _write_figure_source(path, source, options)


def _prepare_candidate_panel_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare final-candidate comparison data."""

    data = df.copy()
    data = data[data["candidate_label"].astype(str).isin(CANDIDATE_ORDER)].copy()
    data["candidate_label"] = pd.Categorical(data["candidate_label"].astype(str), categories=CANDIDATE_ORDER, ordered=True)
    data = data.sort_values("candidate_label")
    metrics = [column for column in ["within_AP", "cross_AP", "spatial_AP"] if column in data.columns]
    if "mean_AP" not in data.columns and metrics:
        data["mean_AP"] = data[metrics].mean(axis=1)
    return data, metrics


def _candidate_axis_labels(data: pd.DataFrame) -> list[str]:
    """Return compact final-candidate labels with feature/model context."""

    return [
        f"{CANDIDATE_COMPACT_LABELS.get(str(row['candidate_label']), _display_label(str(row['candidate_label'])))}\n"
        f"{_display_label(str(row.get('feature_set', '')))} / "
        f"{MODEL_COMPACT_LABELS.get(str(row.get('model_id', '')), _display_label(str(row.get('model_id', ''))))}"
        for _, row in data.iterrows()
    ]


def _save_candidate_main_panels(
    df: pd.DataFrame,
    *,
    dirs: ReportDirs,
    options: ReportOptions,
) -> None:
    """Save Fig.7 as independent panel files."""

    plt = _require_matplotlib_pyplot()
    data, metrics = _prepare_candidate_panel_data(df)
    labels = _candidate_axis_labels(data)
    colors = {"within_AP": "#4f83b8", "cross_AP": "#c96b4a", "spatial_AP": "#5a9f68"}
    path_a = _panel_path(dirs, "fig07/fig07_panel_a_final_candidate_validation_specific_ap.png")
    fig, ax = _new_panel_figure("landscape_wide_left")
    y = np.arange(len(labels))
    height = 0.72 / max(len(metrics), 1)
    all_values: list[float] = []
    for idx, metric in enumerate(metrics):
        values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype="float64")
        all_values.extend([float(value) for value in values if np.isfinite(value)])
        ax.barh(
            y + (idx - (len(metrics) - 1) / 2) * height,
            values,
            height=height,
            color=colors.get(metric, None),
            label=_display_column_name(metric),
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=PANEL_STYLE["tick"])
    ax.invert_yaxis()
    ax.set_xlabel("Average precision (AP)", fontsize=PANEL_STYLE["axis"])
    ax.set_title("Validation-specific AP", fontsize=PANEL_STYLE["title"])
    ax.set_xlim(0, max(1.0, max(all_values) * 1.10 if all_values else 1.0))
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=PANEL_STYLE["tick"])
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(metrics), 3),
        fontsize=max(PANEL_STYLE["legend"] - 1, 8),
        handlelength=1.2,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    _save_panel_outputs(fig, path_a, options)
    plt.close(fig)
    _write_figure_source(path_a, data, options)

    path_b = _panel_path(dirs, "fig07/fig07_panel_b_final_candidate_mean_ap_ranking.png")
    rank_df = data.sort_values("mean_AP", ascending=True)
    rank_values = pd.to_numeric(rank_df["mean_AP"], errors="coerce").to_numpy(dtype="float64")
    rank_labels = _candidate_axis_labels(rank_df)
    y2 = np.arange(rank_df.shape[0])
    fig, ax = _new_panel_figure("landscape_wide_left")
    ax.barh(y2, rank_values, color="#374151")
    ax.set_yticks(y2)
    ax.set_yticklabels(rank_labels, fontsize=PANEL_STYLE["tick"])
    ax.set_xlabel("Integrated AP", fontsize=PANEL_STYLE["axis"])
    ax.set_title("(b) Integrated AP ranking", fontsize=PANEL_STYLE["title"])
    ax.set_xlim(*_nice_ylim(rank_values, lower_floor=0.0))
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(axis="x", alpha=0.25)
    for yi, value in zip(y2, rank_values):
        if np.isfinite(value):
            ax.text(value, yi, f" {value:.3f}", va="center", ha="left", fontsize=PANEL_STYLE["value"])
    _save_panel_outputs(fig, path_b, options)
    plt.close(fig)
    _write_figure_source(path_b, rank_df, options)


def _normalized_final_model_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
    """Return within-model importance shares for cross-model publication summaries."""

    data = feature_importance.copy()
    data["importance"] = pd.to_numeric(data["importance"], errors="coerce")
    data = data[np.isfinite(data["importance"])].copy()
    model_columns = ["site_id", "candidate_label"]
    missing_model_columns = [column for column in model_columns if column not in data.columns]
    if missing_model_columns:
        raise ValueError(
            "Fig.8 group importance requires final-model identity columns: "
            + ", ".join(missing_model_columns)
        )
    model_totals = data.groupby(model_columns, dropna=False)["importance"].transform("sum")
    if (model_totals <= 0.0).any():
        raise ValueError("Fig.8 feature importance requires a positive total for every final model.")
    # Model families expose different native scales (for example, RF sums to 1
    # while CatBoost sums to 100). Convert each model to within-model shares
    # before any cross-model summary while preserving raw Table S8 records.
    data["importance"] = data["importance"] / model_totals
    return data


def _feature_importance_summaries(
    feature_importance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return feature means and equally weighted model-level group means for Fig.8."""

    data = feature_importance.copy()
    data["candidate_label"] = data["candidate_label"].astype(str)
    data = data[data["candidate_label"].isin(CANDIDATE_ORDER)].copy()
    data = _normalized_final_model_importance(data)
    model_columns = ["site_id", "candidate_label"]
    summary = (
        data.groupby("feature", dropna=False)["importance"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_importance",
                "std": "sd_importance",
                "min": "min_importance",
                "max": "max_importance",
                "count": "n",
            }
        )
    )
    summary["feature_group"] = summary["feature"].map(_feature_group)
    summary["sd_importance"] = summary["sd_importance"].fillna(0.0)
    data["feature_group"] = data["feature"].map(_feature_group)
    model_group_importance = (
        data.groupby(model_columns + ["feature_group"], dropna=False)["importance"]
        .sum()
        .unstack("feature_group", fill_value=0.0)
        .reindex(columns=FEATURE_GROUP_ORDER, fill_value=0.0)
    )
    group_summary = (
        model_group_importance.mean(axis=0)
        .reindex(FEATURE_GROUP_ORDER)
        .dropna()
        .rename("mean_group_importance")
        .reset_index()
    )
    group_summary["model_count"] = int(model_group_importance.shape[0])
    return summary, group_summary


def _save_feature_importance_main_panels(
    feature_importance: pd.DataFrame,
    *,
    dirs: ReportDirs,
    options: ReportOptions,
    warnings: list[str],
) -> None:
    """Save Fig.8 as independent panel files."""

    required = {"candidate_label", "feature", "importance"}
    if feature_importance.empty or not required.issubset(feature_importance.columns):
        message = "Fig.8 panels skipped: final candidate feature-importance inputs are incomplete."
        warnings.append(message)
        print(f"Warning: {message}", flush=True)
        return
    summary, group_summary = _feature_importance_summaries(feature_importance)
    if summary.empty:
        message = "Fig.8 panels skipped: no finite final-candidate feature-importance values."
        warnings.append(message)
        print(f"Warning: {message}", flush=True)
        return
    plt = _require_matplotlib_pyplot()
    from matplotlib.patches import Patch

    feature_plot = summary.sort_values("mean_importance", ascending=True).copy()
    fig, ax = _new_panel_figure("landscape_wide_left")
    y = np.arange(feature_plot.shape[0])
    colors = [FEATURE_GROUP_COLORS.get(str(group), FEATURE_GROUP_COLORS["Other"]) for group in feature_plot["feature_group"]]
    ax.barh(y, feature_plot["mean_importance"].to_numpy(dtype="float64"), color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(_display_label(feature), 30) for feature in feature_plot["feature"].astype(str)], fontsize=PANEL_STYLE["tick"])
    ax.set_xlabel("Mean feature importance", fontsize=PANEL_STYLE["axis"])
    ax.grid(axis="x", alpha=0.25)
    values = feature_plot["mean_importance"].to_numpy(dtype="float64")
    for yi, value in zip(y, values):
        if np.isfinite(value):
            ax.text(value, yi, f" {value:.3f}", va="center", ha="left", fontsize=PANEL_STYLE["value"])
    handles = [
        Patch(facecolor=FEATURE_GROUP_COLORS[group], edgecolor="white", label=group)
        for group in FEATURE_GROUP_ORDER[:-1]
        if group in set(summary["feature_group"].astype(str))
    ]
    ax.tick_params(labelsize=PANEL_STYLE["tick"])
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=PANEL_STYLE["legend"])
    path_a = _panel_path(dirs, "fig08/fig08_panel_a_all_feature_importance.png")
    _save_panel_outputs(fig, path_a, options)
    plt.close(fig)
    _write_figure_source(path_a, feature_plot, options)

    group_plot = group_summary[group_summary["feature_group"].astype(str).isin(FEATURE_GROUP_ORDER[:-1])].copy()
    fig, ax = _new_panel_figure("landscape_wide_left")
    y2 = np.arange(group_plot.shape[0])
    ax.barh(
        y2,
        group_plot["mean_group_importance"].to_numpy(dtype="float64"),
        color=[FEATURE_GROUP_COLORS.get(str(group), FEATURE_GROUP_COLORS["Other"]) for group in group_plot["feature_group"].astype(str)],
    )
    ax.set_yticks(y2)
    ax.set_yticklabels(group_plot["feature_group"].astype(str))
    ax.set_xlabel("Mean group importance", fontsize=PANEL_STYLE["axis"])
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    ax.tick_params(labelsize=PANEL_STYLE["tick"])
    path_b = _panel_path(dirs, "fig08/fig08_panel_b_feature_group_contribution.png")
    _save_panel_outputs(fig, path_b, options)
    plt.close(fig)
    _write_figure_source(path_b, group_plot, options)


def _save_primary_habitat_map_main_panels(
    train_summary: pd.DataFrame,
    *,
    dirs: ReportDirs,
    options: ReportOptions,
    warnings: list[str],
) -> None:
    """Save Fig.9 as independent map panels."""

    records = [
        record
        for record in _map_record_iter(train_summary)
        if str(record["candidate_label"]) == PRIMARY_PUBLICATION_CANDIDATE
        and str(record["site_id"]) in SITE_ORDER
    ]
    by_site = {str(record["site_id"]): record for record in records}
    missing_paths: list[str] = []
    for site_id in SITE_ORDER:
        if site_id not in by_site:
            missing_paths.append(
                f"final_maps/{site_id}/{PRIMARY_PUBLICATION_CANDIDATE}/train_summary.csv record"
            )
            continue
        for key in ["probability_src", "valid_mask_src"]:
            if not Path(str(by_site[site_id][key])).exists():
                missing_paths.append(str(by_site[site_id][key]))
    if missing_paths:
        message = (
            f"Fig.9 panels skipped: missing {PRIMARY_PUBLICATION_CANDIDATE} map inputs: "
            f"{'; '.join(missing_paths)}"
        )
        warnings.append(message)
        print(f"Warning: {message}", flush=True)
        return

    plt = _require_matplotlib_pyplot()
    from matplotlib.colors import BoundaryNorm, ListedColormap

    threshold = 0.5
    cmap_prob = plt.get_cmap(options.map_cmap).copy()
    cmap_prob.set_bad("#d9d9d9")
    cmap_bin = ListedColormap(["white", "#064e3b"])
    cmap_bin.set_bad("#d9d9d9")
    binary_norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap_bin.N)
    panel_specs = _publication_panels("fig_09_predicted_habitat_maps")
    quicklooks: dict[str, tuple[np.ma.MaskedArray, np.ma.MaskedArray, Any]] = {}
    for site_id in SITE_ORDER:
        record = by_site[site_id]
        quicklooks[site_id] = _read_probability_binary_quicklook(
            Path(str(record["probability_src"])),
            Path(str(record["valid_mask_src"])),
            options.max_preview_size,
            threshold,
        )
    for panel in panel_specs:
        selection = panel["selection"]
        letter = str(panel["letter"])
        site_id = str(selection["site"])
        kind = str(selection["map_kind"])
        title = str(selection.get("render_title", panel["title"]))
        file_out = f"fig09/{panel['stem']}.png"
        probability, binary, bounds = quicklooks[site_id]
        data = probability if kind == "probability" else binary
        fig, ax = _new_panel_figure("map")
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        if kind == "probability":
            image = ax.imshow(data, cmap=cmap_prob, vmin=0, vmax=1, extent=extent, origin="upper")
            cbar = fig.colorbar(image, ax=ax, fraction=0.060, pad=0.055, ticks=np.linspace(0, 1, 6))
            cbar.set_label("Predicted habitat probability", fontsize=PANEL_STYLE["cbar_label"])
            title = f"{_display_label(site_id)} predicted habitat probability"
        else:
            image = ax.imshow(data, cmap=cmap_bin, norm=binary_norm, extent=extent, origin="upper")
            cbar = fig.colorbar(image, ax=ax, fraction=0.060, pad=0.055, ticks=[0, 1])
            cbar.ax.set_yticklabels(["P < 0.5", "P ≥ 0.5"])
            cbar.set_label("Thresholded habitat prediction", fontsize=PANEL_STYLE["cbar_label"])
            title = f"{_display_label(site_id)} predicted habitat (P ≥ 0.5)"
        cbar.ax.tick_params(labelsize=max(PANEL_STYLE["cbar_tick"] - 2, 8), pad=4)
        ax.set_title(f"({letter}) {title}", fontsize=PANEL_STYLE["title"])
        ax.set_aspect("equal")
        _apply_shared_map_extent(ax, _bounds_tuple(bounds))
        add_map_cartography(ax, bounds)
        path = _panel_path(dirs, file_out)
        _save_panel_outputs(fig, path, options)
        plt.close(fig)
        source = pd.DataFrame([by_site[site_id]])
        source["panel_kind"] = kind
        source["threshold"] = threshold
        _write_figure_source(path, source, options)


def _write_main_panels(
    *,
    dirs: ReportDirs,
    options: ReportOptions,
    output_root: Path,
    analysis: Mapping[str, pd.DataFrame],
    train_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Generate individual main-figure panels under figures/main_panels."""

    print("Writing main figure panels", flush=True)
    main_specs = [
        item for item in configured_figure_specs(PUBLICATION_CONFIG)
        if item["role"] == "main"
    ]
    for spec in main_specs:
        builder = spec["builder"]
        if not options.main_panels_only and builder in {
            "study_areas",
            "workflow_validation",
            "mbes_feature_definitions",
            "feature_set_comparison",
            "model_comparison",
            "feature_model_heatmap",
            "final_candidate_comparison",
            "main_feature_importance",
            "predicted_habitat_maps",
        }:
            expected = [
                dirs.main_panels / str(spec["stem"]) / f"{panel['stem']}.png"
                for panel in spec.get("panels", [])
            ]
            if expected and all(path.exists() for path in expected):
                continue
        if builder == "study_areas":
            _save_study_area_main_panels(
                output_root=output_root, dirs=dirs, options=options, warnings=warnings
            )
        elif builder == "workflow_validation":
            _save_workflow_main_panels(dirs=dirs, options=options)
        elif builder == "mbes_feature_definitions":
            _save_feature_definition_main_panels(dirs=dirs, options=options)
        elif builder == "feature_set_comparison":
            _save_metric_main_panels(
                analysis["stage2_integrated_feature_stats.csv"],
                category="feature_set", order=FEATURE_ORDER, prefix="fig04",
                titles={
                    "validation_file": "fig04/fig04_panel_a_feature_set_validation_specific_ap.png",
                    "validation_title": "Feature-set validation-specific AP",
                    "within_file": "fig04/fig04_panel_b_feature_set_within_site_ap.png",
                    "within_title": "Feature-set within-site AP ranking",
                    "cross_file": "fig04/fig04_panel_c_feature_set_cross_site_ap.png",
                    "cross_title": "Feature-set cross-site AP ranking",
                    "spatial_file": "fig04/fig04_panel_d_feature_set_spatial_cv_ap.png",
                    "spatial_title": "Feature-set spatial block CV AP ranking",
                    "mean_file": "fig04/fig04_panel_e_feature_set_mean_ap_ranking.png",
                    "mean_title": "Feature-set Integrated AP ranking",
                },
                dirs=dirs, options=options,
            )
        elif builder == "model_comparison":
            _save_metric_main_panels(
                analysis["stage2_integrated_model_stats.csv"],
                category="model_id", order=MODEL_ORDER, prefix="fig05",
                titles={
                    "validation_file": "fig05/fig05_panel_a_model_validation_specific_ap.png",
                    "validation_title": "Model validation-specific AP",
                    "within_file": "fig05/fig05_panel_b_model_within_site_ap.png",
                    "within_title": "Model within-site AP ranking",
                    "cross_file": "fig05/fig05_panel_c_model_cross_site_ap.png",
                    "cross_title": "Model cross-site AP ranking",
                    "spatial_file": "fig05/fig05_panel_d_model_spatial_cv_ap.png",
                    "spatial_title": "Model spatial block CV AP ranking",
                    "mean_file": "fig05/fig05_panel_e_model_mean_ap_ranking.png",
                    "mean_title": "Model Integrated AP ranking",
                },
                dirs=dirs, options=options,
            )
        elif builder == "feature_model_heatmap":
            _save_feature_model_heatmap_panel(
                analysis["stage2_integrated_feature_model_stats.csv"], dirs=dirs, options=options
            )
        elif builder == "final_candidate_comparison":
            _save_candidate_main_panels(
                analysis["stage2_final_candidate_comparison.csv"], dirs=dirs, options=options
            )
        elif builder == "main_feature_importance":
            _save_feature_importance_main_panels(
                feature_importance, dirs=dirs, options=options, warnings=warnings
            )
        elif builder == "predicted_habitat_maps":
            _save_primary_habitat_map_main_panels(
                train_summary, dirs=dirs, options=options, warnings=warnings
            )


def _metric_comparison_figure(
    df: pd.DataFrame,
    *,
    category: str,
    path: Path,
    title: str,
    order: Sequence[str],
    options: ReportOptions,
) -> None:
    """Two-panel validation-specific AP and Integrated AP ranking figure."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    data = df.copy()
    data = data[data[category].astype(str).isin([str(value) for value in order])].copy()
    data[category] = pd.Categorical(data[category].astype(str), categories=list(order), ordered=True)
    data = data.sort_values(category)
    metrics = [column for column in ["within_AP", "cross_AP", "spatial_AP"] if column in data.columns]
    if "mean_AP" not in data.columns:
        data["mean_AP"] = data[metrics].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1.0]})
    labels = [_short_label(_display_label(value), 20) for value in data[category].astype(str)]
    x = np.arange(len(labels))
    width = 0.75 / max(len(metrics), 1)
    all_values: list[float] = []
    for idx, metric in enumerate(metrics):
        values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype="float64")
        all_values.extend([float(value) for value in values if np.isfinite(value)])
        axes[0].bar(x + (idx - (len(metrics) - 1) / 2) * width, values, width=width, label=_display_column_name(metric))
    axes[0].set_title("A. Validation-specific AP", fontsize=11)
    axes[0].set_ylabel("Average precision")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylim(*_nice_ylim(all_values, lower_floor=0.0))
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    rank_df = data.sort_values("mean_AP", ascending=True)
    y = np.arange(rank_df.shape[0])
    mean_values = pd.to_numeric(rank_df["mean_AP"], errors="coerce").to_numpy(dtype="float64")
    axes[1].barh(y, mean_values, color="#374151")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([_short_label(_display_label(value), 24) for value in rank_df[category].astype(str)])
    axes[1].set_xlabel("Integrated AP")
    axes[1].set_title("B. Integrated AP ranking", fontsize=11)
    axes[1].set_xlim(*_nice_ylim(mean_values, lower_floor=0.0))
    axes[1].grid(axis="x", alpha=0.25)
    for yi, value in zip(y, mean_values):
        if np.isfinite(value):
            axes[1].text(value, yi, f" {value:.3f}", va="center", ha="left", fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    _write_figure_source(path, data, options)


def _main_metric_comparison_figure(
    df: pd.DataFrame,
    *,
    category: str,
    path: Path,
    title: str,
    order: Sequence[str],
    options: ReportOptions,
) -> None:
    """Main Fig.4/Fig.5 five-panel comparison without inset overlays."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    data = df.copy()
    data = data[data[category].astype(str).isin([str(value) for value in order])].copy()
    data[category] = pd.Categorical(data[category].astype(str), categories=list(order), ordered=True)
    data = data.sort_values(category)
    metrics = [column for column in ["within_AP", "cross_AP", "spatial_AP"] if column in data.columns]
    if "mean_AP" not in data.columns:
        data["mean_AP"] = data[metrics].mean(axis=1)

    fig = plt.figure(figsize=(16.8, 8.8), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.15, 1.0],
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.21,
        hspace=0.58,
        wspace=0.45,
    )
    ax_group = fig.add_subplot(gs[0, :])
    ranking_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
    ]
    labels = [_short_label(_display_label(value), 16) for value in data[category].astype(str)]
    x = np.arange(len(labels))
    width = 0.72 / max(len(metrics), 1)
    all_values: list[float] = []
    colors = {
        "within_AP": "#4f83b8",
        "cross_AP": "#c96b4a",
        "spatial_AP": "#5a9f68",
        "mean_AP": "#374151",
    }
    for idx, metric in enumerate(metrics):
        values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype="float64")
        all_values.extend([float(value) for value in values if np.isfinite(value)])
        ax_group.bar(
            x + (idx - (len(metrics) - 1) / 2) * width,
            values,
            width=width,
            color=colors.get(metric, None),
            label=_display_column_name(metric),
        )
    ax_group.set_title("(a) Validation-specific AP", fontsize=11)
    ax_group.set_ylabel("Average precision")
    ax_group.set_xticks(x)
    ax_group.set_xticklabels(labels, rotation=0, ha="center")
    ax_group.set_ylim(0, max(1.0, max(all_values) * 1.10 if all_values else 1.0))
    ax_group.grid(axis="y", alpha=0.25)
    handles, legend_labels = ax_group.get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.055))

    ranking_specs = [
        ("(b) Within-site AP", "within_AP"),
        ("(c) Cross-site AP", "cross_AP"),
        ("(d) Spatial block CV AP", "spatial_AP"),
        ("(e) Integrated AP", "mean_AP"),
    ]
    for ax, (panel_title, metric) in zip(ranking_axes, ranking_specs):
        if metric not in data.columns:
            ax.set_axis_off()
            continue
        rank_df = data.sort_values(metric, ascending=True)
        y = np.arange(rank_df.shape[0])
        values = pd.to_numeric(rank_df[metric], errors="coerce").to_numpy(dtype="float64")
        ax.barh(y, values, color=colors.get(metric, "#374151"))
        ax.set_yticks(y)
        ax.set_yticklabels([_short_label(_display_label(value), 22) for value in rank_df[category].astype(str)], fontsize=8)
        ax.set_xlabel("AP")
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlim(*_nice_ylim(values, lower_floor=0.0))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(axis="x", alpha=0.25)
        for yi, value in zip(y, values):
            if np.isfinite(value):
                ax.text(value, yi, f" {value:.3f}", va="center", ha="left", fontsize=7)
    figure_id = (
        "fig_04_feature_set_comparison"
        if path.stem == "fig_04_feature_set_comparison"
        else "fig_05_model_comparison"
    )
    panel_specs = _publication_panels(figure_id)
    metric_sources = [data] + [data[[category, metric]].copy() for _, metric in ranking_specs]
    _export_main_axes(
        fig=fig,
        axes=[ax_group, *ranking_axes],
        composite_path=path,
        panels=[
            (
                str(panel["letter"]),
                str(panel["title"]),
                str(panel["stem"]),
                metric_sources[index],
            )
            for index, panel in enumerate(panel_specs)
        ],
        options=options,
        plot_function="_main_metric_comparison_figure",
    )
    fig.suptitle(title, fontsize=MAIN_TITLE_FONTSIZE, fontweight="bold", y=0.975)
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    _write_figure_source(path, data, options)


def _save_horizontal_ranking(
    df: pd.DataFrame,
    *,
    category: str,
    value: str,
    path: Path,
    title: str,
    order: Sequence[str] | None,
    options: ReportOptions,
    top: int | None = None,
) -> None:
    """Publication-style horizontal ranking bar chart."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    data = df.copy()
    if value not in data.columns:
        _save_text_figure(path, title, ["No source metric available."], options.dpi, options.overwrite)
        return
    data[value] = pd.to_numeric(data[value], errors="coerce")
    if order is not None:
        data = data[data[category].astype(str).isin([str(item) for item in order])].copy()
    data = data.sort_values(value, ascending=False)
    if top is not None:
        data = data.head(top)
    plot_df = data.sort_values(value, ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, max(4.2, 0.45 * len(plot_df))), constrained_layout=True)
    values = plot_df[value].to_numpy(dtype="float64")
    y = np.arange(plot_df.shape[0])
    ax.barh(y, values, color="#2563eb")
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(_display_label(item), 28) for item in plot_df[category].astype(str)])
    ax.set_xlabel(_display_column_name(value))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(*_nice_ylim(values, lower_floor=0.0))
    ax.grid(axis="x", alpha=0.25)
    for yi, val in zip(y, values):
        if np.isfinite(val):
            ax.text(val, yi, f" {val:.3f}", va="center", ha="left", fontsize=8)
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    _write_figure_source(path, data, options)


def _save_rank_stability_lollipop(
    df: pd.DataFrame,
    *,
    category: str,
    path: Path,
    title: str,
    options: ReportOptions,
    top: int | None = None,
) -> None:
    """Integrated AP with min-max range lollipop chart."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    required = [category, "mean_AP", "min_AP", "max_AP"]
    if any(column not in df.columns for column in required):
        _save_text_figure(path, title, ["Rank stability columns unavailable."], options.dpi, options.overwrite)
        return
    data = df.copy()
    for column in ["mean_AP", "min_AP", "max_AP"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values("mean_AP", ascending=False)
    if top is not None:
        data = data.head(top)
    plot_df = data.sort_values("mean_AP", ascending=True)
    y = np.arange(plot_df.shape[0])
    fig, ax = plt.subplots(figsize=(9.2, max(4.5, 0.45 * len(plot_df))), constrained_layout=True)
    ax.hlines(y, plot_df["min_AP"], plot_df["max_AP"], color="#9ca3af", linewidth=3)
    ax.scatter(plot_df["mean_AP"], y, color="#111827", s=36, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(_display_label(value), 30) for value in plot_df[category].astype(str)])
    ax.set_xlabel("Average precision")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(*_nice_ylim(pd.concat([plot_df["min_AP"], plot_df["max_AP"]]).to_numpy(dtype="float64"), lower_floor=0.0))
    ax.grid(axis="x", alpha=0.25)
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    _write_figure_source(path, data, options)


def _save_candidate_comparison_figure(df: pd.DataFrame, path: Path, options: ReportOptions) -> None:
    """Main Fig. 7 final candidate comparison."""

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    data = df.copy()
    data = data[data["candidate_label"].astype(str).isin(CANDIDATE_ORDER)].copy()
    data["candidate_label"] = pd.Categorical(data["candidate_label"].astype(str), categories=CANDIDATE_ORDER, ordered=True)
    data = data.sort_values("candidate_label")
    metrics = [column for column in ["within_AP", "cross_AP", "spatial_AP"] if column in data.columns]
    if "mean_AP" not in data.columns:
        data["mean_AP"] = data[metrics].mean(axis=1)

    labels: list[str] = []
    for _, row in data.iterrows():
        labels.append(
            f"{_display_label(str(row['candidate_label']))}\n"
            f"{_display_label(str(row.get('feature_set', '')))} / {_display_label(str(row.get('model_id', '')))}"
        )
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(15.8, 6.6),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.45, 1.05]},
    )
    fig.subplots_adjust(left=0.27, right=0.98, top=0.88, bottom=0.19, wspace=0.62)
    y = np.arange(len(labels))
    height = 0.72 / max(len(metrics), 1)
    colors = {"within_AP": "#4f83b8", "cross_AP": "#c96b4a", "spatial_AP": "#5a9f68"}
    all_values: list[float] = []
    for idx, metric in enumerate(metrics):
        values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype="float64")
        all_values.extend([float(value) for value in values if np.isfinite(value)])
        ax_a.barh(
            y + (idx - (len(metrics) - 1) / 2) * height,
            values,
            height=height,
            color=colors.get(metric, None),
            label=_display_column_name(metric),
        )
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=9)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Average precision")
    ax_a.set_title("(a) Validation-specific AP", fontsize=11)
    ax_a.set_xlim(0, max(1.0, max(all_values) * 1.10 if all_values else 1.0))
    ax_a.grid(axis="x", alpha=0.25)

    rank_df = data.sort_values("mean_AP", ascending=True)
    rank_labels = [
        f"{_display_label(str(row['candidate_label']))}\n{_display_label(str(row.get('feature_set', '')))} / {_display_label(str(row.get('model_id', '')))}"
        for _, row in rank_df.iterrows()
    ]
    mean_values = pd.to_numeric(rank_df["mean_AP"], errors="coerce").to_numpy(dtype="float64")
    y2 = np.arange(rank_df.shape[0])
    ax_b.barh(y2, mean_values, color="#374151")
    ax_b.set_yticks(y2)
    ax_b.set_yticklabels(rank_labels, fontsize=8.5)
    ax_b.set_xlabel("Integrated AP")
    ax_b.set_title("(b) Integrated AP ranking", fontsize=11)
    ax_b.set_xlim(*_nice_ylim(mean_values, lower_floor=0.0))
    ax_b.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax_b.grid(axis="x", alpha=0.25)
    for yi, value in zip(y2, mean_values):
        if np.isfinite(value):
            ax_b.text(value, yi, f" {value:.3f}", va="center", ha="left", fontsize=8)

    handles, legend_labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.045))
    panel_specs = _publication_panels("fig_07_final_candidate_comparison")
    _export_main_axes(
        fig=fig,
        axes=[ax_a, ax_b],
        composite_path=path,
        panels=[
            (str(panel_specs[0]["letter"]), str(panel_specs[0]["title"]), str(panel_specs[0]["stem"]), data),
            (str(panel_specs[1]["letter"]), str(panel_specs[1]["title"]), str(panel_specs[1]["stem"]), rank_df),
        ],
        options=options,
        plot_function="_save_candidate_comparison_figure",
    )
    _add_centered_main_title(fig, "Final candidate comparison", y=0.965)
    _save_main_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    _write_figure_source(path, data, options)


def _save_feature_importance_main(
    feature_importance: pd.DataFrame,
    path: Path,
    options: ReportOptions,
    warnings: list[str],
) -> None:
    """Main Fig. 8 integrated feature importance across final candidates."""

    required = {"candidate_label", "feature", "importance"}
    if feature_importance.empty or not required.issubset(feature_importance.columns):
        message = "Fig.8 skipped: final candidate feature-importance inputs are incomplete."
        warnings.append(message)
        print(f"Warning: {message}", flush=True)
        return
    final_data = feature_importance.copy()
    final_data["candidate_label"] = final_data["candidate_label"].astype(str)
    candidate_set = set(final_data["candidate_label"].dropna().astype(str))
    missing_candidates = [label for label in CANDIDATE_ORDER if label not in candidate_set]
    missing_sites = [
        site_id
        for site_id in SITE_ORDER
        if "site_id" not in final_data.columns
        or site_id not in set(final_data["site_id"].dropna().astype(str))
    ]
    if missing_candidates:
        message = (
            "Fig.8 skipped: all four final candidates are required for integrated "
            f"feature importance; missing {', '.join(missing_candidates)}."
        )
        warnings.append(message)
        print(f"Warning: {message}", flush=True)
        return
    if missing_sites:
        message = (
            "Fig.8 skipped: final candidate feature importance is required for both sites; "
            f"missing {', '.join(missing_sites)}."
        )
        warnings.append(message)
        print(f"Warning: {message}", flush=True)
        return
    data = final_data[final_data["candidate_label"].isin(CANDIDATE_ORDER)].copy()
    data["importance_source"] = "final_maps"
    data["importance"] = pd.to_numeric(data["importance"], errors="coerce")
    data = data[np.isfinite(data["importance"])].copy()
    if data.empty:
        message = "Fig.8 skipped: no finite final-candidate feature-importance values."
        warnings.append(message)
        print(f"Warning: {message}", flush=True)
        return
    if float(data["importance"].abs().max()) > 100.0:
        message = (
            "Fig.8 skipped: feature importance scale is unexpectedly large; "
            "only final_maps feature_importance.csv should be used."
        )
        warnings.append(message)
        print(f"Warning: {message}", flush=True)
        return
    summary, group_summary = _feature_importance_summaries(data)
    feature_plot = summary.sort_values("mean_importance", ascending=True).copy()

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    fig_height = max(6.2, 1.9 + 0.34 * feature_plot.shape[0])
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, fig_height),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.35, 0.9]},
    )
    fig.subplots_adjust(left=0.24, right=0.98, top=0.84, bottom=0.10, wspace=0.52)
    y = np.arange(feature_plot.shape[0])
    colors = [FEATURE_GROUP_COLORS.get(str(group), FEATURE_GROUP_COLORS["Other"]) for group in feature_plot["feature_group"]]
    axes[0].barh(
        y,
        feature_plot["mean_importance"].to_numpy(dtype="float64"),
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_short_label(_display_label(feature), 28) for feature in feature_plot["feature"].astype(str)])
    axes[0].set_xlabel("Mean feature importance")
    axes[0].set_title("(a) Mean feature importance", fontsize=11)
    axes[0].grid(axis="x", alpha=0.25)

    group_summary = group_summary[
        group_summary["feature_group"].astype(str).isin(FEATURE_GROUP_ORDER[:-1])
    ].copy()
    y2 = np.arange(group_summary.shape[0])
    axes[1].barh(
        y2,
        group_summary["mean_group_importance"].to_numpy(dtype="float64"),
        color=[
            FEATURE_GROUP_COLORS.get(str(group), FEATURE_GROUP_COLORS["Other"])
            for group in group_summary["feature_group"].astype(str)
        ],
    )
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(group_summary["feature_group"].astype(str))
    axes[1].set_xlabel("Mean group importance")
    axes[1].set_title("(b) Mean feature-group importance", fontsize=11)
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].invert_yaxis()

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=FEATURE_GROUP_COLORS[group], edgecolor="white", label=group)
        for group in FEATURE_GROUP_ORDER[:-1]
        if group in set(summary["feature_group"].astype(str))
    ]
    axes[0].legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    panel_specs = _publication_panels("fig_08_feature_importance")
    _export_main_axes(
        fig=fig,
        axes=list(axes),
        composite_path=path,
        panels=[
            (str(panel_specs[0]["letter"]), str(panel_specs[0]["title"]), str(panel_specs[0]["stem"]), feature_plot),
            (str(panel_specs[1]["letter"]), str(panel_specs[1]["title"]), str(panel_specs[1]["stem"]), group_summary),
        ],
        options=options,
        plot_function="_save_feature_importance_main",
    )
    _add_centered_main_title(fig, "Feature importance across final candidates", y=0.965)
    _save_main_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    source = pd.concat(
        [
            feature_plot.assign(panel="feature_importance"),
            group_summary.assign(feature="", sd_importance=np.nan, min_importance=np.nan, max_importance=np.nan, n=np.nan, panel="group_importance"),
        ],
        ignore_index=True,
        sort=False,
    )
    _write_figure_source(path, source, options)


def _save_primary_habitat_maps_figure(
    train_summary: pd.DataFrame,
    dirs: ReportDirs,
    path: Path,
    options: ReportOptions,
    warnings: list[str],
) -> None:
    """Main Fig. 9 primary-balanced probability and binary habitat maps."""

    records = [
        record
        for record in _map_record_iter(train_summary)
        if str(record["candidate_label"]) == PRIMARY_PUBLICATION_CANDIDATE
        and str(record["site_id"]) in SITE_ORDER
    ]
    records = sorted(records, key=lambda r: SITE_ORDER.index(str(r["site_id"])) if str(r["site_id"]) in SITE_ORDER else 999)
    by_site = {str(record["site_id"]): record for record in records}
    missing_paths: list[str] = []
    for site_id in SITE_ORDER:
        if site_id not in by_site:
            missing_paths.append(
                f"final_maps/{site_id}/{PRIMARY_PUBLICATION_CANDIDATE}/train_summary.csv record"
            )
            continue
        for key in ["probability_src", "valid_mask_src"]:
            source_path = Path(str(by_site[site_id][key]))
            if not source_path.exists():
                missing_paths.append(str(source_path))
    if missing_paths:
        message = (
            f"Fig.9 skipped: {PRIMARY_PUBLICATION_CANDIDATE} final_maps probability and valid-mask "
            f"inputs are required; missing {'; '.join(missing_paths)}."
        )
        warnings.append(message)
        print(f"Warning: {message}", flush=True)
        return
    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    ordered_records = [by_site[site_id] for site_id in SITE_ORDER]
    threshold = 0.5
    quicklooks: dict[str, tuple[np.ma.MaskedArray, np.ma.MaskedArray, Any]] = {}
    for record in ordered_records:
        site_id = str(record["site_id"])
        quicklooks[site_id] = _read_probability_binary_quicklook(
            Path(str(record["probability_src"])),
            Path(str(record["valid_mask_src"])),
            options.max_preview_size,
            threshold,
        )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.91, bottom=0.12, hspace=0.34, wspace=0.28)
    cmap_prob = plt.get_cmap(options.map_cmap).copy()
    cmap_prob.set_bad("#d9d9d9")
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap_bin = ListedColormap(["white", "#064e3b"])
    cmap_bin.set_bad("#d9d9d9")
    binary_norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap_bin.N)
    panels = [
        ("(a)", 0, 0, ordered_records[0], "probability"),
        ("(b)", 0, 1, ordered_records[0], "binary"),
        ("(c)", 1, 0, ordered_records[1], "probability"),
        ("(d)", 1, 1, ordered_records[1], "binary"),
    ]
    for panel_label, row, col, record, kind in panels:
        probability, binary, bounds = quicklooks[str(record["site_id"])]
        data = probability if kind == "probability" else binary
        ax = axes[row, col]
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        if kind == "probability":
            image = ax.imshow(data, cmap=cmap_prob, vmin=0, vmax=1, extent=extent, origin="upper")
            cbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.045)
            cbar.set_label("Predicted habitat probability")
            map_label = "predicted habitat probability"
        else:
            image = ax.imshow(data, cmap=cmap_bin, norm=binary_norm, extent=extent, origin="upper")
            cbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.045, ticks=[0, 1])
            cbar.ax.set_yticklabels(["P < 0.5", "P ≥ 0.5"])
            cbar.set_label("Thresholded habitat prediction")
            map_label = f"predicted habitat (P ≥ {threshold:g})"
        ax.set_title(f"{panel_label} {_display_label(record['site_id'])} {map_label}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m, UTM Zone 52N)", fontsize=8)
        ax.set_ylabel("Northing (m, UTM Zone 52N)", fontsize=8)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=7, rotation=0)
        add_map_cartography(ax, bounds)
    panel_specs = _publication_panels("fig_09_predicted_habitat_maps")
    _export_main_axes(
        fig=fig,
        axes=list(axes.ravel()),
        composite_path=path,
        panels=[
            (
                str(panel_specs[index]["letter"]),
                str(panel_specs[index]["title"]),
                str(panel_specs[index]["stem"]),
                pd.DataFrame([
                    {
                        **panels[index][3],
                        "panel_kind": panels[index][4],
                        "threshold": threshold,
                    }
                ]),
            )
            for index in range(len(panels))
        ],
        options=options,
        plot_function="_save_primary_habitat_maps_figure",
    )
    _add_centered_main_title(fig, "Predicted habitat maps", y=0.975)
    _save_main_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    source = pd.DataFrame(ordered_records)
    source["threshold"] = threshold
    source["candidate_label"] = PRIMARY_PUBLICATION_CANDIDATE
    source["feature_set"] = ordered_records[0]["feature_set"]
    source["model_id"] = ordered_records[0]["model_id"]
    source["crs"] = "EPSG:32652"
    _write_figure_source(path, source, options)


def _curve_mean_ci(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    grid: np.ndarray,
) -> pd.DataFrame:
    """Interpolate fold curves to a common grid and return mean/CI."""

    if data.empty or x_col not in data.columns or y_col not in data.columns:
        return pd.DataFrame(columns=[x_col, "mean", "lower", "upper"])
    group_col = "source_path" if "source_path" in data.columns else None
    curves = []
    iterable = data.groupby(group_col) if group_col else [(None, data)]
    for _, group in iterable:
        x = pd.to_numeric(group[x_col], errors="coerce").to_numpy(dtype="float64")
        y = pd.to_numeric(group[y_col], errors="coerce").to_numpy(dtype="float64")
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.size < 2:
            continue
        order = np.argsort(x)
        x = x[order]
        y = np.clip(y[order], 0, 1)
        unique_x, unique_idx = np.unique(x, return_index=True)
        unique_y = y[unique_idx]
        if unique_x.size < 2:
            continue
        curves.append(np.interp(grid, unique_x, unique_y, left=unique_y[0], right=unique_y[-1]))
    if not curves:
        return pd.DataFrame(columns=[x_col, "mean", "lower", "upper"])
    arr = np.vstack(curves)
    return pd.DataFrame(
        {
            x_col: grid,
            "mean": np.nanmean(arr, axis=0),
            "lower": np.nanpercentile(arr, 2.5, axis=0),
            "upper": np.nanpercentile(arr, 97.5, axis=0),
            "curve_count": arr.shape[0],
        }
    )


def _save_validation_composite(
    *,
    output_root: Path,
    strategy_root: str,
    path: Path,
    title: str,
    options: ReportOptions,
) -> None:
    """Supplementary validation composite: confusion, PR mean/CI, ROC mean/CI."""

    root = output_root / strategy_root
    roc_raw = _collect_curve_frames(root, "roc_curve.csv")
    pr_raw = _collect_curve_frames(root, "pr_curve.csv")
    confusion = _collect_confusion_frames(root)
    fpr_grid = np.linspace(0, 1, 200)
    recall_grid = np.linspace(0, 1, 200)
    roc = _curve_mean_ci(roc_raw, x_col="fpr", y_col="tpr", grid=fpr_grid)
    pr = _curve_mean_ci(pr_raw, x_col="recall", y_col="precision", grid=recall_grid)

    plt = _require_matplotlib_pyplot()
    if path.exists() and not options.overwrite:
        print(f"Skipped existing figure due to --no-overwrite: {path}", flush=True)
        return
    fig = plt.figure(figsize=(11, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35])
    ax_cm = fig.add_subplot(gs[:, 0])
    ax_pr = fig.add_subplot(gs[0, 1])
    ax_roc = fig.add_subplot(gs[1, 1])

    if not confusion.empty and all(col in confusion.columns for col in ["TP", "TN", "FP", "FN"]):
        totals = {col: int(pd.to_numeric(confusion[col], errors="coerce").fillna(0).sum()) for col in ["TP", "TN", "FP", "FN"]}
        matrix = np.array([[totals["TN"], totals["FP"]], [totals["FN"], totals["TP"]]], dtype="float64")
        image = ax_cm.imshow(matrix, cmap="Blues")
        for y in range(2):
            for x in range(2):
                ax_cm.text(x, y, f"{int(matrix[y, x]):,}", ha="center", va="center", fontsize=11)
        ax_cm.set_xticks([0, 1])
        ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_yticklabels(["True 0", "True 1"])
        ax_cm.set_title("A. Confusion matrix", fontsize=11)
        fig.colorbar(image, ax=ax_cm, fraction=0.046, pad=0.04)
    else:
        ax_cm.text(0.5, 0.5, "No confusion data", ha="center", va="center")
        ax_cm.set_axis_off()

    for raw, ax, mean_df, x_col, y_col, label, panel in [
        (pr_raw, ax_pr, pr, "recall", "precision", "Precision-recall", "B"),
        (roc_raw, ax_roc, roc, "fpr", "tpr", "ROC", "C"),
    ]:
        if not raw.empty and x_col in raw.columns and y_col in raw.columns:
            for _, group in raw.groupby("source_path" if "source_path" in raw.columns else raw.index):
                ax.plot(
                    pd.to_numeric(group[x_col], errors="coerce"),
                    pd.to_numeric(group[y_col], errors="coerce"),
                    color="#64748b",
                    alpha=0.10,
                    linewidth=0.6,
                )
        if not mean_df.empty:
            ax.fill_between(mean_df[x_col], mean_df["lower"], mean_df["upper"], color="#60a5fa", alpha=0.22, label="95% CI")
            ax.plot(mean_df[x_col], mean_df["mean"], color="#1d4ed8", linewidth=3.0, label="Mean curve")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{panel}. {label}", fontsize=11)
        ax.set_xlabel("Recall" if x_col == "recall" else "False positive rate")
        ax.set_ylabel("Precision" if y_col == "precision" else "True positive rate")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    source = pd.concat(
        [
            pr.assign(curve_type="PR") if not pr.empty else pd.DataFrame(),
            roc.assign(curve_type="ROC") if not roc.empty else pd.DataFrame(),
            confusion.assign(curve_type="confusion") if not confusion.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    _write_figure_source(path, source, options)


def _supp_panel_label(ax: Any, label: str) -> None:
    """Place a supplementary panel letter in a consistent top-left position."""

    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )


def _supp_style_axis(ax: Any, *, grid_axis: str | None = "x") -> None:
    """Apply the publication-style axis treatment used by supplementary figures."""

    ax.tick_params(labelsize=10, width=0.8, length=4)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#111827")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#9ca3af", alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)


def _feature_set_family(feature_set: str) -> str:
    """Return a compact family label for spacing feature-set rankings."""

    text = str(feature_set)
    if text.startswith("single"):
        return "Single"
    if text.startswith("dual"):
        return "Dual"
    if text == "full_multi":
        return "Full-band"
    return "Other"


def _supp_canonical_feature_order(values: Sequence[Any]) -> list[str]:
    """Return available feature sets in canonical Single, Dual, Full-band order."""

    available = [str(value) for value in values]
    available_set = set(available)
    ordered = [feature_set for feature_set in FEATURE_ORDER if feature_set in available_set]
    ordered.extend(feature_set for feature_set in available if feature_set not in FEATURE_ORDER)
    return list(dict.fromkeys(ordered))


def _supp_order_feature_rows(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Order feature rows canonically so each feature family remains contiguous."""

    plot_df = df.copy()
    canonical_order = _supp_canonical_feature_order(plot_df[category].astype(str))
    order_lookup = {feature_set: idx for idx, feature_set in enumerate(canonical_order)}
    plot_df["_order"] = plot_df[category].astype(str).map(order_lookup).fillna(len(order_lookup))
    return plot_df.sort_values("_order", kind="stable")


def _feature_model_label(row: Mapping[str, Any]) -> str:
    """Return a compact feature x model label."""

    feature = _display_label(row.get("feature_set", ""))
    model = _display_label(row.get("model_id", ""))
    return f"{feature} / {model}"


def _candidate_compact_label(candidate: Any) -> str:
    """Return compact final-candidate label for crowded supplementary axes."""

    label = CANDIDATE_COMPACT_LABELS.get(str(candidate), _display_label(candidate))
    return label.replace(" ", "\n", 1) if "\n" not in label else label


def _supp_candidate_label(candidate: Any) -> str:
    """Return the publication candidate label shared by Tables and S6 panels."""

    return _publication_candidate_label(candidate)


def _supp_order_candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return candidate rows in the fixed final-candidate order."""

    if df.empty or "candidate_label" not in df.columns:
        return df.copy()
    ordered = df.copy()
    lookup = {candidate: idx for idx, candidate in enumerate(CANDIDATE_ORDER)}
    ordered["_candidate_order"] = ordered["candidate_label"].astype(str).map(lookup)
    ordered = ordered.dropna(subset=["_candidate_order"])
    return ordered.sort_values("_candidate_order", kind="stable").drop(columns="_candidate_order")


def _supp_candidate_color(candidate: Any) -> str:
    """Return the stable candidate color shared by S6 panels."""

    return SUPPLEMENTARY_CANDIDATE_COLORS.get(str(candidate), "#6b7280")


def _supp_y_positions(labels: Sequence[str], groups: Sequence[str] | None = None) -> np.ndarray:
    """Return y positions with a small visual gap whenever the group changes."""

    positions: list[float] = []
    current = 0.0
    previous_group: str | None = None
    for idx, _ in enumerate(labels):
        group = groups[idx] if groups is not None and idx < len(groups) else None
        if previous_group is not None and group is not None and group != previous_group:
            current += 0.45
        positions.append(current)
        current += 1.0
        previous_group = group
    return np.asarray(positions, dtype="float64")


def _supp_horizontal_ap_bar(
    ax: Any,
    df: pd.DataFrame,
    *,
    category: str,
    value: str = "AP_mean",
    order: Sequence[str] | None = None,
    sort_desc: bool = True,
    color: str = "#4f83b8",
    xlabel: str = "Average precision (AP)",
    group_gap: bool = False,
    top: int | None = None,
) -> pd.DataFrame:
    """Draw a supplementary horizontal AP/importance ranking panel."""

    if df.empty or category not in df.columns or value not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return pd.DataFrame()
    plot_df = df.copy()
    plot_df[value] = pd.to_numeric(plot_df[value], errors="coerce")
    plot_df = plot_df.dropna(subset=[value])
    if order is not None:
        order_lookup = {str(item): idx for idx, item in enumerate(order)}
        plot_df["_order"] = plot_df[category].astype(str).map(order_lookup).fillna(len(order_lookup))
    else:
        plot_df["_order"] = np.arange(plot_df.shape[0])
    if group_gap and category == "feature_set":
        plot_df = _supp_order_feature_rows(plot_df, category)
    elif sort_desc:
        plot_df = plot_df.sort_values([value, "_order"], ascending=[False, True])
    else:
        plot_df = plot_df.sort_values("_order")
    if top is not None:
        plot_df = plot_df.head(top)
    labels = [_display_label(item) for item in plot_df[category].astype(str)]
    groups = [_feature_set_family(item) for item in plot_df[category].astype(str)] if group_gap else None
    y = _supp_y_positions(labels, groups)
    values = plot_df[value].to_numpy(dtype="float64")
    ax.barh(y, values, color=color, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    max_val = float(np.nanmax(values)) if values.size else 1.0
    ax.set_xlim(0, max(max_val * 1.16, 0.05))
    ax.set_xlabel(xlabel)
    for yi, val in zip(y, values):
        ax.text(val, yi, f" {val:.3f}", va="center", ha="left", fontsize=9)
    _supp_style_axis(ax, grid_axis="x")
    return plot_df.drop(columns=["_order"], errors="ignore")


def _supp_grouped_horizontal_ap(
    ax: Any,
    df: pd.DataFrame,
    *,
    category: str,
    metrics: Sequence[str],
    order: Sequence[str],
    ylabel: str,
    show_legend: bool = True,
) -> pd.DataFrame:
    """Draw validation-specific AP columns as grouped horizontal bars."""

    plot_df = df.copy()
    plot_df = plot_df[plot_df[category].astype(str).isin([str(item) for item in order])].copy()
    order_lookup = {str(item): idx for idx, item in enumerate(order)}
    plot_df["_order"] = plot_df[category].astype(str).map(order_lookup)
    plot_df = plot_df.sort_values("_order")
    labels = [_display_label(item) for item in plot_df[category].astype(str)]
    y = np.arange(len(labels), dtype="float64")
    height = 0.72 / max(len(metrics), 1)
    colors = {"within_AP": "#4f83b8", "cross_AP": "#c96b4a", "spatial_AP": "#5a9f68"}
    all_values: list[float] = []
    for idx, metric in enumerate(metrics):
        values = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy(dtype="float64")
        all_values.extend([float(v) for v in values if np.isfinite(v)])
        ax.barh(
            y + (idx - (len(metrics) - 1) / 2) * height,
            values,
            height=height,
            color=colors.get(metric, "#6b7280"),
            label=_display_column_name(metric),
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Average precision (AP)")
    ax.set_xlim(0, max(1.03, max(all_values) * 1.12 if all_values else 1.03))
    if show_legend:
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(len(metrics), 3),
            fontsize=9,
        )
    _supp_style_axis(ax, grid_axis="x")
    return plot_df.drop(columns=["_order"], errors="ignore")


def _supp_heatmap_panel(
    fig: Any,
    ax: Any,
    df: pd.DataFrame,
    *,
    row: str,
    column: str,
    value: str,
    row_order: Sequence[str] | None = None,
    column_order: Sequence[str] | None = None,
    cbar_label: str = "AP",
) -> pd.DataFrame:
    """Draw a heatmap panel with white cell boundaries and bold row maxima."""

    if df.empty or row not in df.columns or column not in df.columns or value not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return pd.DataFrame()
    pivot = df.pivot_table(index=row, columns=column, values=value, aggfunc="mean")
    if row == "feature_set":
        available_rows = pivot.index if row_order is None else [item for item in row_order if item in pivot.index]
        pivot = pivot.reindex(_supp_canonical_feature_order(available_rows))
    elif row_order is not None:
        pivot = pivot.reindex([item for item in row_order if item in pivot.index])
    if column_order is not None:
        pivot = pivot.reindex(columns=[item for item in column_order if item in pivot.columns])
    values = pivot.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if math.isclose(vmin, vmax):
        vmin -= 0.01
        vmax += 0.01
    image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    xticklabels = [
        "Random\nForest" if str(item) == "random_forest" else _short_label(_display_label(item), 16)
        for item in pivot.columns
    ]
    ax.set_xticklabels(xticklabels, rotation=0, ha="center")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([_short_label(_display_label(item), 24) for item in pivot.index])
    ax.set_xticks(np.arange(-0.5, pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    row_max = np.nanmax(values, axis=1) if values.size else np.array([])
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.iloc[y, x]
            if pd.notna(val):
                norm = (float(val) - vmin) / (vmax - vmin)
                text_color = "black" if norm > 0.62 else "white"
                weight = "bold" if np.isfinite(row_max[y]) and math.isclose(float(val), float(row_max[y]), rel_tol=1e-9, abs_tol=1e-9) else "normal"
                ax.text(x, y, f"{float(val):.3f}", ha="center", va="center", fontsize=9.5, color=text_color, fontweight=weight)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.018)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    return pivot.reset_index()


def _supp_dumbbell_rank(
    ax: Any,
    df: pd.DataFrame,
    *,
    label_func: Any,
    top: int | None = None,
    order_key: Any | None = None,
) -> pd.DataFrame:
    """Draw best/mean/worst rank as a dumbbell plot."""

    required = {"best_rank", "worst_rank", "mean_rank"}
    if df.empty or not required.issubset(df.columns):
        ax.text(0.5, 0.5, "No rank data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return pd.DataFrame()
    plot_df = df.copy()
    for col in ["best_rank", "worst_rank", "mean_rank"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=["best_rank", "worst_rank", "mean_rank"])
    if order_key is None:
        plot_df = plot_df.sort_values("mean_rank")
    else:
        plot_df["_plot_order"] = [order_key(row) for _, row in plot_df.iterrows()]
        plot_df = plot_df.sort_values("_plot_order", kind="stable")
    if top is not None:
        plot_df = plot_df.head(top)
    labels = [label_func(row) for _, row in plot_df.iterrows()]
    y = np.arange(plot_df.shape[0], dtype="float64")
    best = plot_df["best_rank"].to_numpy(dtype="float64")
    worst = plot_df["worst_rank"].to_numpy(dtype="float64")
    mean = plot_df["mean_rank"].to_numpy(dtype="float64")
    ax.hlines(y, best, worst, color="#94a3b8", linewidth=2.0)
    ax.scatter(best, y, color="#4f83b8", s=26, label="Best")
    ax.scatter(mean, y, color="#111827", edgecolor="white", linewidth=0.7, s=48, label="Mean", zorder=3)
    ax.scatter(worst, y, color="#c96b4a", s=26, label="Worst")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Rank (1 = best)")
    ax.set_xlim(0.5, max(float(np.nanmax(worst)) + 0.8, 2.0))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
    _supp_style_axis(ax, grid_axis="x")
    return plot_df.drop(columns="_plot_order", errors="ignore")


def _read_existing_or_current(current: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Use current data when available, otherwise fall back to an existing CSV."""

    if current is not None and not current.empty:
        return current.copy()
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _supplementary_panel_stem(composite_stem: str, letter: str, title: str) -> str:
    """Return the configured stable Supplementary panel stem.

    Reader-facing titles may change without renaming machine-facing Figure IDs
    or filenames.  Derive a slug only for unconfigured auxiliary figures.
    """

    for figure in configured_figure_specs(PUBLICATION_CONFIG):
        if str(figure.get("stem")) != composite_stem:
            continue
        for panel in figure.get("panels", []):
            if str(panel.get("letter")) == str(letter) and panel.get("stem"):
                return str(panel["stem"])

    match = re.match(r"^(fig_s\d{2})_", composite_stem)
    if match is None:
        raise ValueError(f"Unexpected Supplementary Figure stem: {composite_stem}")
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"{match.group(1)}{letter}_{slug}"


def _axes_export_bbox(fig: Any, ax: Any, main_axes: Sequence[Any]) -> Any:
    """Return a padded tight bbox for one axes and its adjacent colorbar."""

    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer)]
    ax_position = ax.get_position()
    for extra_ax in fig.axes:
        if extra_ax in main_axes:
            continue
        position = extra_ax.get_position()
        if (
            position.x0 <= ax_position.x0
            and position.x1 >= ax_position.x1
            and position.y0 <= ax_position.y0
            and position.y1 >= ax_position.y1
        ):
            # Do not pull a full-figure scaffold/parent axes into an inset-panel crop.
            continue
        vertical_overlap = max(
            0.0,
            min(ax_position.y1, position.y1) - max(ax_position.y0, position.y0),
        )
        horizontal_gap = max(
            0.0,
            position.x0 - ax_position.x1,
            ax_position.x0 - position.x1,
        )
        if (
            vertical_overlap > 0.45 * min(ax_position.height, position.height)
            and horizontal_gap <= 0.25 * ax_position.width
        ):
            boxes.append(extra_ax.get_tightbbox(renderer))
    bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    return bbox.expanded(1.06, 1.10)


def _export_main_axes(
    *,
    fig: Any,
    axes: Sequence[Any],
    composite_path: Path,
    panels: Sequence[tuple[str, str, str, pd.DataFrame]],
    options: ReportOptions,
    plot_function: str,
) -> None:
    """Crop official Main panels from the exact composite axes authority."""

    if len(axes) != len(panels):
        raise ValueError(
            f"Main panel/axes count mismatch for {composite_path.stem}: "
            f"{len(panels)} panels vs {len(axes)} axes."
        )
    panel_root = composite_path.parent.parent / "panels" / composite_path.stem
    panel_root.mkdir(parents=True, exist_ok=True)
    for ax, (letter, title, stem, source) in zip(axes, panels):
        path = panel_root / f"{stem}.png"
        bbox = _axes_export_bbox(fig, ax, axes)
        fig.savefig(path, dpi=PANEL_DPI, bbox_inches=bbox, pad_inches=0.08, facecolor="white")
        if options.save_vector:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches=bbox, pad_inches=0.08, facecolor="white")
            fig.savefig(path.with_suffix(".svg"), bbox_inches=bbox, pad_inches=0.08, facecolor="white")
        _write_figure_source(path, source, options)
        if options.generate_metadata:
            _write_figure_metadata(
                path,
                parent_figure=composite_path.stem,
                panel_letter=letter,
                panel_title=title,
                official_or_auxiliary="official_panel",
                plot_function=plot_function,
                source_data=source,
                options=options,
                extra={"graphical_authority": "cropped_from_composite_axes"},
            )


def _export_supplementary_axes(
    *,
    fig: Any,
    axes: Sequence[Any],
    composite_path: Path,
    panels: Sequence[tuple[str, str, pd.DataFrame]],
    options: ReportOptions,
    plot_function: str,
    metadata_extra: Mapping[str, Any] | None = None,
) -> None:
    """Export labeled Supplementary axes as vector-preserving panel artifacts."""

    if len(axes) != len(panels):
        raise ValueError(
            f"Panel/axes count mismatch for {composite_path.stem}: "
            f"{len(panels)} panels vs {len(axes)} axes."
        )
    panel_root = composite_path.parent.parent / "panels" / composite_path.stem
    panel_root.mkdir(parents=True, exist_ok=True)
    if options.overwrite:
        for existing in panel_root.iterdir():
            if existing.is_file():
                existing.unlink()
    for ax, (letter, title, source) in zip(axes, panels):
        stem = _supplementary_panel_stem(composite_path.stem, letter, title)
        path = panel_root / f"{stem}.png"
        bbox = _axes_export_bbox(fig, ax, axes)
        fig.savefig(path, dpi=PANEL_DPI, bbox_inches=bbox, pad_inches=0.08, facecolor="white")
        if options.save_vector:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches=bbox, pad_inches=0.08, facecolor="white")
            fig.savefig(path.with_suffix(".svg"), bbox_inches=bbox, pad_inches=0.08, facecolor="white")
        _write_figure_source(path, source, options)
        if options.generate_captions:
            _write_sidecar_text(
                path.with_name(path.stem + "_caption.txt"),
                f"Panel ({letter}). {title}.\n",
                options.overwrite,
            )
        if options.generate_metadata:
            _write_figure_metadata(
                path,
                parent_figure=composite_path.stem,
                panel_letter=letter,
                panel_title=title,
                official_or_auxiliary="official_panel",
                plot_function=plot_function,
                source_data=source,
                options=options,
                extra=metadata_extra,
            )


def _save_supplementary_s1_s3(
    *,
    model_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_model_df: pd.DataFrame,
    path: Path,
    options: ReportOptions,
    prefix: str,
) -> None:
    """Save S1-S3 validation ranking composite figures."""

    plt = _require_matplotlib_pyplot()
    fig = plt.figure(figsize=(8.2, 12.2), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.55, 2.25], hspace=0.48)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    source_frames: list[pd.DataFrame] = []
    source_frames.append(_supp_horizontal_ap_bar(axes[0], model_df, category="model_id", value="AP_mean", order=MODEL_ORDER, color="#4f83b8"))
    _supp_panel_label(axes[0], "(a)")
    source_frames.append(_supp_horizontal_ap_bar(axes[1], feature_df, category="feature_set", value="AP_mean", order=FEATURE_ORDER, color="#5a9f68", group_gap=True))
    _supp_panel_label(axes[1], "(b)")
    source_frames.append(
        _supp_heatmap_panel(
            fig,
            axes[2],
            feature_model_df,
            row="feature_set",
            column="model_id",
            value="AP_mean",
            row_order=FEATURE_ORDER,
            column_order=MODEL_ORDER,
            cbar_label="AP",
        )
    )
    _supp_panel_label(axes[2], "(c)")
    fig.subplots_adjust(left=0.24, right=0.90, bottom=0.07, top=0.97)
    _export_supplementary_axes(
        fig=fig,
        axes=axes,
        composite_path=path,
        panels=[
            ("a", "Model ranking", source_frames[0].assign(supplementary_figure=prefix, panel="a")),
            ("b", "Feature-set ranking", source_frames[1].assign(supplementary_figure=prefix, panel="b")),
            ("c", "Feature by model heatmap", source_frames[2].assign(supplementary_figure=prefix, panel="c")),
        ],
        options=options,
        plot_function="_save_supplementary_s1_s3",
        metadata_extra={"validation_strategy": "within_site" if prefix == "S1" else "spatial_block_cv"},
    )
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    source = pd.concat([frame.assign(panel=panel) for frame, panel in zip(source_frames, ["a", "b", "c"]) if not frame.empty], ignore_index=True)
    source["supplementary_figure"] = prefix
    _write_figure_source(path, source, options)


def _save_supplementary_s2(
    *,
    model_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_model_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    path: Path,
    options: ReportOptions,
) -> None:
    """Save S2 cross-site validation composite."""

    plt = _require_matplotlib_pyplot()
    fig = plt.figure(figsize=(8.2, 15.2), constrained_layout=False)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.35, 2.0, 1.8], hspace=0.48)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    source_frames: list[pd.DataFrame] = []
    source_frames.append(_supp_horizontal_ap_bar(axes[0], model_df, category="model_id", value="AP_mean", order=MODEL_ORDER, color="#4f83b8"))
    _supp_panel_label(axes[0], "(a)")
    source_frames.append(_supp_horizontal_ap_bar(axes[1], feature_df, category="feature_set", value="AP_mean", order=FEATURE_ORDER, color="#5a9f68", group_gap=True))
    _supp_panel_label(axes[1], "(b)")
    source_frames.append(_supp_heatmap_panel(fig, axes[2], feature_model_df, row="feature_set", column="model_id", value="AP_mean", row_order=FEATURE_ORDER, column_order=MODEL_ORDER))
    _supp_panel_label(axes[2], "(c)")
    direction = direction_df.copy()
    if {"train_site_id", "test_site_id"}.issubset(direction.columns):
        direction["direction"] = direction["train_site_id"].astype(str).map(_display_label) + " → " + direction["test_site_id"].astype(str).map(_display_label)
    direction_col = "direction" if "direction" in direction.columns else ("site_id" if "site_id" in direction.columns else "test_site_id")
    source_frames.append(_supp_heatmap_panel(fig, axes[3], direction, row="feature_set", column=direction_col, value="AP_mean", row_order=FEATURE_ORDER, cbar_label="AP"))
    _supp_panel_label(axes[3], "(d)")
    fig.subplots_adjust(left=0.24, right=0.90, bottom=0.06, top=0.98)
    _export_supplementary_axes(
        fig=fig,
        axes=axes,
        composite_path=path,
        panels=[
            ("a", "Model ranking", source_frames[0].assign(supplementary_figure="S2", panel="a")),
            ("b", "Feature-set ranking", source_frames[1].assign(supplementary_figure="S2", panel="b")),
            ("c", "Feature by model heatmap", source_frames[2].assign(supplementary_figure="S2", panel="c")),
            ("d", "Transfer-direction feature heatmap", source_frames[3].assign(supplementary_figure="S2", panel="d")),
        ],
        options=options,
        plot_function="_save_supplementary_s2",
        metadata_extra={"validation_strategy": "cross_site"},
    )
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    source = pd.concat([frame.assign(panel=panel) for frame, panel in zip(source_frames, ["a", "b", "c", "d"]) if not frame.empty], ignore_index=True)
    source["supplementary_figure"] = "S2"
    _write_figure_source(path, source, options)


def _save_supplementary_s4(
    *,
    model_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_model_df: pd.DataFrame,
    path: Path,
    options: ReportOptions,
) -> None:
    """Save S4 integrated comparison composite."""

    plt = _require_matplotlib_pyplot()
    fig = plt.figure(figsize=(8.7, 16.5), constrained_layout=False)
    gs = fig.add_gridspec(5, 1, height_ratios=[1.25, 1.05, 1.8, 1.45, 2.2], hspace=0.55)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    metrics = [col for col in ["within_AP", "cross_AP", "spatial_AP"] if col in model_df.columns]
    source_frames: list[pd.DataFrame] = []
    source_frames.append(
        _supp_grouped_horizontal_ap(
            axes[0],
            model_df,
            category="model_id",
            metrics=metrics,
            order=MODEL_ORDER,
            ylabel="Model",
            show_legend=False,
        )
    )
    _supp_panel_label(axes[0], "(a)")
    source_frames.append(_supp_horizontal_ap_bar(axes[1], model_df, category="model_id", value="mean_AP", order=MODEL_ORDER, color="#374151"))
    _supp_panel_label(axes[1], "(b)")
    feature_metrics = [col for col in ["within_AP", "cross_AP", "spatial_AP"] if col in feature_df.columns]
    source_frames.append(
        _supp_grouped_horizontal_ap(
            axes[2],
            feature_df,
            category="feature_set",
            metrics=feature_metrics,
            order=FEATURE_ORDER,
            ylabel="Feature set",
            show_legend=False,
        )
    )
    _supp_panel_label(axes[2], "(c)")
    source_frames.append(_supp_horizontal_ap_bar(axes[3], feature_df, category="feature_set", value="mean_AP", order=FEATURE_ORDER, color="#374151", group_gap=True))
    _supp_panel_label(axes[3], "(d)")
    source_frames.append(_supp_heatmap_panel(fig, axes[4], feature_model_df, row="feature_set", column="model_id", value="mean_AP", row_order=FEATURE_ORDER, column_order=MODEL_ORDER, cbar_label="Integrated AP"))
    _supp_panel_label(axes[4], "(e)")
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=len(legend_handles),
            fontsize=9,
        )
    fig.subplots_adjust(left=0.24, right=0.90, bottom=0.075, top=0.985)
    panel_legends = []
    for panel_ax in [axes[0], axes[2]]:
        handles, labels = panel_ax.get_legend_handles_labels()
        if handles:
            panel_legends.append(
                panel_ax.legend(
                    handles,
                    labels,
                    frameon=False,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.30),
                    ncol=len(handles),
                    fontsize=8.5,
                )
            )
    _export_supplementary_axes(
        fig=fig,
        axes=axes,
        composite_path=path,
        panels=[
            ("a", "Validation-specific model comparison", source_frames[0].assign(supplementary_figure="S4", panel="a")),
            ("b", "Integrated model ranking", source_frames[1].assign(supplementary_figure="S4", panel="b")),
            ("c", "Validation-specific feature comparison", source_frames[2].assign(supplementary_figure="S4", panel="c")),
            ("d", "Integrated feature ranking", source_frames[3].assign(supplementary_figure="S4", panel="d")),
            ("e", "Integrated feature by model heatmap", source_frames[4].assign(supplementary_figure="S4", panel="e")),
        ],
        options=options,
        plot_function="_save_supplementary_s4",
        metadata_extra={"validation_strategy": "integrated"},
    )
    for legend in panel_legends:
        legend.remove()
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    source = pd.concat([frame.assign(panel=panel) for frame, panel in zip(source_frames, ["a", "b", "c", "d", "e"]) if not frame.empty], ignore_index=True)
    source["supplementary_figure"] = "S4"
    _write_figure_source(path, source, options)


def _save_supplementary_s5(
    *,
    feature_rank_df: pd.DataFrame,
    feature_model_rank_df: pd.DataFrame,
    path: Path,
    options: ReportOptions,
) -> None:
    """Save S5 rank-stability dumbbell plots."""

    plt = _require_matplotlib_pyplot()
    feature_lookup = {feature: idx for idx, feature in enumerate(FEATURE_ORDER)}
    model_lookup = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 18.5),
        constrained_layout=False,
        gridspec_kw={"height_ratios": [1.15, 4.1]},
    )
    source_a = _supp_dumbbell_rank(
        axes[0],
        feature_rank_df,
        label_func=lambda row: _display_label(row.get("feature_set", "")),
        order_key=lambda row: feature_lookup.get(str(row.get("feature_set", "")), len(feature_lookup)),
    )
    _supp_panel_label(axes[0], "(a)")
    source_b = _supp_dumbbell_rank(
        axes[1],
        feature_model_rank_df,
        label_func=_feature_model_label,
        top=28,
        order_key=lambda row: (
            feature_lookup.get(str(row.get("feature_set", "")), len(feature_lookup)),
            model_lookup.get(str(row.get("model_id", "")), len(model_lookup)),
        ),
    )
    axes[1].tick_params(axis="y", labelsize=7.8, pad=3)
    _supp_panel_label(axes[1], "(b)")
    fig.subplots_adjust(left=0.38, right=0.96, bottom=0.055, top=0.98, hspace=0.34)
    _export_supplementary_axes(
        fig=fig,
        axes=list(axes),
        composite_path=path,
        panels=[
            ("a", "Feature rank stability", source_a.assign(supplementary_figure="S5", panel="a")),
            ("b", "Feature by model rank stability", source_b.assign(supplementary_figure="S5", panel="b")),
        ],
        options=options,
        plot_function="_save_supplementary_s5",
        metadata_extra={"validation_strategy": "integrated"},
    )
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    source = pd.concat([source_a.assign(panel="a"), source_b.assign(panel="b")], ignore_index=True)
    source["supplementary_figure"] = "S5"
    _write_figure_source(path, source, options)


def _publication_site_plot_styles(
    site_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return display styles keyed by stable configured site identity."""

    requested = list(SITE_ORDER if site_ids is None else site_ids)
    ordered = [site_id for site_id in SITE_ORDER if site_id in requested]
    ordered.extend(site_id for site_id in requested if site_id not in ordered)
    if not ordered:
        return {}
    offsets = np.linspace(-0.14, 0.14, len(ordered)) if len(ordered) > 1 else np.asarray([0.0])
    markers = ("o", "s", "^", "D", "P", "X")
    point_colors = ("#4b5563", "#9ca3af", "#6b7280", "#d1d5db")
    bar_colors = ("#6b7280", "#b6bbc3", "#8b929c", "#d1d5db")
    styles: dict[str, dict[str, Any]] = {}
    for index, site_id in enumerate(ordered):
        hatch, alpha = PUBLICATION_SITE_STYLES.get(site_id, ("", 1.0))
        styles[site_id] = {
            "label": _display_label(site_id),
            "offset": float(offsets[index]),
            "marker": markers[index % len(markers)],
            "point_color": point_colors[index % len(point_colors)],
            "legend_point_color": "#6b7280",
            "bar_color": bar_colors[index % len(bar_colors)],
            "hatch": hatch,
            "alpha": alpha,
            "order": index,
        }
    return styles


def _save_supplementary_s6(
    *,
    rank_stability_table: pd.DataFrame,
    probability_stats: pd.DataFrame,
    binary_stats: pd.DataFrame,
    dirs: ReportDirs,
    path: Path,
    options: ReportOptions,
) -> None:
    """Save S6 final-candidate analysis composite."""

    plt = _require_matplotlib_pyplot()
    probability = _read_existing_or_current(
        probability_stats,
        dirs.table_source_maps / "probability_statistics.csv",
    )
    binary = _read_existing_or_current(
        binary_stats,
        dirs.table_source_maps / "binary_statistics.csv",
    )
    fig = plt.figure(figsize=(8.8, 12.8), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.45, 1.55, 1.25], hspace=0.52)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])

    candidate_metadata_frames = []
    candidate_metadata_columns = ["candidate_label", "feature_set", "model_id"]
    for frame in [probability, binary]:
        required = set(candidate_metadata_columns)
        if not frame.empty and required.issubset(frame.columns):
            candidate_metadata_frames.append(frame[candidate_metadata_columns].drop_duplicates())
    if not candidate_metadata_frames:
        raise ValueError("S6 requires candidate feature/model metadata from existing final-map statistics.")
    candidate_metadata = _supp_order_candidate_rows(
        pd.concat(candidate_metadata_frames, ignore_index=True).drop_duplicates()
    )
    candidate_metadata = candidate_metadata.drop_duplicates("candidate_label", keep="first")
    rank_columns = ["within_rank", "cross_rank", "spatial_rank"]
    required_rank_columns = {"entity_type", "entity_label", *rank_columns}
    if not required_rank_columns.issubset(rank_stability_table.columns):
        raise ValueError("S6 requires the publication rank-stability table for final candidates.")
    candidate_metadata["entity_label"] = candidate_metadata.apply(
        lambda row: _publication_entity_label("model_feature", row["model_id"], row["feature_set"]),
        axis=1,
    )
    candidate_metadata["candidate_display"] = candidate_metadata["candidate_label"].map(
        _publication_candidate_label
    )
    rank_rows = rank_stability_table[
        rank_stability_table["entity_type"].astype(str) == "model_feature"
    ]
    shift_source = candidate_metadata.merge(
        rank_rows,
        on="entity_label",
        how="left",
        validate="one_to_one",
    )
    shift_source = _supp_order_candidate_rows(shift_source)
    if shift_source.shape[0] != len(CANDIDATE_ORDER) or shift_source[rank_columns].isna().any().any():
        raise ValueError("Stored validation ranks were not found for all four final candidates.")
    validation_x = np.arange(3, dtype="float64")
    validation_labels = ["Within-site", "Cross-site", "Spatial block CV"]
    for row in shift_source.itertuples(index=False):
        candidate = str(row.candidate_label)
        ranks = np.asarray([row.within_rank, row.cross_rank, row.spatial_rank], dtype="float64")
        ax_a.plot(
            validation_x,
            ranks,
            color=_supp_candidate_color(candidate),
            marker="o",
            markersize=8.2 if candidate == PRIMARY_PUBLICATION_CANDIDATE else 6.5,
            linewidth=3.1 if candidate == PRIMARY_PUBLICATION_CANDIDATE else 1.8,
            markeredgecolor="#111827" if candidate == PRIMARY_PUBLICATION_CANDIDATE else "white",
            markeredgewidth=1.8 if candidate == PRIMARY_PUBLICATION_CANDIDATE else 0.6,
            label=_supp_candidate_label(candidate),
        )
    ax_a.set_xticks(validation_x)
    ax_a.set_xticklabels(validation_labels)
    ax_a.set_ylabel("Rank among 28 combinations\n(1 = highest AP)")
    ax_a.set_xlim(-0.12, 2.12)
    ax_a.set_ylim(float(shift_source[rank_columns].max().max()) + 1.5, 0.5)
    ax_a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2, fontsize=8.5)
    _supp_style_axis(ax_a, grid_axis="y")
    shift_source["data_semantics"] = "Stored rank of each selected candidate within the 28 feature-model combinations"
    _supp_panel_label(ax_a, "(a)")

    probability_source = pd.DataFrame()
    if not probability.empty and "candidate_label" in probability.columns:
        prob = probability.copy()
        positive_columns = [
            "positive_probability_mean",
            "positive_probability_median",
            "positive_probability_p25",
            "positive_probability_p75",
        ]
        for col in positive_columns:
            if col in prob.columns:
                prob[col] = pd.to_numeric(prob[col], errors="coerce")
        required_quantiles = {"site_id", *positive_columns}
        prob = _supp_order_candidate_rows(prob)
        if required_quantiles.issubset(prob.columns) and not prob.empty:
            candidate_y = {candidate: idx for idx, candidate in enumerate(CANDIDATE_ORDER)}
            site_styles = _publication_site_plot_styles()
            for row in prob.itertuples(index=False):
                candidate = str(row.candidate_label)
                site_id = str(row.site_id)
                if candidate not in candidate_y or site_id not in site_styles:
                    continue
                style = site_styles[site_id]
                y = candidate_y[candidate] + style["offset"]
                color = _supp_candidate_color(candidate)
                ax_b.hlines(
                    y,
                    float(row.positive_probability_p25),
                    float(row.positive_probability_p75),
                    color=color,
                    linewidth=5.0,
                    alpha=0.95,
                )
                ax_b.scatter(
                    float(row.positive_probability_median),
                    y,
                    marker=style["marker"],
                    s=46,
                    facecolor=style["point_color"],
                    edgecolor=color,
                    linewidth=1.2,
                    zorder=3,
                    clip_on=False,
                )
                ax_b.vlines(
                    float(row.positive_probability_mean),
                    y - 0.09,
                    y + 0.09,
                    color="#111827",
                    linewidth=1.1,
                    zorder=4,
                )
            from matplotlib.lines import Line2D

            site_handles = [
                Line2D(
                    [0], [0], marker=style["marker"], color="none",
                    markerfacecolor=style["legend_point_color"], markersize=6,
                    label=style["label"],
                )
                for style in site_styles.values()
            ]
            ax_b.set_yticks(np.arange(len(CANDIDATE_ORDER)))
            ax_b.set_yticklabels([_supp_candidate_label(candidate) for candidate in CANDIDATE_ORDER])
            ax_b.invert_yaxis()
            ax_b.set_xlabel("Habitat-pixel probability (p >= 0.5)")
            low = float(prob["positive_probability_p25"].min())
            high = float(prob["positive_probability_p75"].max())
            padding = max((high - low) * 0.10, 0.005)
            ax_b.set_xlim(max(0.5, low - padding), min(1.0, high + padding))
            ax_b.legend(handles=site_handles, frameon=False, loc="upper right", fontsize=8.5)
            probability_source = prob.copy()
            probability_source["candidate_order"] = probability_source["candidate_label"].map(
                {candidate: index + 1 for index, candidate in enumerate(CANDIDATE_ORDER)}
            )
            probability_source["site_order"] = probability_source["site_id"].map(
                {site: index + 1 for index, site in enumerate(SITE_ORDER)}
            )
            probability_source["data_semantics"] = (
                "Site-specific p25-p75 interval, median point, and mean tick among pixels "
                "classified as habitat at p >= 0.5; no site pooling"
            )
            probability_source = probability_source[
                [
                    "site_id",
                    "site_name",
                    "candidate_label",
                    "feature_set",
                    "model_id",
                    "candidate_order",
                    "site_order",
                    "positive_probability_mean",
                    "positive_probability_median",
                    "positive_probability_p25",
                    "positive_probability_p75",
                    "data_semantics",
                ]
            ].copy()
            _supp_style_axis(ax_b, grid_axis="x")
        else:
            ax_b.text(0.5, 0.5, "No probability stats", ha="center", va="center", transform=ax_b.transAxes)
            ax_b.set_axis_off()
    else:
        ax_b.text(0.5, 0.5, "No probability stats", ha="center", va="center", transform=ax_b.transAxes)
        ax_b.set_axis_off()
    _supp_panel_label(ax_b, "(b)")

    binary_source = pd.DataFrame()
    if not binary.empty and "candidate_label" in binary.columns:
        ratio_col = "positive_ratio_valid" if "positive_ratio_valid" in binary.columns else ("positive_ratio" if "positive_ratio" in binary.columns else None)
        if ratio_col:
            area = binary.copy()
            area[ratio_col] = pd.to_numeric(area[ratio_col], errors="coerce")
            area = _supp_order_candidate_rows(area.dropna(subset=[ratio_col]))
            x = np.arange(len(CANDIDATE_ORDER), dtype="float64")
            width = 0.34
            max_value = 0.0
            site_styles = _publication_site_plot_styles()
            for site_id, style in site_styles.items():
                site_rows = area[area["site_id"].astype(str) == site_id].set_index("candidate_label")
                values = np.asarray([site_rows.at[candidate, ratio_col] for candidate in CANDIDATE_ORDER], dtype="float64")
                positions = x + (style["order"] - (len(site_styles) - 1) / 2) * width
                bars = ax_c.bar(
                    positions,
                    values,
                    width=width,
                    color=[_supp_candidate_color(candidate) for candidate in CANDIDATE_ORDER],
                    alpha=style["alpha"],
                    hatch=style["hatch"],
                    edgecolor="#374151",
                    linewidth=0.5,
                    label=style["label"],
                )
                max_value = max(max_value, float(np.nanmax(values)))
                for bar, value in zip(bars, values):
                    ax_c.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{value:.1%}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            ax_c.set_xticks(x)
            ax_c.set_xticklabels([_candidate_compact_label(candidate) for candidate in CANDIDATE_ORDER], rotation=0)
            from matplotlib.ticker import PercentFormatter

            ax_c.set_ylabel("Predicted habitat extent\n(% of valid mapped area)")
            ax_c.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax_c.set_ylim(0, max(max_value * 1.32, 0.01))
            from matplotlib.patches import Patch

            site_handles = [
                Patch(
                    facecolor=style["bar_color"], edgecolor="#374151",
                    hatch=style["hatch"], label=style["label"],
                )
                for style in site_styles.values()
            ]
            ax_c.legend(handles=site_handles, frameon=False, loc="upper center", ncol=2, fontsize=8.5)
            _supp_style_axis(ax_c, grid_axis="y")
            binary_source = area.copy()
            binary_source["data_semantics"] = "Site-specific positive pixel ratio among valid pixels at the stored binary threshold"
        else:
            ax_c.text(0.5, 0.5, "No positive-ratio column", ha="center", va="center", transform=ax_c.transAxes)
            ax_c.set_axis_off()
    else:
        ax_c.text(0.5, 0.5, "No binary stats", ha="center", va="center", transform=ax_c.transAxes)
        ax_c.set_axis_off()
    _supp_panel_label(ax_c, "(c)")

    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.075, top=0.965)
    _export_supplementary_axes(
        fig=fig,
        axes=[ax_a, ax_b, ax_c],
        composite_path=path,
        panels=[
            ("a", "Final-candidate rank shift", shift_source.assign(supplementary_figure="S6", panel="a")),
            ("b", "Positive-pixel probability distribution", probability_source.assign(supplementary_figure="S6", panel="b")),
            ("c", "Predicted habitat ratio of valid area", binary_source.assign(supplementary_figure="S6", panel="c")),
        ],
        options=options,
        plot_function="_save_supplementary_s6",
        metadata_extra={"validation_strategy": "final_candidate_comparison", "threshold": options.threshold},
    )
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    source = pd.concat(
        [
            shift_source.assign(panel="a"),
            probability_source.assign(panel="b"),
            binary_source.assign(panel="c"),
        ],
        ignore_index=True,
    )
    source["supplementary_figure"] = "S6"
    _write_figure_source(path, source, options)


def _importance_summary_by_site(
    feature_importance: pd.DataFrame,
    site_id: str | None = None,
    feature_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return mean feature importance for a site or all sites."""

    data = _normalized_final_model_importance(feature_importance)
    if site_id is not None and "site_id" in data.columns:
        data = data[data["site_id"].astype(str) == site_id]
    if data.empty:
        return pd.DataFrame()
    data["importance"] = pd.to_numeric(data["importance"], errors="coerce")
    out = data.groupby("feature", dropna=False)["importance"].mean().reset_index()
    if feature_order is None:
        out = out.sort_values("importance", ascending=False)
    else:
        lookup = {str(feature): idx for idx, feature in enumerate(feature_order)}
        out["_feature_order"] = out["feature"].astype(str).map(lookup)
        out = out.dropna(subset=["_feature_order"]).sort_values("_feature_order", kind="stable")
        out = out.drop(columns="_feature_order")
    out["feature_group"] = out["feature"].map(_feature_group)
    out["feature_label"] = out["feature"].map(_display_label)
    if feature_order is not None:
        order = {str(feature): index + 1 for index, feature in enumerate(feature_order)}
        out["selection_order"] = out["feature"].astype(str).map(order).astype("Int64")
        out["selection_rule"] = "Top 12 by combined mean importance; common order across panels"
    return out


def _save_supplementary_s7(
    *,
    feature_importance: pd.DataFrame,
    path: Path,
    options: ReportOptions,
) -> None:
    """Save S7 feature-importance comparison composite."""

    plt = _require_matplotlib_pyplot()
    fig = plt.figure(figsize=(8.8, 15.2), constrained_layout=False)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.25, 1.25, 1.25, 1.25], hspace=0.48)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    sources: list[pd.DataFrame] = []
    combined_summary = _importance_summary_by_site(feature_importance)
    feature_order = combined_summary.head(12)["feature"].astype(str).tolist()
    panel_specs = _publication_panels("fig_s07_feature_importance")
    site_panels = panel_specs[:2]
    site_ids = [str(panel["selection"]["site"]) for panel in site_panels]
    panel_summaries = [
        *[_importance_summary_by_site(feature_importance, site_id, feature_order) for site_id in site_ids],
        _importance_summary_by_site(feature_importance, None, feature_order),
    ]
    finite_maxima = [
        float(pd.to_numeric(summary["importance"], errors="coerce").max())
        for summary in panel_summaries
        if not summary.empty
    ]
    common_xlim = max(max(finite_maxima, default=0.05) * 1.19, 0.05)
    configured_panels = [
        (
            axes[index],
            site_ids[index],
            f"({site_panels[index]['letter']})",
            str(site_panels[index]["title"]).removesuffix(" feature importance"),
            panel_summaries[index],
        )
        for index in range(2)
    ]
    configured_panels.append((axes[2], None, "(c)", "Combined", panel_summaries[2]))
    for ax, site_id, panel, panel_title, summary in configured_panels:
        if summary.empty:
            ax.text(0.5, 0.5, "No feature importance", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            colors = [FEATURE_GROUP_COLORS.get(group, "#6b7280") for group in summary["feature_group"]]
            y = np.arange(summary.shape[0])
            vals = summary["importance"].to_numpy(dtype="float64")
            ax.barh(y, vals, color=colors)
            ax.set_yticks(y)
            ax.set_yticklabels(summary["feature_label"].astype(str).tolist())
            ax.invert_yaxis()
            ax.set_xlabel("Mean feature importance")
            ax.set_xlim(0, common_xlim)
            ax.set_title(panel_title, fontsize=11, pad=7)
            for yi, val in zip(y, vals):
                ax.text(val, yi, f" {val:.3f}", va="center", ha="left", fontsize=9)
            from matplotlib.patches import Patch

            present_groups = FEATURE_GROUP_ORDER[:4]
            handles = [
                Patch(facecolor=FEATURE_GROUP_COLORS[group], edgecolor="none", label=group)
                for group in present_groups
            ]
            ax.legend(
                handles=handles,
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8.5,
            )
            _supp_style_axis(ax, grid_axis="x")
            sources.append(summary.assign(panel=panel.strip("()"), site_id="Combined" if site_id is None else site_id))
        _supp_panel_label(ax, panel)
    data = _normalized_final_model_importance(feature_importance)
    if not data.empty and {"feature", "importance"}.issubset(data.columns):
        data["importance"] = pd.to_numeric(data["importance"], errors="coerce")
        grouped = [
            data.loc[data["feature"].astype(str) == feature, "importance"].dropna().to_numpy()
            for feature in feature_order
        ]
        if grouped and all(values.size >= 2 for values in grouped):
            axes[3].boxplot(
                grouped,
                tick_labels=[_short_label(_display_label(item), 22) for item in feature_order],
                showfliers=False,
                vert=False,
            )
            axes[3].set_xlabel("Feature importance across final models")
            axes[3].set_title("Final-model importance distribution", fontsize=11, pad=7)
            axes[3].invert_yaxis()
            _supp_style_axis(axes[3], grid_axis="x")
            distribution_source = data[data["feature"].astype(str).isin(feature_order)].copy()
            distribution_source["feature_label"] = distribution_source["feature"].map(_display_label)
            distribution_source["feature_group"] = distribution_source["feature"].map(_feature_group)
            selection_order = {feature: index + 1 for index, feature in enumerate(feature_order)}
            distribution_source["selection_order"] = distribution_source["feature"].map(selection_order).astype("Int64")
            distribution_source["selection_rule"] = (
                "Top 12 by combined mean importance; common order across panels"
            )
            distribution_source["distribution_unit"] = "Final retrained model (site x candidate); seed 42"
            distribution_source["distribution_observation_id"] = (
                distribution_source["site_id"].astype(str)
                + " / "
                + distribution_source["candidate_label"].astype(str)
            )
            counts = distribution_source.groupby("feature")["importance"].transform("count")
            distribution_source["n_observations_for_feature"] = counts.astype(int)
            sources.append(distribution_source.assign(panel="d"))
        else:
            axes[3].text(0.5, 0.5, "Insufficient repeated final-model values", ha="center", va="center", transform=axes[3].transAxes)
            axes[3].set_axis_off()
    else:
        axes[3].text(0.5, 0.5, "No feature importance", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_axis_off()
    _supp_panel_label(axes[3], "(d)")
    fig.subplots_adjust(left=0.31, right=0.73, bottom=0.065, top=0.98)
    panel_sources = {
        str(frame["panel"].iloc[0]): frame
        for frame in sources
        if not frame.empty and "panel" in frame.columns
    }
    _export_supplementary_axes(
        fig=fig,
        axes=axes,
        composite_path=path,
        panels=[
            ("a", "Hujeong feature importance", panel_sources.get("a", pd.DataFrame())),
            ("b", "Bongpyeong feature importance", panel_sources.get("b", pd.DataFrame())),
            ("c", "Combined feature importance", panel_sources.get("c", pd.DataFrame())),
            ("d", "Final-model feature-importance distribution", panel_sources.get("d", pd.DataFrame())),
        ],
        options=options,
        plot_function="_save_supplementary_s7",
        metadata_extra={"model": "final retrained models", "seed": 42},
    )
    _save_figure_outputs(fig, path, dpi=options.dpi, save_vector=options.save_vector)
    plt.close(fig)
    source = pd.concat(sources, ignore_index=True) if sources else pd.DataFrame()
    source["supplementary_figure"] = "S7"
    _write_figure_source(path, source, options)


def _read_supp_representative_predictions(
    path: Path,
    *,
    site_id: str,
    validation_strategy: str,
    train_site_id: str,
    seed: int,
    fold_id: int | None = None,
) -> pd.DataFrame:
    """Read and strictly validate one representative prediction file."""

    data = _read_csv(path, path.name)
    required = {"site_id", "feature_set", "model_id", "seed", "split", "sample_index", "y_true", "y_score"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Representative prediction file is missing columns {missing}: {path}")
    representative = _representative_run(site_id)
    expected_values = {
        "site_id": site_id,
        "feature_set": representative["feature_set"],
        "model_id": representative["model"],
        "seed": seed,
        "split": "test",
    }
    for column, expected in expected_values.items():
        values = data[column].dropna().unique().tolist()
        if values != [expected]:
            raise ValueError(f"Unexpected {column} values {values}; expected {expected!r}: {path}")
    if validation_strategy == "cross_site":
        cross_expected = {
            "validation_type": "cross_site",
            "train_site_id": train_site_id,
            "test_site_id": site_id,
        }
        for column, expected in cross_expected.items():
            if column not in data.columns or data[column].dropna().unique().tolist() != [expected]:
                raise ValueError(f"Unexpected {column}; expected {expected!r}: {path}")
    elif validation_strategy == "spatial_block_cv":
        if (
            "validation_type" not in data.columns
            or data["validation_type"].dropna().unique().tolist() != ["spatial_block_cv"]
        ):
            raise ValueError(f"Unexpected validation_type; expected 'spatial_block_cv': {path}")
    if fold_id is not None:
        if "fold_id" not in data.columns or data["fold_id"].dropna().unique().tolist() != [fold_id]:
            raise ValueError(f"Unexpected fold_id; expected {fold_id}: {path}")
    if data["sample_index"].duplicated().any():
        raise ValueError(f"Duplicate sample_index values within representative prediction file: {path}")
    y_true = pd.to_numeric(data["y_true"], errors="coerce")
    y_score = pd.to_numeric(data["y_score"], errors="coerce")
    if y_true.isna().any() or not set(y_true.unique()).issubset({0, 1}):
        raise ValueError(f"Invalid y_true values in representative prediction file: {path}")
    if y_score.isna().any() or (~y_score.between(0, 1)).any():
        raise ValueError(f"Invalid y_score values in representative prediction file: {path}")
    data = data.copy()
    data["y_true"] = y_true.astype(int)
    data["y_score"] = y_score.astype(float)
    data["source_prediction_path"] = path.name
    return data


def _supp_representative_paths(output_root: Path, target_site: str) -> list[dict[str, Any]]:
    """Return configured representative validation inputs for one target site."""

    run = _representative_run(target_site)
    train_site = str(run["train_site"])
    feature_set = str(run["feature_set"])
    model_id = str(run["model"])
    seed = int(run["seed"])
    folds = list(run["spatial_folds"])
    seed_dir = f"seed_{seed}"
    within_root = output_root / "within_site" / target_site / feature_set / model_id / seed_dir
    spatial_root = output_root / "spatial_block_cv" / target_site / feature_set / model_id / seed_dir
    cross_root = (
        output_root
        / "cross_site"
        / f"train_{train_site}_test_{target_site}"
        / feature_set
        / model_id
        / seed_dir
    )
    return [
        {
            "validation_strategy": "within_site",
            "column_label": f"Within-site\n{_display_label(target_site)}",
            "prediction_paths": [within_root / "predictions.csv"],
            "evaluation_path": within_root / "evaluation_metrics.csv",
            "aggregation_method": f"Single stored seed-{seed} test run",
            "train_site_id": target_site,
        },
        {
            "validation_strategy": "spatial_block_cv",
            "column_label": f"Spatial block CV\n{_display_label(target_site)}\n5-fold OOF",
            "prediction_paths": [spatial_root / f"fold_{fold}" / "predictions.csv" for fold in folds],
            "evaluation_path": None,
            "aggregation_method": str(run["spatial_aggregation_method"]),
            "train_site_id": target_site,
        },
        {
            "validation_strategy": "cross_site",
            "column_label": (
                f"Cross-site\n{_display_label(train_site)} →\n{_display_label(target_site)}"
            ),
            "prediction_paths": [cross_root / "predictions.csv"],
            "evaluation_path": cross_root / "evaluation_metrics.csv",
            "aggregation_method": f"Single stored seed-{seed} transfer test run",
            "train_site_id": train_site,
        },
    ]


def _supp_even_curve_sample(data: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    """Evenly reduce plotted curve rows while retaining both endpoints."""

    if data.shape[0] <= max_points:
        return data.copy()
    indices = np.linspace(0, data.shape[0] - 1, max_points).round().astype(int)
    return data.iloc[np.unique(indices)].copy()


def _supp_curves_from_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, int, int]:
    """Compute metrics and curves once from one representative prediction set."""

    try:
        from sklearn.metrics import (
            average_precision_score,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )
    except ModuleNotFoundError as exc:
        raise ImportError("scikit-learn is required to create representative validation curves.") from exc
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_score = predictions["y_score"].to_numpy(dtype="float64")
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    pr_full = pd.DataFrame(
        {
            "recall": recall,
            "precision": precision,
            "curve_threshold": np.append(pr_thresholds, np.nan),
        }
    )
    roc_full = pd.DataFrame(
        {"fpr": fpr, "tpr": tpr, "curve_threshold": roc_thresholds}
    )
    return (
        _supp_even_curve_sample(pr_full),
        _supp_even_curve_sample(roc_full),
        float(average_precision_score(y_true, y_score)),
        float(roc_auc_score(y_true, y_score)),
        int(pr_full.shape[0]),
        int(roc_full.shape[0]),
    )


def _supp_validate_stored_probability_metrics(
    path: Path | None,
    *,
    average_precision: float,
    roc_auc: float,
    tolerance: float = 1e-12,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Confirm raw-prediction AP/AUC agree with the stored run metrics."""

    if path is None:
        return None, None, None, None
    metrics = _read_csv(path, "Representative evaluation metrics")
    if metrics.shape[0] != 1 or not {"AP", "AUC"}.issubset(metrics.columns):
        raise ValueError(f"Unexpected representative evaluation metric schema: {path}")
    stored_ap = float(metrics.iloc[0]["AP"])
    stored_auc = float(metrics.iloc[0]["AUC"])
    ap_difference = abs(average_precision - stored_ap)
    auc_difference = abs(roc_auc - stored_auc)
    if ap_difference > tolerance or auc_difference > tolerance:
        raise ValueError(
            "Raw prediction metrics do not match stored evaluation metrics: "
            f"AP difference={ap_difference}, AUC difference={auc_difference}: {path}"
        )
    return stored_ap, stored_auc, ap_difference, auc_difference


def _supp_confusion_from_predictions(
    predictions: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return count and actual-class normalized confusion matrices."""

    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_pred = predictions["y_score"].to_numpy(dtype="float64") >= threshold
    tn = int(np.sum((y_true == 0) & (~y_pred)))
    fp = int(np.sum((y_true == 0) & y_pred))
    fn = int(np.sum((y_true == 1) & (~y_pred)))
    tp = int(np.sum((y_true == 1) & y_pred))
    counts = np.asarray([[tn, fp], [fn, tp]], dtype="float64")
    row_totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0) * 100.0
    rows = []
    class_labels = {0: "Non-habitat", 1: "Habitat"}
    for actual in range(2):
        for predicted in range(2):
            rows.append(
                {
                    "actual_class_id": actual,
                    "actual_class": class_labels[actual],
                    "predicted_class_id": predicted,
                    "predicted_class": class_labels[predicted],
                    "count": int(counts[actual, predicted]),
                    "actual_class_percentage": float(percentages[actual, predicted]),
                    "threshold": threshold,
                }
            )
    return counts, percentages, pd.DataFrame(rows)


def _plot_supp_representative_confusion(
    ax: Any,
    percentages: np.ndarray,
) -> Any:
    """Draw one actual-class normalized representative confusion matrix."""

    image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    for row in range(2):
        for col in range(2):
            value = percentages[row, col]
            color = "white" if value >= 55 else "#111827"
            ax.text(
                col,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=9.5,
                color=color,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Background", "Vegetation"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Background", "Vegetation"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Reference class")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    return image


def _plot_supp_representative_curve(
    ax: Any,
    curve: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color: str,
    roc_reference: bool = False,
) -> None:
    """Draw one stored or pooled representative validation curve."""

    x = pd.to_numeric(curve[x_col], errors="coerce")
    y = pd.to_numeric(curve[y_col], errors="coerce")
    valid = x.notna() & y.notna()
    ax.plot(x[valid], y[valid], color=color, linewidth=2.2)
    if roc_reference:
        ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _supp_style_axis(ax, grid_axis=None)
    ax.grid(color="#9ca3af", alpha=0.22, linewidth=0.6)


def _write_supplementary_validation_panels(
    *,
    source_frames: Sequence[pd.DataFrame],
    composite_path: Path,
    target_site: str,
    supplementary_figure: str,
    options: ReportOptions,
) -> None:
    """Render S8/S9 panels from the exact composite source frames."""

    plt = _require_matplotlib_pyplot()
    representative = _representative_run(target_site)
    sources = {
        str(frame["panel"].iloc[0]): frame.copy()
        for frame in source_frames
        if not frame.empty and "panel" in frame.columns
    }
    strategy_colors = {
        item["id"]: item["color"]
        for item in configured_publication_display(PUBLICATION_CONFIG, "validations")
    }
    metric_titles = {
        "a": "confusion matrix",
        "b": "confusion matrix",
        "c": "confusion matrix",
        "d": "precision-recall curve",
        "e": "precision-recall curve",
        "f": "precision-recall curve",
        "g": "ROC curve",
        "h": "ROC curve",
        "i": "ROC curve",
    }
    panel_root = composite_path.parent.parent / "panels" / composite_path.stem
    panel_root.mkdir(parents=True, exist_ok=True)
    if options.overwrite:
        for existing in panel_root.iterdir():
            if existing.is_file():
                existing.unlink()
    for letter in "abcdefghi":
        source = sources.get(letter)
        if source is None or source.empty:
            raise ValueError(f"Missing {supplementary_figure} panel ({letter}) source data.")
        first = source.iloc[0]
        validation_strategy = str(first["validation_strategy"])
        if validation_strategy == "within_site":
            strategy_title = f"Within-site {_display_label(target_site)}"
        elif validation_strategy == "spatial_block_cv":
            strategy_title = f"Spatial block CV {_display_label(target_site)} (5-fold OOF)"
        else:
            strategy_title = (
                f"{_display_label(str(first['train_site_id']))} → "
                f"{_display_label(target_site)} cross-site"
            )
        panel_title = f"{strategy_title} {metric_titles[letter]}"
        fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=False)
        fig.patch.set_facecolor("white")
        if str(first["source_type"]) == "confusion_matrix":
            percentages = np.zeros((2, 2), dtype="float64")
            for row in source.itertuples(index=False):
                percentages[int(row.actual_class_id), int(row.predicted_class_id)] = float(
                    row.actual_class_percentage
                )
            image = _plot_supp_representative_confusion(ax, percentages)
            colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.05)
            colorbar.set_label("Actual-class percentage (%)")
            fig.subplots_adjust(left=0.23, right=0.86, bottom=0.18, top=0.84)
        elif str(first["source_type"]) == "precision_recall_curve":
            _plot_supp_representative_curve(
                ax,
                source,
                x_col="recall",
                y_col="precision",
                color=strategy_colors[validation_strategy],
            )
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.text(
                0.96,
                0.06,
                f"AP = {float(first['average_precision']):.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
            )
            fig.subplots_adjust(left=0.18, right=0.96, bottom=0.17, top=0.84)
        else:
            _plot_supp_representative_curve(
                ax,
                source,
                x_col="fpr",
                y_col="tpr",
                color=strategy_colors[validation_strategy],
                roc_reference=True,
            )
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.text(
                0.96,
                0.06,
                f"AUC = {float(first['roc_auc']):.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
            )
            fig.subplots_adjust(left=0.18, right=0.96, bottom=0.17, top=0.84)
        ax.set_title(
            f"({letter}) {metric_titles[letter].capitalize()}\n{strategy_title}",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )
        stem = _supplementary_panel_stem(composite_path.stem, letter, panel_title)
        path = panel_root / f"{stem}.png"
        _save_figure_outputs(fig, path, dpi=PANEL_DPI, save_vector=options.save_vector)
        plt.close(fig)
        _write_figure_source(path, source, options)
        if options.generate_captions:
            _write_sidecar_text(
                path.with_name(path.stem + "_caption.txt"),
                f"Panel ({letter}). {panel_title}.\n",
                options.overwrite,
            )
        if options.generate_metadata:
            _write_figure_metadata(
                path,
                parent_figure=composite_path.stem,
                panel_letter=letter,
                panel_title=panel_title,
                official_or_auxiliary="official_panel",
                plot_function="_write_supplementary_validation_panels",
                source_data=source,
                options=options,
                extra={
                    "model": representative["model"],
                    "feature_set": representative["feature_set"],
                    "seed": representative["seed"],
                    "target_site": target_site,
                    "validation_strategy": validation_strategy,
                    "threshold": 0.5,
                    "fold_or_pooled_oof": first.get("fold_or_pooled_oof", "NA"),
                    "transfer_direction": first.get("transfer_direction", "NA"),
                },
            )


def _save_supplementary_validation_site(
    *,
    output_root: Path,
    target_site: str,
    supplementary_figure: str,
    path: Path,
    options: ReportOptions,
) -> None:
    """Save one site-specific 3x3 representative validation composite."""

    plt = _require_matplotlib_pyplot()
    threshold = 0.5
    representative = _representative_run(target_site)
    representative_seed = int(representative["seed"])
    representative_feature = str(representative["feature_set"])
    representative_model = str(representative["model"])
    strategy_colors = {
        item["id"]: item["color"]
        for item in configured_publication_display(PUBLICATION_CONFIG, "validations")
    }
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.2), constrained_layout=False)
    source_frames: list[pd.DataFrame] = []
    metadata_blocks: list[str] = []
    confusion_image = None
    for col, spec in enumerate(_supp_representative_paths(output_root, target_site)):
        prediction_frames = []
        fold_sample_counts: list[str] = []
        for fold_index, prediction_path in enumerate(spec["prediction_paths"]):
            fold_id = fold_index if spec["validation_strategy"] == "spatial_block_cv" else None
            prediction_frame = _read_supp_representative_predictions(
                prediction_path,
                site_id=target_site,
                validation_strategy=spec["validation_strategy"],
                train_site_id=spec["train_site_id"],
                seed=representative_seed,
                fold_id=fold_id,
            )
            prediction_frames.append(prediction_frame)
            fold_name = f"fold_{fold_id}" if fold_id is not None else "single_test_run"
            fold_sample_counts.append(f"{fold_name}:{prediction_frame.shape[0]}")
        predictions = pd.concat(prediction_frames, ignore_index=True)
        if predictions["sample_index"].duplicated().any():
            raise ValueError(
                f"Duplicate sample_index values across representative {spec['validation_strategy']} predictions for {target_site}."
            )
        counts, percentages, confusion_source = _supp_confusion_from_predictions(predictions, threshold=threshold)
        confusion_image = _plot_supp_representative_confusion(axes[0, col], percentages)
        axes[0, col].set_title(spec["column_label"], fontsize=10.5, fontweight="bold", pad=9)
        axes[0, col].text(
            -0.12,
            1.12,
            f"({chr(97 + col)})",
            transform=axes[0, col].transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )

        pr_curve, roc_curve, average_precision, roc_auc, full_pr_points, full_roc_points = (
            _supp_curves_from_predictions(predictions)
        )
        stored_ap, stored_auc, ap_difference, auc_difference = (
            _supp_validate_stored_probability_metrics(
                spec["evaluation_path"],
                average_precision=average_precision,
                roc_auc=roc_auc,
            )
        )
        prediction_paths = "; ".join(str(item.resolve()) for item in spec["prediction_paths"])
        curve_method = (
            "Pooled five-fold spatial OOF predictions"
            if spec["validation_strategy"] == "spatial_block_cv"
            else "Single seed-42 raw test prediction"
        )

        color = strategy_colors[spec["validation_strategy"]]
        _plot_supp_representative_curve(
            axes[1, col],
            pr_curve,
            x_col="recall",
            y_col="precision",
            color=color,
        )
        axes[1, col].set_xlabel("Recall")
        axes[1, col].set_ylabel("Precision" if col == 0 else "")
        axes[1, col].text(
            0.96,
            0.06,
            f"AP = {average_precision:.3f}",
            transform=axes[1, col].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
        _supp_panel_label(axes[1, col], f"({chr(100 + col)})")
        _plot_supp_representative_curve(
            axes[2, col],
            roc_curve,
            x_col="fpr",
            y_col="tpr",
            color=color,
            roc_reference=True,
        )
        axes[2, col].set_xlabel("False positive rate")
        axes[2, col].set_ylabel("True positive rate" if col == 0 else "")
        axes[2, col].text(
            0.96,
            0.06,
            f"AUC = {roc_auc:.3f}",
            transform=axes[2, col].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
        _supp_panel_label(axes[2, col], f"({chr(103 + col)})")

        n_positive = int((predictions["y_true"] == 1).sum())
        n_negative = int((predictions["y_true"] == 0).sum())
        tn, fp, fn, tp = (int(counts[0, 0]), int(counts[0, 1]), int(counts[1, 0]), int(counts[1, 1]))
        transfer_direction = (
            f"{spec['train_site_id']} -> {target_site}"
            if spec["validation_strategy"] == "cross_site"
            else f"within {target_site}"
        )
        common_metadata = {
            "supplementary_figure": supplementary_figure,
            "validation_strategy": spec["validation_strategy"],
            "target_site": target_site,
            "transfer_direction": transfer_direction,
            "train_site_id": spec["train_site_id"],
            "feature_set": representative_feature,
            "model_id": representative_model,
            "seed": representative_seed,
            "fold_scope": (
                f"fold_{min(representative['spatial_folds'])}-fold_{max(representative['spatial_folds'])}"
                if spec["validation_strategy"] == "spatial_block_cv" else "not_applicable"
            ),
            "fold_or_pooled_oof": "pooled_oof" if spec["validation_strategy"] == "spatial_block_cv" else "single_test_run",
            "pooled_oof": spec["validation_strategy"] == "spatial_block_cv",
            "fold_sample_counts": "; ".join(fold_sample_counts),
            "aggregation_method": spec["aggregation_method"],
            "source_prediction_path": prediction_paths,
            "sample_id_column": "sample_index",
            "y_true_column": "y_true",
            "probability_column": "y_score",
            "n_samples": int(predictions.shape[0]),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "duplicate_sample_count": 0,
            "threshold": threshold,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "normalized_TN_pct": float(percentages[0, 0]),
            "normalized_FP_pct": float(percentages[0, 1]),
            "normalized_FN_pct": float(percentages[1, 0]),
            "normalized_TP_pct": float(percentages[1, 1]),
            "average_precision": average_precision,
            "roc_auc": roc_auc,
            "stored_evaluation_metrics_path": (
                str(spec["evaluation_path"].resolve()) if spec["evaluation_path"] is not None else "not_applicable"
            ),
            "stored_average_precision": stored_ap,
            "stored_roc_auc": stored_auc,
            "stored_AP_absolute_difference": ap_difference,
            "stored_AUC_absolute_difference": auc_difference,
            "confidence_interval": "none",
        }
        source_frames.append(
            confusion_source.assign(
                panel=chr(97 + col),
                source_type="confusion_matrix",
                **common_metadata,
            )
        )
        source_frames.append(
            pr_curve.assign(
                panel=chr(100 + col),
                source_type="precision_recall_curve",
                curve_method=curve_method,
                full_curve_point_count=full_pr_points,
                saved_curve_point_count=int(pr_curve.shape[0]),
                **common_metadata,
            )
        )
        source_frames.append(
            roc_curve.assign(
                panel=chr(103 + col),
                source_type="roc_curve",
                curve_method=curve_method,
                full_curve_point_count=full_roc_points,
                saved_curve_point_count=int(roc_curve.shape[0]),
                **common_metadata,
            )
        )
        metadata_blocks.append(
            "\n".join(
                [
                    f"validation_strategy: {spec['validation_strategy']}",
                    f"transfer_direction: {transfer_direction}",
                    f"source_prediction_path: {prediction_paths}",
                    f"fold_sample_counts: {'; '.join(fold_sample_counts)}",
                    f"n_samples: {predictions.shape[0]}",
                    f"n_positive: {n_positive}",
                    f"n_negative: {n_negative}",
                    "duplicate_sample_count: 0",
                    f"average_precision: {average_precision:.15g}",
                    f"roc_auc: {roc_auc:.15g}",
                ]
            )
        )

    if confusion_image is not None:
        colorbar_ax = fig.add_axes([0.925, 0.692, 0.014, 0.205])
        colorbar = fig.colorbar(confusion_image, cax=colorbar_ax)
        colorbar.set_label("Actual-class percentage (%)", fontsize=9.5)
        colorbar.ax.tick_params(labelsize=8.5)
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.08, top=0.88, wspace=0.40, hspace=0.36)
    source_by_letter = {
        str(frame["panel"].iloc[0]): frame
        for frame in source_frames
        if not frame.empty
    }
    figure_spec = next(
        item
        for item in configured_figure_specs(PUBLICATION_CONFIG)
        if str(item["number"]).replace("Figure ", "") == supplementary_figure
    )
    configured_panels = {str(item["letter"]): item for item in figure_spec.get("panels", [])}
    _export_supplementary_axes(
        fig=fig,
        axes=list(axes.ravel()),
        composite_path=path,
        panels=[
            (
                letter,
                str(configured_panels[letter]["title"]),
                source_by_letter[letter],
            )
            for letter in "abcdefghi"
        ],
        options=options,
        plot_function="_save_supplementary_validation_site",
        metadata_extra={
            "model": representative_model,
            "feature_set": representative_feature,
            "seed": representative_seed,
            "target_site": target_site,
            "threshold": threshold,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=options.dpi, facecolor="white")
    if options.save_vector:
        fig.savefig(path.with_suffix(".pdf"), facecolor="white")
        fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    source = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in source_frames if not frame.empty],
        ignore_index=True,
    )
    source["confusion_threshold"] = threshold
    source["confusion_threshold_basis"] = "Project/final-map fixed probability threshold; no threshold optimization"
    _write_figure_source(path, source, options)
    if options.generate_captions:
        cross_train = _display_label(representative["train_site"])
        target_name = _display_label(target_site)
        caption = (
            f"Figure {supplementary_figure}. Representative validation performance for {target_name}. "
            f"All panels use {_display_label(representative['model'])} with the "
            f"{_display_label(representative['feature_set'])} feature set and seed "
            f"{representative['seed']}. "
            f"Columns show the within-site held-out test subset, five-fold pooled spatial "
            f"out-of-fold predictions, and cross-site transfer from {cross_train} to {target_name} "
            "using the complete target-site dataset. Confusion matrices are "
            "normalized within actual class using p >= 0.5. Precision-recall and ROC curves are "
            "computed from the same raw predictions as their corresponding confusion matrix. "
            "No bootstrap or confidence-interval band is shown.\n"
        )
        _write_sidecar_text(path.with_name(path.stem + "_caption.txt"), caption, options.overwrite)
    if options.generate_metadata:
        metadata = (
            f"supplementary_figure: {supplementary_figure}\n"
            f"target_site: {target_site}\n"
            f"model_id: {representative['model']}\n"
            f"feature_set: {representative['feature_set']}\n"
            f"seed: {representative['seed']}\n"
            "confusion_threshold: 0.5\n"
            "confusion_normalization: actual class rows\n"
            "spatial_cv_method: five-fold pooled out-of-fold prediction\n"
            "curve_source: raw predictions.csv\n"
            "confidence_interval: none\n"
            f"generated_time: {FIGURE_GENERATED_AT}\n"
            f"git_commit: {FIGURE_GIT_COMMIT}\n\n"
            + "\n\n".join(metadata_blocks)
            + "\n"
        )
        _write_sidecar_text(path.with_name(path.stem + "_metadata.txt"), metadata, options.overwrite)


def _write_configured_figure(
    spec: Mapping[str, Any],
    *,
    dirs: ReportDirs,
    options: ReportOptions,
    output_root: Path,
    analysis: Mapping[str, pd.DataFrame],
    feature_configs: Mapping[str, Mapping[str, Any]],
    train_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    probability_stats: pd.DataFrame,
    binary_stats: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Dispatch one configured Figure to an allowlisted existing builder."""

    root = dirs.main_composites if spec["role"] == "main" else dirs.supplementary_composites
    path = root / f"{spec['stem']}.png"
    builder = spec["builder"]
    if builder == "study_areas":
        _save_study_area_figure(train_summary, probability_stats, output_root, path, options, warnings)
    elif builder == "workflow_validation":
        _save_workflow_validation_figure(path, options)
        _write_figure_source(
            path,
            pd.DataFrame(
                {
                    "node": [
                        "MBES acquisition", "Preprocessing", "Feature generation", "Labeled samples",
                        "Model x feature matrix", "Validation strategies",
                        "Integrated candidate selection", "Final retraining and mapping",
                    ],
                    "order": list(range(1, 9)),
                }
            ),
            options,
        )
    elif builder == "mbes_feature_definitions":
        _save_feature_definition_figure(feature_configs, path, options)
    elif builder == "feature_set_comparison":
        _main_metric_comparison_figure(
            analysis["stage2_integrated_feature_stats.csv"],
            category="feature_set", path=path, title=spec["title"],
            order=FEATURE_ORDER, options=options,
        )
    elif builder == "model_comparison":
        _main_metric_comparison_figure(
            analysis["stage2_integrated_model_stats.csv"],
            category="model_id", path=path, title=spec["title"],
            order=MODEL_ORDER, options=options,
        )
    elif builder == "feature_model_heatmap":
        data = analysis["stage2_integrated_feature_model_stats.csv"]
        _save_heatmap(
            data, row="feature_set", column="model_id", value="mean_AP",
            path=path, title=spec["title"], dpi=options.dpi,
            overwrite=options.overwrite, row_order=FEATURE_ORDER,
            column_order=MODEL_ORDER, main_suptitle=True,
            options=options,
        )
        _write_figure_source(path, data, options)
    elif builder == "final_candidate_comparison":
        _save_candidate_comparison_figure(
            analysis["stage2_final_candidate_comparison.csv"], path, options
        )
    elif builder == "main_feature_importance":
        _save_feature_importance_main(feature_importance, path, options, warnings)
    elif builder == "predicted_habitat_maps":
        _save_primary_habitat_maps_figure(train_summary, dirs, path, options, warnings)
    elif builder == "within_site_validation":
        _save_supplementary_s1_s3(
            model_df=analysis["within_overall_model.csv"],
            feature_df=analysis["within_overall_feature.csv"],
            feature_model_df=analysis["within_overall_feature_model.csv"],
            path=path, options=options,
            prefix=str(spec["number"]).replace("Figure ", ""),
        )
    elif builder == "cross_site_validation":
        _save_supplementary_s2(
            model_df=analysis["cross_overall_model.csv"],
            feature_df=analysis["cross_overall_feature.csv"],
            feature_model_df=analysis["cross_overall_feature_model.csv"],
            direction_df=analysis["cross_by_direction_feature.csv"],
            path=path, options=options,
        )
    elif builder == "spatial_cv_validation":
        _save_supplementary_s1_s3(
            model_df=analysis["spatial_overall_model.csv"],
            feature_df=analysis["spatial_overall_feature.csv"],
            feature_model_df=analysis["spatial_overall_feature_model.csv"],
            path=path, options=options,
            prefix=str(spec["number"]).replace("Figure ", ""),
        )
    elif builder == "integrated_comparison":
        _save_supplementary_s4(
            model_df=analysis["stage2_integrated_model_stats.csv"],
            feature_df=analysis["stage2_integrated_feature_stats.csv"],
            feature_model_df=analysis["stage2_integrated_feature_model_stats.csv"],
            path=path, options=options,
        )
    elif builder == "ranking_stability":
        _save_supplementary_s5(
            feature_rank_df=analysis["stage2_feature_rank_stability.csv"],
            feature_model_rank_df=analysis["stage2_feature_model_rank_stability.csv"],
            path=path, options=options,
        )
    elif builder == "final_candidate_analysis":
        _save_supplementary_s6(
            rank_stability_table=_rank_stability_shift_summary(analysis),
            probability_stats=probability_stats, binary_stats=binary_stats,
            dirs=dirs, path=path, options=options,
        )
    elif builder == "supplementary_feature_importance":
        _save_supplementary_s7(feature_importance=feature_importance, path=path, options=options)
    elif builder == "representative_validation":
        runs = [
            run for run in configured_representative_runs(PUBLICATION_CONFIG)
            if run["figure_id"] == spec["id"]
        ]
        if len(runs) != 1:
            raise ValueError(
                f"Figure {spec['id']!r} requires exactly one configured representative run."
            )
        run = runs[0]
        _save_supplementary_validation_site(
            output_root=output_root,
            target_site=run["target_site"],
            supplementary_figure=str(spec["number"]).replace("Figure ", ""),
            path=path,
            options=options,
        )
    else:
        raise ValueError(f"Unsupported configured Figure builder: {builder!r}")


def _write_figures(
    *,
    dirs: ReportDirs,
    options: ReportOptions,
    output_root: Path,
    analysis: Mapping[str, pd.DataFrame],
    feature_configs: Mapping[str, Mapping[str, Any]],
    train_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    probability_stats: pd.DataFrame,
    binary_stats: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Generate the revised publication-candidate figure set."""

    print("Writing report figures", flush=True)
    if not options.main_panels_only:
        for spec in configured_figure_specs(PUBLICATION_CONFIG):
            _write_configured_figure(
                spec,
                dirs=dirs,
                options=options,
                output_root=output_root,
                analysis=analysis,
                feature_configs=feature_configs,
                train_summary=train_summary,
                feature_importance=feature_importance,
                probability_stats=probability_stats,
                binary_stats=binary_stats,
                warnings=warnings,
            )

    if options.main_panels or options.main_panels_only:
        _write_main_panels(
            dirs=dirs,
            options=options,
            output_root=output_root,
            analysis=analysis,
            train_summary=train_summary,
            feature_importance=feature_importance,
            warnings=warnings,
        )
    if options.main_panels_only:
        return

    _write_official_composite_sidecars(dirs, options)
    if options.main_panels or options.main_panels_only:
        _write_main_panel_sidecars(dirs, options)
    _write_supplementary_panel_sidecars(dirs, options)


def _source_sidecar_frame(path: Path) -> pd.DataFrame:
    """Read a Figure source sidecar, returning an explicit source manifest if absent."""

    source_path = path.with_name(path.stem + "_source.csv")
    if source_path.exists():
        return pd.read_csv(source_path)
    return pd.DataFrame(
        [
            {
                "figure_file": str(path.resolve()),
                "source_description": "Existing stage-13 plotting inputs",
            }
        ]
    )


def _metadata_extra_from_source(source: pd.DataFrame) -> dict[str, Any]:
    """Extract stable metadata fields represented by a Figure source frame."""

    aliases = {
        "model": ["model_id", "model"],
        "feature_set": ["feature_set"],
        "seed": ["seed"],
        "site": ["site_id"],
        "target_site": ["target_site"],
        "validation_strategy": ["validation_strategy", "validation_type"],
        "threshold": ["threshold", "confusion_threshold"],
        "transfer_direction": ["transfer_direction"],
        "fold_or_pooled_oof": ["fold_or_pooled_oof"],
        "aggregation_method": ["aggregation_method"],
        "n_samples": ["n_samples"],
        "average_precision": ["average_precision"],
        "roc_auc": ["roc_auc"],
    }
    extra: dict[str, Any] = {}
    for output_name, candidates in aliases.items():
        for column in candidates:
            if column not in source.columns:
                continue
            values = source[column].dropna().astype(str).unique().tolist()
            if values:
                extra[output_name] = "; ".join(values[:20])
                break
    return extra


def _official_caption(number: str, title: str, stem: str) -> str:
    """Return a publication-ready caption for one official composite Figure."""

    if stem not in OFFICIAL_CAPTIONS:
        raise ValueError(f"No publication caption configured for {stem!r}.")
    return OFFICIAL_CAPTIONS[stem]

    captions = {
        "fig_01_study_areas": (
            "Figure 1. Study areas and multibeam echosounder survey context for Bongpyeong and Hujeong, "
            "Republic of Korea. Panels (a) and (c) show bathymetry within the valid survey footprints for "
            "Bongpyeong and Hujeong, respectively; panels (b) and (d) show the corresponding 300 kHz "
            "backscatter. Map coordinates use UTM Zone 52N (EPSG:32652), and scale bars report horizontal "
            "distance. Color scales describe the mapped bathymetric or acoustic variable, not sampling density "
            "or model performance. Table 1 provides complementary site and survey metadata."
        ),
        "fig_02_workflow_validation_framework": (
            "Figure 2. Multibeam echosounder (MBES) habitat-prediction workflow and validation framework. "
            "Panels (a-h) trace MBES acquisition, preprocessing, environmental-feature generation, labeled "
            "sampling, the model by feature-set experiment matrix, within-site validation, directional "
            "cross-site transfer, spatial block cross-validation (CV), integrated candidate selection, and "
            "final retraining and habitat mapping. Within-site validation uses held-out samples from the same "
            "site, cross-site validation trains on one site and evaluates the other, and spatial block CV "
            "separates geographically defined folds. Table 1 summarizes site context and Table S2 gives the "
            "model configurations; the diagram describes process rather than performance magnitude."
        ),
        "fig_03_mbes_feature_definitions": (
            "Figure 3. Concepts represented by the MBES-derived environmental features. Panel (a) shows "
            "acoustic intensity at 200, 300, and 400 kHz; panel (b) shows normalized frequency-difference "
            "indices (NDIs); panel (c) shows bathymetric features, including robust and relative depth; and "
            "panel (d) shows low-pass, high-pass, and Laplacian-of-Gaussian texture features. Individual "
            "features are measurement layers, whereas a feature set is a prescribed combination supplied to a "
            "model. Table S1 lists the exact composition of the seven Single, Dual, and Full 10-band sets."
        ),
        "fig_04_feature_set_comparison": (
            "Figure 4. Average-precision (AP) comparison among the seven feature sets. Panel (a) contrasts "
            "within-site, cross-site, and spatial-block CV marginal AP; panels (b-d) rank feature sets within "
            "each validation strategy; and panel (e) ranks their equal-weight Integrated AP across the three "
            "strategies. Marginal values average the stored model-level summaries for the corresponding feature "
            "set, and rank 1 is best. Higher AP indicates better precision-recall performance. Table 2 reports "
            "the exact top five model-feature combinations within each site or transfer scope rather than these "
            "feature-set marginals."
        ),
        "fig_05_model_comparison": (
            "Figure 5. Average-precision (AP) comparison among Random Forest, LightGBM, CatBoost, and XGBoost. "
            "Panel (a) contrasts within-site, cross-site, and spatial-block cross-validation (CV) marginal AP; "
            "panels (b-d) rank models within each validation strategy; and panel (e) ranks their equal-weight "
            "Integrated AP across the three strategies. Marginal values average the stored feature-set summaries for "
            "each model, rank 1 is best, and higher AP is better. Table 2 provides the exact leading "
            "model-feature combinations for each evaluation scope."
        ),
        "fig_06_feature_model_heatmap": (
            "Figure 6. Integrated performance of all 28 feature-set by model combinations. Rows are the seven "
            "feature sets, columns are Random Forest, LightGBM, CatBoost, and XGBoost, and each cell is the "
            "equal-weight mean average precision (AP) across within-site, cross-site, and spatial-block "
            "cross-validation summaries. Lighter or higher-valued cells indicate greater AP; the values are not "
            "sample-weighted or pooled predictions. Table 2 supplies the top five combinations within each "
            "individual evaluation scope."
        ),
        "fig_07_final_candidate_comparison": (
            "Figure 7. Validation performance of the four retained Random Forest mapping candidates, shown in "
            "canonical order: Balanced candidate, Cross-site optimized candidate, Spatial block CV optimized candidate, and Dual-frequency transfer candidate. "
            "Panel (a) compares within-site, cross-site, and spatial-block cross-validation average precision "
            "(AP); panel (b) ranks candidates by integrated AP. Integrated AP is the equal-weight arithmetic mean "
            "of the three validation-strategy AP values and is neither sample-weighted nor a pooled AP. "
            "Candidate colors and order are used consistently in subsequent figures. Table 3 reports the exact "
            "values, robustness summaries, and retained analytical roles."
        ),
        "fig_08_feature_importance": (
            "Figure 8. Random Forest impurity-based feature importance across the final site-candidate models. "
            "Panel (a) summarizes mean importance for individual features. Panel (b) shows mean feature-group "
            "importance: within each of the eight final site-candidate Random Forest models, feature importances "
            "were first summed within the acoustic-intensity, frequency-difference, bathymetric, and texture "
            "groups, with groups absent from a candidate feature set contributing zero, and the resulting group "
            "totals were then averaged equally across models. These are impurity-based importances, not "
            "permutation importance, causal effects, ecological effect sizes, or validation metrics. "
            "Table S8 provides the complete site- and candidate-specific importance values."
        ),
        "fig_09_predicted_habitat_maps": (
            "Figure 9. Final habitat predictions from the primary Balanced candidate (Random Forest with the "
            "Full 10-band feature set). Panels (a) and (c) show pixel-level habitat-class probability for "
            "Bongpyeong and Hujeong; panels (b) and (d) show binary habitat predictions at the fixed threshold "
            "p >= 0.5. Valid mapped area comprises finite prediction pixels accepted by the final feature mask; "
            "NoData and invalid-feature pixels are excluded. The maps indicate model-predicted habitat rather "
            "than independently mapped ground truth. Table 4 reports valid mapped area and predicted habitat "
            "area for each site."
        ),
        "fig_s01_within_site_validation": (
            "Figure S1. Held-out within-site validation across Bongpyeong and Hujeong for seven feature sets and "
            "four models. Panels (a) and (b) rank model and feature-set marginal mean average precision (AP), "
            "respectively, and panel (c) shows the feature-set by model AP matrix. The underlying "
            "site-model-feature summaries average five seed-specific held-out evaluations; marginal panels "
            "further average over dimensions they do not display, and any uncertainty shown is the corresponding "
            "stored summary standard deviation. Rank 1 is best. Table S3 gives the complete site, model, and "
            "feature-set results."
        ),
        "fig_s02_cross_site_validation": (
            "Figure S2. Directional cross-site transfer performance for Bongpyeong to Hujeong and Hujeong to "
            "Bongpyeong. Panels (a) and (b) rank model and feature-set marginal mean average precision (AP), "
            "panel (c) shows the feature-set by model matrix, and panel (d) separates feature-set AP by transfer "
            "direction. Models are trained at the origin site and evaluated on the complete target-site dataset; "
            "the two directions are not pooled. Underlying direction-model-feature summaries average five seeds; "
            "marginal panels further average over omitted dimensions, and rank 1 is best. Table S4 provides the "
            "complete direction-specific results."
        ),
        "fig_s03_spatial_block_cv_validation": (
            "Figure S3. Spatial block cross-validation (CV) for both sites across seven feature sets and four "
            "models. Panels (a) and (b) rank model and feature-set marginal mean average precision (AP), and "
            "panel (c) shows the feature-set by model AP matrix. Underlying site-model-feature summaries average "
            "five seeds by five spatial folds (25 fold-runs), and marginal panels further average over omitted "
            "dimensions. These are fold-run means, not a pooled out-of-fold (OOF) estimate; rank 1 is best. "
            "Table S5 provides the complete spatial block CV summaries."
        ),
        "fig_s04_integrated_comparison": (
            "Figure S4. Integrated marginal comparison of validation strategies, models, and feature sets. "
            "Panels (a) and (c) show within-site, cross-site, and spatial-block cross-validation average precision "
            "(AP) for model and feature-set marginals; panels (b) and (d) rank their equal-weight means; and "
            "panel (e) shows the integrated feature-set by model matrix. Integrated values give equal weight to "
            "the three validation strategies and are not sample-weighted or pooled AP. Marginal rankings do not "
            "by themselves define the selected individual model-feature candidate; canonical values are supplied "
            "in the unnumbered integrated marginal and feature-model Source Data."
        ),
        "fig_s05_ranking_stability": (
            "Figure S5. Stability of average-precision ranks across within-site, cross-site, and spatial-block "
            "cross-validation. Panel (a) shows each feature set's best, mean, and worst rank; panel (b) shows the "
            "same range for all 28 feature-set by model combinations in canonical feature and model order. Rank "
            "1 denotes the highest AP. A shorter best-to-worst interval indicates more stable relative placement, "
            "whereas signed rank shifts in Table S6 identify the direction and magnitude of movement between "
            "validation strategies."
        ),
        "fig_s06_final_candidate_analysis": (
            "Figure S6. Final-candidate robustness and map sensitivity. Panel (a) compares stored validation "
            "ranks for the four Random Forest candidates; rank 1 is best and the primary Balanced candidate is "
            "outlined for emphasis, with exact rank and shift values in Table S6. Panel (b) shows site-specific "
            "means, histogram-approximated medians, and interquartile ranges (p25-p75) of pixel-level habitat-class "
            "probability among positive prediction pixels only (p >= 0.5); probability is not average precision "
            "(AP). Panel (c) reports predicted habitat extent as a percentage of the full valid mapped area. Tables "
            "S6 and S7 provide the corresponding rank and map-statistic values."
        ),
        "fig_s07_feature_importance": (
            "Figure S7. Final-model Random Forest impurity-based feature importance. Panels (a-c) show the 12 "
            "features with the greatest combined mean importance, displayed for Hujeong, Bongpyeong, and both "
            "sites combined in a common order. Colors denote the same bathymetric, acoustic-intensity, texture, "
            "and frequency-difference groups used in Figure 8. Panel (d) shows distributions for those selected "
            "features across final retrained site by candidate models (seed 42); each observation is one final "
            "site-candidate model. Table S8 contains all 60 feature-importance records, including features not in "
            "the displayed top-12 subset. Importance is not permutation importance or validation performance."
        ),
        "fig_s08_validation_performance_hujeong": (
            "Figure S8. Representative validation performance for target site Hujeong using Random Forest, the "
            "Full 10-band feature set, and seed 42. Columns compare within-site held-out test predictions, "
            "five-fold spatial-block pooled out-of-fold (OOF) predictions, and complete-target cross-site "
            "predictions transferred from Bongpyeong; rows show confusion matrices, precision-recall curves, and "
            "receiver operating characteristic (ROC) curves. Confusion matrices use p >= 0.5, whereas average "
            "precision (AP) and ROC area under the curve (AUC) are threshold independent. No confidence interval "
            "or multiple-run average is shown. Table S9 provides the numerical AP, ROC AUC, and confusion counts."
        ),
        "fig_s09_validation_performance_bongpyeong": (
            "Figure S9. Representative validation performance for target site Bongpyeong using Random Forest, "
            "the Full 10-band feature set, and seed 42. Columns compare within-site held-out test predictions, "
            "five-fold spatial-block pooled out-of-fold (OOF) predictions, and complete-target cross-site "
            "predictions transferred from Hujeong; rows show confusion matrices, precision-recall curves, and "
            "receiver operating characteristic (ROC) curves. Confusion matrices use p >= 0.5, whereas average "
            "precision (AP) and ROC area under the curve (AUC) are threshold independent. No confidence interval "
            "or multiple-run average is shown. Table S10 provides the numerical AP, ROC AUC, and confusion counts."
        ),
    }
    return captions.get(stem, f"{number}. {title}.") + "\n"


def _write_official_composite_sidecars(
    dirs: ReportDirs,
    options: ReportOptions,
) -> None:
    """Refresh captions and metadata for all official composite Figures."""

    groups = [
        (MAIN_FIGURE_SPECS, dirs.main_composites),
        (SUPPLEMENTARY_FIGURE_SPECS, dirs.supplementary_composites),
    ]
    for specs, root in groups:
        for number, title, stem in specs:
            path = root / f"{stem}.png"
            if not path.exists():
                continue
            source = _portable_dataframe(_source_sidecar_frame(path))
            if options.generate_source_data:
                _write_figure_source(path, source, options)
            if options.generate_captions:
                _write_sidecar_text(
                    path.with_name(path.stem + "_caption.txt"),
                    _official_caption(number, title, stem),
                    options.overwrite,
                )
            if options.generate_metadata:
                _write_figure_metadata(
                    path,
                    parent_figure=number,
                    panel_letter="composite",
                    panel_title=title,
                    official_or_auxiliary="official_composite",
                    plot_function="_write_figures",
                    source_data=source,
                    options=options,
                    extra=_metadata_extra_from_source(source),
                )


def _write_main_panel_sidecars(dirs: ReportDirs, options: ReportOptions) -> None:
    """Refresh panel-specific captions and metadata for all Main panels."""

    for spec in _main_panel_specs():
        path = dirs.main_panels / spec["filename_png"]
        if not path.exists():
            continue
        source = _portable_dataframe(_source_sidecar_frame(path))
        if options.generate_source_data:
            _write_figure_source(path, source, options)
        if options.generate_captions:
            _write_sidecar_text(
                path.with_name(path.stem + "_caption.txt"),
                f"Panel ({spec['panel_letter']}). {spec['panel_title']}.\n",
                options.overwrite,
            )
        if options.generate_metadata:
            _write_figure_metadata(
                path,
                parent_figure=spec["figure_number"],
                panel_letter=spec["panel_letter"],
                panel_title=spec["panel_title"],
                official_or_auxiliary="official_panel",
                plot_function="_write_main_panels",
                source_data=source,
                options=options,
                extra=_metadata_extra_from_source(source),
            )


def _write_supplementary_panel_sidecars(dirs: ReportDirs, options: ReportOptions) -> None:
    """Normalize Supplementary panel sidecars after panel export."""

    for path in _configured_panel_paths(dirs.supplementary_panels, role="supplementary"):
        if not path.exists():
            continue
        metadata_path = path.with_name(path.stem + "_metadata.txt")
        metadata = _metadata_mapping(metadata_path)
        letter = metadata.get("panel_letter", "NA")
        title = metadata.get("panel_title", path.stem.replace("_", " "))
        source = _portable_dataframe(_source_sidecar_frame(path))
        if options.generate_source_data:
            _write_figure_source(path, source, options)
        if options.generate_metadata:
            _write_figure_metadata(
                path,
                parent_figure=_figure_number_from_stem(path.parent.name),
                panel_letter=letter,
                panel_title=title,
                official_or_auxiliary="official_panel",
                plot_function=metadata.get("plot_function", "_write_supplementary_composite_figures"),
                source_data=source,
                options=options,
                extra=_metadata_extra_from_source(source),
            )


def _contract_manifest_row(
    *,
    number: str,
    title: str,
    path: Path,
    usage: str,
    official: bool,
    parent_figure: str | None = None,
    panel_letter: str | None = None,
) -> dict[str, Any]:
    """Return one six-file Figure contract manifest row."""

    figure_root = next((parent for parent in path.parents if parent.name == "figures"), None)
    report_root = figure_root.parent if figure_root is not None else path.parent

    def portable(artifact: Path) -> str:
        try:
            return str(artifact.relative_to(report_root))
        except ValueError:
            return artifact.name

    publication = _figure_publication_details(path, panel_letter or "composite")
    return {
        "figure_number": number,
        "title": title,
        "parent_figure": parent_figure or number,
        "parent_figure_stem": publication.get("parent_figure_stem", path.stem),
        "panel_letter": panel_letter or "composite",
        "official": bool(official),
        "recommended_usage": usage,
        "paired_table_ids": publication.get("paired_table_ids", "NA"),
        "paired_source_data_ids": publication.get("paired_source_data_ids", "NA"),
        "png_path": portable(path),
        "svg_path": portable(path.with_suffix(".svg")),
        "pdf_path": portable(path.with_suffix(".pdf")),
        "source_path": portable(path.with_name(path.stem + "_source.csv")),
        "caption_path": portable(path.with_name(path.stem + "_caption.txt")),
        "metadata_path": portable(path.with_name(path.stem + "_metadata.txt")),
    }


def _metadata_mapping(path: Path) -> dict[str, str]:
    """Read simple key/value metadata sidecars."""

    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def _panel_manifest_rows(root: Path, usage: str) -> list[dict[str, Any]]:
    """Return manifest rows for generated panel directories."""

    rows: list[dict[str, Any]] = []
    role = "main" if usage == "Main panel" else "supplementary"
    for path in _configured_panel_paths(root, role=role):
        metadata = _metadata_mapping(path.with_name(path.stem + "_metadata.txt"))
        parent = _figure_number_from_stem(path.parent.name)
        letter = metadata.get("panel_letter", "NA")
        title = metadata.get("panel_title", path.stem.replace("_", " "))
        rows.append(
            _contract_manifest_row(
                number=f"{parent}({letter})",
                title=title,
                path=path,
                usage=usage,
                official=True,
                parent_figure=parent,
                panel_letter=letter,
            )
        )
    return rows


def _figure_manifest_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return a parseable Figure manifest even when an inventory is empty."""

    columns = [
        "figure_number", "title", "parent_figure", "parent_figure_stem",
        "panel_letter", "official", "recommended_usage", "paired_table_ids",
        "paired_source_data_ids", "png_path", "svg_path", "pdf_path",
        "source_path", "caption_path", "metadata_path",
    ]
    return pd.DataFrame(rows, columns=columns)


def _file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for a report artifact."""

    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publication_output_records(report_root: Path) -> list[dict[str, Any]]:
    """Return deterministic path, size, and hash records for publication outputs."""

    excluded = {
        Path("paper_manifest/publication_output_manifest.json"),
        Path("paper_manifest/publication_output_manifest.sha256"),
    }
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in report_root.rglob("*") if item.is_file()):
        relative = path.relative_to(report_root)
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return records


def _write_publication_output_manifest(
    *,
    dirs: ReportDirs,
    options: ReportOptions,
    output_root: Path,
    project_paths: Mapping[str, Path],
) -> None:
    """Write the Stage 13 authority and output-hash manifest."""

    authority_paths = {
        "publication_config": (
            project_paths["experiment_root"] / "publication.yaml",
            f"03_experiments/{options.project_id}/publication.yaml",
        ),
        "analysis_config": (
            project_paths["experiment_root"] / "analysis.yaml",
            f"03_experiments/{options.project_id}/analysis.yaml",
        ),
        "final_mapping_freeze": (
            output_root / "final_mapping_freeze" / "final_mapping_freeze.json",
            f"05_outputs/{options.project_id}/final_mapping_freeze/final_mapping_freeze.json",
        ),
        "candidate_authority": (
            output_root / "analysis_compare" / "stage2_final_candidate_comparison.csv",
            f"05_outputs/{options.project_id}/analysis_compare/stage2_final_candidate_comparison.csv",
        ),
    }
    authorities = {
        name: {
            "path": locator,
            "exists": path.is_file(),
            "sha256": _file_sha256(path),
        }
        for name, (path, locator) in authority_paths.items()
    }
    output_files = _publication_output_records(dirs.root)
    manifest = {
        "schema_version": 1,
        "project_id": options.project_id,
        "stage": "13_report_tables_figures",
        "stage13_source_sha256": _file_sha256(Path(__file__).resolve()),
        "git_commit": FIGURE_GIT_COMMIT,
        "generated_at": FIGURE_GENERATED_AT,
        "authorities": authorities,
        "output_file_count": len(output_files),
        "output_files": output_files,
    }
    manifest_path = dirs.paper_manifest / "publication_output_manifest.json"
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _write_text(manifest_path, content, options.overwrite)
    _write_text(
        dirs.paper_manifest / "publication_output_manifest.sha256",
        _file_sha256(manifest_path) + "\n",
        options.overwrite,
    )


def _table_manifest_record(
    dirs: ReportDirs,
    *,
    location: str,
    stem: str,
    table_number: str,
    title: str,
    table_type: str,
    official: bool,
    source_function: str,
    description: str,
    used_by: str,
    active: bool = True,
    duplicate_of: str = "",
) -> dict[str, Any]:
    """Return one current table inventory/manifest record."""

    roots = {
        "table_main": dirs.table_main,
        "table_supplementary": dirs.table_supplementary,
        "table_source_validation": dirs.table_source_validation,
        "table_source_figures": dirs.table_source_figures,
        "table_source_maps": dirs.table_source_maps,
        "table_source_inventories": dirs.table_source_inventories,
        "table_internal_supporting": dirs.table_internal_supporting,
        "table_internal_qc": dirs.table_internal_qc,
    }
    try:
        root = roots[location]
    except KeyError as exc:
        raise ValueError(f"Unsupported table location: {location}") from exc
    csv_path = root / f"{stem}.csv"
    caption_path = root / f"{stem}_caption.txt"
    metadata_path = root / f"{stem}_metadata.txt"
    dictionary_path = root / f"{stem}_dictionary.md"
    if official:
        paper_number = table_number.removeprefix("Table ")
        stem_match = re.match(r"table_(s?\d+)_", stem, flags=re.IGNORECASE)
        raw_stem_number = stem_match.group(1).upper() if stem_match else ""
        if raw_stem_number.startswith("S"):
            file_stem_number = f"S{int(raw_stem_number[1:])}"
        elif raw_stem_number:
            file_stem_number = str(int(raw_stem_number))
        else:
            file_stem_number = ""
        number_match = paper_number.upper() == file_stem_number
    else:
        paper_number = "NA"
        file_stem_number = "NA"
        number_match = True
    return {
        "table_id": stem,
        "table_number": table_number,
        "paper_table_number": paper_number,
        "file_stem": stem,
        "file_stem_number": file_stem_number,
        "number_match": number_match,
        "title": title,
        "table_type": table_type,
        "official": official,
        "active": active,
        "manifest_registered": active,
        "filename_csv": csv_path.name,
        "caption_file": caption_path.name,
        "metadata_file": metadata_path.name,
        "dictionary_file": dictionary_path.name if official else "NA",
        "recommended_usage": table_type,
        "csv_path": str(csv_path.relative_to(dirs.root)),
        "caption_path": str(caption_path.relative_to(dirs.root)),
        "metadata_path": str(metadata_path.relative_to(dirs.root)),
        "dictionary_path": str(dictionary_path.relative_to(dirs.root)) if official else "NA",
        "source_function": source_function,
        "description": description,
        "used_by": used_by,
        "duplicate_of": duplicate_of,
        "content_sha256": _file_sha256(csv_path),
        "csv_exists": csv_path.is_file() and csv_path.stat().st_size > 0,
        "caption_exists": caption_path.is_file() and caption_path.stat().st_size > 0,
        "metadata_exists": metadata_path.is_file() and metadata_path.stat().st_size > 0,
        "dictionary_exists": (
            dictionary_path.is_file() and dictionary_path.stat().st_size > 0
            if official
            else True
        ),
    }


def _current_table_manifest_rows(
    dirs: ReportDirs,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return official, source-data, and internal table records."""

    main_rows = [
        _table_manifest_record(
            dirs,
            location="table_main",
            stem=stem,
            table_number=number,
            title=title,
            table_type="official_main",
            official=True,
            source_function=source_function,
            description=f"Official main-paper table: {title}.",
            used_by="main manuscript",
        )
        for number, title, stem, source_function in MAIN_TABLE_SPECS
    ]
    supplementary_rows = [
        _table_manifest_record(
            dirs,
            location="table_supplementary",
            stem=stem,
            table_number=number,
            title=title,
            table_type="official_supplementary",
            official=True,
            source_function=source_function,
            description=f"Official supplementary table: {title}.",
            used_by="supplementary material",
        )
        for number, title, stem, source_function in SUPPLEMENTARY_TABLE_SPECS
    ]
    category_details = {
        "validation": ("table_source_validation", "source_validation", "validation reproducibility"),
        "figure": ("table_source_figures", "source_figure", "Figure source data"),
        "map": ("table_source_maps", "source_map", "map source data"),
        "inventory": ("table_source_inventories", "source_inventory", "portable provenance"),
    }
    source_rows = []
    for spec in configured_source_data_specs(PUBLICATION_CONFIG):
        location, table_type, used_by = category_details[spec["category"]]
        source_rows.append(
            _table_manifest_record(
                dirs,
                location=location,
                stem=spec["stem"],
                table_number="NA",
                title=spec["stem"].replace("_", " "),
                table_type=table_type,
                official=False,
                source_function="_write_tables",
                description="Canonical unnumbered source data.",
                used_by=used_by,
            )
        )
    internal_rows = []
    for location, root, table_type in [
        ("table_internal_supporting", dirs.table_internal_supporting, "internal_supporting"),
        ("table_internal_qc", dirs.table_internal_qc, "internal_qc"),
    ]:
        for path in sorted(root.glob("*.csv")):
            internal_rows.append(
                _table_manifest_record(
                    dirs,
                    location=location,
                    stem=path.stem,
                    table_number="NA",
                    title=path.stem.replace("_", " "),
                    table_type=table_type,
                    official=False,
                    source_function="_write_tables",
                    description="Internal supporting or QC table; not a publication table.",
                    used_by="local reproducibility and QC",
                )
            )
    return main_rows, supplementary_rows, source_rows, internal_rows


def _validate_table_contract_rows(
    dirs: ReportDirs,
    rows: Sequence[Mapping[str, Any]],
    *,
    require_caption: bool,
) -> pd.DataFrame:
    """Validate the CSV/caption/metadata contract for active table rows."""

    contract_rows: list[dict[str, Any]] = []
    for record in rows:
        artifacts = [("csv", "csv_path"), ("metadata", "metadata_path")]
        if require_caption:
            artifacts.insert(1, ("caption", "caption_path"))
            artifacts.append(("dictionary", "dictionary_path"))
        for artifact, column in artifacts:
            path = dirs.root / str(record[column])
            exists = path.is_file()
            size = path.stat().st_size if exists else 0
            contract_rows.append(
                {
                    "table_id": record["table_id"],
                    "table_number": record["table_number"],
                    "table_type": record["table_type"],
                    "artifact": artifact,
                    "path": str(path.relative_to(dirs.root)),
                    "exists": exists,
                    "size_bytes": size,
                    "status": "ok" if exists and size > 0 else "missing_or_empty",
                }
            )
    return pd.DataFrame(contract_rows)


def _write_paper_manifest(dirs: ReportDirs, options: ReportOptions) -> None:
    """Write current Figure and Table manifests for the report output tree."""

    print("Writing reorganized paper manifest", flush=True)
    main_rows = [
        _contract_manifest_row(
            number=number,
            title=title,
            path=dirs.main_composites / f"{stem}.png",
            usage="Main",
            official=True,
        )
        for number, title, stem in MAIN_FIGURE_SPECS
    ]
    supplementary_rows = [
        _contract_manifest_row(
            number=number,
            title=title,
            path=dirs.supplementary_composites / f"{stem}.png",
            usage="Supplementary",
            official=True,
        )
        for number, title, stem in SUPPLEMENTARY_FIGURE_SPECS
    ]
    main_panel_rows = _panel_manifest_rows(dirs.main_panels, "Main panel")
    supplementary_panel_rows = _panel_manifest_rows(
        dirs.supplementary_panels,
        "Supplementary panel",
    )
    main_table_rows, supplementary_table_rows, source_table_rows, internal_table_rows = (
        _current_table_manifest_rows(dirs)
    )
    official_table_rows = main_table_rows + supplementary_table_rows
    official_numbers = [str(row["table_number"]) for row in official_table_rows]
    if len(official_numbers) != len(set(official_numbers)):
        raise ValueError("Duplicate official table number detected in active table manifest specs")
    active_table_rows = official_table_rows + source_table_rows + internal_table_rows
    mismatched_numbers = [row for row in official_table_rows if not bool(row["number_match"])]
    if mismatched_numbers:
        raise ValueError(
            "Official table numbers do not match canonical filename stems: "
            + ", ".join(str(row["table_id"]) for row in mismatched_numbers)
        )
    table_contract = _validate_table_contract_rows(
        dirs, official_table_rows, require_caption=True
    )
    source_contract = _validate_table_contract_rows(
        dirs, source_table_rows, require_caption=False
    )
    internal_contract = _validate_table_contract_rows(
        dirs, internal_table_rows, require_caption=False
    )
    table_inventory = pd.DataFrame(active_table_rows)
    recommended_rows = main_rows + supplementary_rows
    mapping_rows = recommended_rows + main_panel_rows + supplementary_panel_rows
    outputs = {
        "main_figures.csv": _figure_manifest_frame(main_rows),
        "supplementary_figures.csv": _figure_manifest_frame(supplementary_rows),
        "main_panel_figures.csv": _figure_manifest_frame(main_panel_rows),
        "supplementary_panel_figures.csv": _figure_manifest_frame(supplementary_panel_rows),
        "recommended_paper_layout.csv": _figure_manifest_frame(recommended_rows),
        "figure_file_mapping.csv": _figure_manifest_frame(mapping_rows),
        "main_tables.csv": pd.DataFrame(main_table_rows),
        "supplementary_tables.csv": pd.DataFrame(supplementary_table_rows),
        "source_data_tables.csv": pd.DataFrame(source_table_rows),
        "internal_tables.csv": pd.DataFrame(internal_table_rows),
        "table_file_mapping.csv": pd.DataFrame(active_table_rows),
        "table_inventory_audit.csv": table_inventory,
        "table_contract_validation.csv": table_contract,
        "source_data_contract_validation.csv": pd.concat(
            [source_contract, internal_contract], ignore_index=True
        ),
    }
    if options.tables_only and not dirs.main_composites.exists():
        outputs = {
            name: frame
            for name, frame in outputs.items()
            if name
            in {
                "main_tables.csv",
                "supplementary_tables.csv",
                "source_data_tables.csv",
                "internal_tables.csv",
                "table_file_mapping.csv",
                "table_inventory_audit.csv",
                "table_contract_validation.csv",
                "source_data_contract_validation.csv",
            }
        }
    for name, frame in outputs.items():
        _write_csv(frame, dirs.paper_manifest / name, options.overwrite)
    contract = configured_publication_contract(PUBLICATION_CONFIG)
    expected_counts = {
        "main_figures.csv": contract["main_figures"],
        "supplementary_figures.csv": contract["supplementary_figures"],
        "main_panel_figures.csv": contract["main_panels"],
        "supplementary_panel_figures.csv": contract["supplementary_panels"],
        "recommended_paper_layout.csv": contract["main_figures"] + contract["supplementary_figures"],
        "main_tables.csv": contract["main_tables"],
        "supplementary_tables.csv": contract["supplementary_tables"],
        "source_data_tables.csv": len(source_table_rows),
        "internal_tables.csv": len(internal_table_rows),
        "table_file_mapping.csv": len(active_table_rows),
    }
    for name, expected in expected_counts.items():
        if name not in outputs:
            continue
        actual = int(outputs[name].shape[0])
        if actual != expected:
            raise ValueError(f"{name} has {actual} rows; expected {expected}")
    failures = table_contract[table_contract["status"] != "ok"]
    if not failures.empty:
        raise ValueError(
            "Table sidecar contract validation failed; see "
            f"{dirs.paper_manifest / 'table_contract_validation.csv'}"
        )
    source_failures = pd.concat([source_contract, internal_contract], ignore_index=True)
    source_failures = source_failures[source_failures["status"] != "ok"]
    if not source_failures.empty:
        raise ValueError(
            "Source/internal table contract validation failed; see "
            f"{dirs.paper_manifest / 'source_data_contract_validation.csv'}"
        )


def _validate_figure_contracts(dirs: ReportDirs, options: ReportOptions) -> pd.DataFrame:
    """Validate all manifest Figure rows against the six-file contract."""

    manifest_names = [
        "main_figures.csv",
        "supplementary_figures.csv",
        "main_panel_figures.csv",
        "supplementary_panel_figures.csv",
    ]
    rows: list[dict[str, Any]] = []
    path_columns = [
        "png_path",
        "svg_path",
        "pdf_path",
        "source_path",
        "caption_path",
        "metadata_path",
    ]
    for manifest_name in manifest_names:
        manifest_path = dirs.paper_manifest / manifest_name
        frame = pd.read_csv(manifest_path)
        if frame.duplicated(subset=["png_path"]).any():
            raise ValueError(f"Duplicate Figure rows in {manifest_path}")
        for record in frame.to_dict(orient="records"):
            for column in path_columns:
                path = Path(str(record[column]))
                if not path.is_absolute():
                    path = dirs.root / path
                exists = path.exists()
                size = path.stat().st_size if exists else 0
                rows.append(
                    {
                        "manifest": manifest_name,
                        "figure_number": record.get("figure_number", ""),
                        "artifact": column,
                        "path": str(path.relative_to(dirs.root)),
                        "exists": exists,
                        "size_bytes": size,
                        "status": "ok" if exists and size > 0 else "missing_or_empty",
                    }
                )
    result = pd.DataFrame(rows)
    failures = result[result["status"] != "ok"]
    _write_csv(
        result,
        dirs.paper_manifest / "figure_contract_validation.csv",
        options.overwrite,
    )
    if not failures.empty:
        raise ValueError(
            f"Figure six-file contract failed for {failures.shape[0]} artifact(s); "
            f"see {dirs.paper_manifest / 'figure_contract_validation.csv'}"
        )
    loose = [path for path in dirs.figures.iterdir() if path.is_file() and path.name != "README_output_structure.md"]
    if loose:
        raise ValueError("Loose files remain in figures root: " + ", ".join(str(path) for path in loose))
    return result


def _write_figure_structure_readme(dirs: ReportDirs, options: ReportOptions) -> None:
    """Document the current Figure output contract and locations."""

    text = """<!-- AUTO:folder-guide:start -->
# report_outputs/figures

## Purpose
Canonical Main and Supplementary publication Figures and vector-preserving panel exports.

## Created by
Stage 13; synchronized documentation markers are maintained by Stage 99.

## Used by
Publication assembly and paper manifests.

## Directory structure
`main/composites/`, `main/panels/`, `supplementary/composites/`, and `supplementary/panels/`.

## Key files
- PNG 600 dpi
- SVG and PDF vector outputs
- portable source CSV
- publication caption TXT
- metadata TXT with canonical Table and Source Data pairing

## Interpretation notes
Every official composite and panel follows the six-file contract recorded in `paper_manifest/`. Figure sidecars use repository-relative paths or artifact identifiers and inherit pairing from one canonical Stage 13 specification.

## Rebuild dependency
Rerun Stage 13 from saved scientific outputs; no model retraining is required.

## Preservation policy
KEEP official Figure files and sidecars. Legacy flat outputs are not active results.
<!-- AUTO:folder-guide:end -->

<!-- MANUAL:local-notes:start -->
<!-- Add human-authored local notes here. Stage 99 preserves this block. -->
<!-- MANUAL:local-notes:end -->
"""
    _write_text(dirs.figures / "README_output_structure.md", text, options.overwrite)


def _write_table_structure_readme(dirs: ReportDirs, options: ReportOptions) -> None:
    """Document the publication-centered Table hierarchy."""

    contract = configured_publication_contract(PUBLICATION_CONFIG)
    text = f"""<!-- AUTO:folder-guide:start -->
# report_outputs/tables

## Purpose
Official Main/Supplementary Tables, canonical source data, and internal/QC tables.

## Created by
Stage 13; synchronized documentation markers are maintained by Stage 99.

## Used by
Publication assembly, Figure generation, and reproducibility review.

## Directory structure
`main/`, `supplementary/`, `source_data/{{validation,figures,maps,inventories}}/`, and `internal/{{supporting,qc}}/`.

## Key files
- official four-file contract: CSV, caption TXT, metadata TXT, and dictionary Markdown
- `TABLE_INDEX.md` links every official table to its contract artifacts
- table manifests and contract validation

## Interpretation notes
Only Main and Supplementary Tables carry paper numbers; source and internal artifacts are unnumbered. The official set contains {contract['main_tables']} Main Tables and {contract['supplementary_tables']} Supplementary Tables.

## Rebuild dependency
Rerun Stage 13 from stored validation, analysis, and final-map results.

## Preservation policy
KEEP official tables and all source/metadata sidecars.
<!-- AUTO:folder-guide:end -->

<!-- MANUAL:local-notes:start -->
<!-- Add human-authored local notes here. Stage 99 preserves this block. -->
<!-- MANUAL:local-notes:end -->
"""
    _write_text(dirs.tables / "README_table_structure.md", text, options.overwrite)


def _write_report_structure_readme(dirs: ReportDirs, options: ReportOptions) -> None:
    """Document the purpose of each clean report-output directory."""

    text = f"""# {options.project_id} report outputs

- `figures/main`: publication Main Figure composites and logical panels.
- `figures/supplementary`: publication Supplementary Figure composites and logical panels.
- `tables/main`: official Main Tables.
- `tables/supplementary`: official Supplementary Tables.
- `tables/source_data`: unnumbered validation, Figure, map, and inventory source data.
- `tables/internal`: local supporting comparisons and QC inventories.
- `maps`: GIS-ready GeoTIFFs and map previews.
- `asciis`: official `native` and `resampled_0p5m` ASC/TXT grids retained for Surfer analysis and visualization.
- `diagnostics`: active quality-control and reproducibility checks.
- `paper_manifest`: official file mappings and artifact-contract validation.
- `grids`: optional XYZ exports created only when an XYZ CLI option is enabled.

The active report tree does not contain legacy aliases, backups, archives, or default XYZ grids.
ASC and TXT exports are both official retained formats and are not duplicate-removal candidates.
This tree is generated from existing validation, analysis, and final-map results by Stage 13.
"""
    _write_text(dirs.root / "README_report_outputs.md", text, options.overwrite)


def _expected_file_inventory(
    *,
    output_root: Path,
    dirs: ReportDirs,
    train_summary: pd.DataFrame,
    analysis_config: Mapping[str, Any],
    options: ReportOptions,
) -> pd.DataFrame:
    """Build a QC inventory of core inputs and expected report outputs."""

    rows: list[dict[str, Any]] = []
    for name in [*CORE_ANALYSIS_FILES, *_focused_analysis_files(analysis_config)]:
        path = _analysis_path(output_root, name)
        rows.append(
            {
                "category": "input_analysis",
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    for record in _map_record_iter(train_summary):
        for key in ["probability_src", "valid_mask_src", "train_summary_path"]:
            path = Path(record[key])
            rows.append(
                {
                    "category": "input_final_map",
                    "path": str(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                }
            )
        site_id = record["site_id"]
        candidate_label = record["candidate_label"]
        expected = [
            _probability_output_path(dirs, site_id, candidate_label),
            _binary_output_path(dirs, site_id, candidate_label),
            _png_output_path(dirs, site_id, candidate_label, "probability"),
            _png_output_path(dirs, site_id, candidate_label, "binary_05"),
        ]
        if not options.no_asc:
            ascii_dir = dirs.asciis / site_id / candidate_label
            if not options.skip_native_asc:
                expected.extend(
                    [
                        ascii_dir / "native" / "probability.asc",
                        ascii_dir / "native" / "probability.txt",
                        ascii_dir / "native" / "binary_05.asc",
                        ascii_dir / "native" / "binary_05.txt",
                    ]
                )
            if not options.skip_0p5m_asc:
                expected.extend(
                    [
                        ascii_dir / "resampled_0p5m" / "probability.asc",
                        ascii_dir / "resampled_0p5m" / "probability.txt",
                        ascii_dir / "resampled_0p5m" / "binary_05.asc",
                        ascii_dir / "resampled_0p5m" / "binary_05.txt",
                    ]
                )
        for path in expected:
            rows.append(
                {
                    "category": "report_output",
                    "path": path.relative_to(dirs.root).as_posix(),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                }
            )
    return pd.DataFrame(rows)


def _write_diagnostics(
    *,
    dirs: ReportDirs,
    options: ReportOptions,
    file_inventory: pd.DataFrame,
    train_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    probability_stats: pd.DataFrame,
    binary_stats: pd.DataFrame,
    export_inventory: pd.DataFrame,
    alignment_check: pd.DataFrame,
    warnings: Sequence[str],
    generated_counts: Mapping[str, Any],
) -> None:
    """Write QC diagnostic CSV files."""

    print("Writing report diagnostics", flush=True)
    missing = file_inventory[~file_inventory["exists"]].copy() if not file_inventory.empty else pd.DataFrame()
    valid_counts = probability_stats[
        [
            "site_id",
            "candidate_label",
            "valid_pixel_count",
            "invalid_pixel_count",
            "valid_ratio",
        ]
    ].copy() if not probability_stats.empty else pd.DataFrame()
    timestamp = datetime.now().isoformat(timespec="seconds")
    generation_rows = [
        {
            "item_type": "report_root",
            "item_name": "report_outputs",
            "status": "ok",
            "message": f"project_id={options.project_id}; threshold={options.threshold}",
            "output_path": ".",
            "source_path": "",
            "timestamp": timestamp,
        }
    ]
    for key, value in dict(generated_counts).items():
        generation_rows.append(
            {
                "item_type": "count",
                "item_name": key,
                "status": "ok",
                "message": str(value),
                "output_path": ".",
                "source_path": "",
                "timestamp": timestamp,
            }
        )
    for warning in dict.fromkeys(warnings):
        generation_rows.append(
            {
                "item_type": "warning",
                "item_name": "report_warning",
                "status": "warning",
                "message": str(warning),
                "output_path": ".",
                "source_path": "",
                "timestamp": timestamp,
            }
        )
    generation_summary = pd.DataFrame(
        generation_rows,
        columns=[
            "item_type",
            "item_name",
            "status",
            "message",
            "output_path",
            "source_path",
            "timestamp",
        ],
    )

    _write_csv(file_inventory, dirs.diagnostics / "qc_file_inventory.csv", options.overwrite)
    _write_csv(missing, dirs.diagnostics / "qc_missing_outputs.csv", options.overwrite)
    _write_csv(train_summary, dirs.diagnostics / "qc_train_summary_combined.csv", options.overwrite)
    _write_csv(feature_importance, dirs.diagnostics / "qc_feature_importance_combined.csv", options.overwrite)
    _write_csv(probability_stats, dirs.diagnostics / "qc_probability_stats.csv", options.overwrite)
    _write_csv(binary_stats, dirs.diagnostics / "qc_binary_05_stats.csv", options.overwrite)
    _write_csv(valid_counts, dirs.diagnostics / "qc_valid_pixel_counts.csv", options.overwrite)
    _write_csv(export_inventory, dirs.diagnostics / "qc_export_inventory.csv", options.overwrite)
    _write_csv(export_inventory, dirs.diagnostics / "qc_ascii_export_summary.csv", options.overwrite)
    _write_csv(alignment_check, dirs.diagnostics / "qc_raster_alignment_check.csv", options.overwrite)
    _write_csv(generation_summary, dirs.diagnostics / "qc_report_generation_summary.csv", options.overwrite)


def _estimate_ascii_size(width: int, height: int, bytes_per_cell: float) -> float:
    """Estimate ASCII export size in bytes."""

    return float(width) * float(height) * bytes_per_cell


def _dry_run_report(
    *,
    report_root: Path,
    output_root: Path,
    sites: Sequence[str],
    candidates: pd.DataFrame,
    train_summary: pd.DataFrame,
    options: ReportOptions,
) -> None:
    """Print dry-run report without writing files."""

    contract = configured_publication_contract(PUBLICATION_CONFIG)
    total_native = 0.0
    total_resampled = 0.0
    raster_summaries: list[dict[str, Any]] = []
    for record in _map_record_iter(train_summary):
        metadata = _raster_profile_record(record["probability_src"])
        width = int(metadata["width"])
        height = int(metadata["height"])
        native_prob = _estimate_ascii_size(width, height, 12.0)
        native_binary = _estimate_ascii_size(width, height, 3.0)
        bounds = metadata["bounds"]
        resampled_width = int(math.ceil((bounds.right - bounds.left) / ASCII_RESAMPLED_RESOLUTION))
        resampled_height = int(math.ceil((bounds.top - bounds.bottom) / ASCII_RESAMPLED_RESOLUTION))
        resampled_prob = _estimate_ascii_size(resampled_width, resampled_height, 12.0)
        resampled_binary = _estimate_ascii_size(resampled_width, resampled_height, 3.0)
        if not options.skip_native_asc:
            total_native += 2 * (native_prob + native_binary)
        if not options.skip_0p5m_asc:
            total_resampled += 2 * (resampled_prob + resampled_binary)
        raster_summaries.append(
            {
                "site_id": record["site_id"],
                "candidate_label": record["candidate_label"],
                "width": width,
                "height": height,
                "native_ascii_txt_est_gib": 0.0
                if options.skip_native_asc
                else 2 * (native_prob + native_binary) / 1024**3,
                "resampled_0p5m_ascii_txt_est_gib": 0.0
                if options.skip_0p5m_asc
                else 2 * (resampled_prob + resampled_binary) / 1024**3,
            }
        )

    print("Dry-run report generation", flush=True)
    print(f"  output_root: {output_root}", flush=True)
    print(f"  report_outputs: {report_root}", flush=True)
    print(f"  detected sites: {list(sites)}", flush=True)
    print(
        "  detected candidates: "
        + str(candidates[["candidate_label", "feature_set", "model_id"]].to_dict("records")),
        flush=True,
    )
    print(f"  final map sets: {len(train_summary)}", flush=True)
    print(
        "  output structure: tables/main, tables/supplementary, tables/source_data, "
        "tables/internal, figures/main/composites, figures/main/panels, "
        "figures/supplementary/composites, figures/supplementary/panels, "
        "maps, diagnostics, asciis, paper_manifest",
        flush=True,
    )
    if options.allow_full_xyz or options.xyz_stride is not None:
        print("  grids folder: enabled", flush=True)
    else:
        print("  XYZ disabled; grids folder will not be created", flush=True)
    print(f"  expected Main Figure count = {contract['main_figures']}", flush=True)
    if options.main_panels or options.main_panels_only:
        print(f"  expected Main Figure panel count = {len(_main_panel_specs())}", flush=True)
        print("  planned main panel files:", flush=True)
        for row in _main_panel_specs():
            print(
                f"    {row['figure_number']} panel {row['panel_letter']}: "
                f"{row['filename_png']} / {row['filename_svg']}",
                flush=True,
            )
    supplementary_count = 0 if options.main_panels_only else contract["supplementary_figures"]
    main_table_count = 0 if options.main_panels_only else contract["main_tables"]
    supplementary_table_count = 0 if options.main_panels_only else contract["supplementary_tables"]
    map_file_count = 0 if options.main_panels_only else len(train_summary) * 4
    print(f"  expected Supplementary Figure count = {supplementary_count}", flush=True)
    print(f"  expected Main Table count: {main_table_count}", flush=True)
    print(f"  expected Supplementary Table count: {supplementary_table_count}", flush=True)
    print(f"  expected configured Source Data tables: {contract['source_data']}", flush=True)
    print(f"  expected map files: {map_file_count} (probability/binary GeoTIFF+PNG)", flush=True)
    native_files = 0 if options.skip_native_asc or options.main_panels_only else len(train_summary) * 4
    resampled_files = 0 if options.skip_0p5m_asc or options.main_panels_only else len(train_summary) * 4
    print(f"  expected native ASC/TXT files: {native_files}", flush=True)
    print(f"  expected 0.5m ASC/TXT files: {resampled_files}", flush=True)
    print("  expected figure manifest files: 6", flush=True)
    print("  expected figure source/caption files: one *_source.csv and *_caption.txt per generated figure", flush=True)
    print(f"  vector SVG/PDF output enabled: {options.save_vector}", flush=True)
    if options.main_panels_only:
        print("  expected per-map outputs: skipped by --main-panels-only", flush=True)
        total_native = 0.0
        total_resampled = 0.0
    else:
        print(
            "  expected per-map outputs: 4 map files + 8 ASC/TXT files + 3 diagnostic text files"
            + (" + 2 XYZ files when enabled" if options.allow_full_xyz or options.xyz_stride else ""),
            flush=True,
        )
    print(f"  estimated native ASC/TXT size: {total_native / 1024**3:.2f} GiB", flush=True)
    print(f"  estimated 0.5m ASC/TXT size: {total_resampled / 1024**3:.2f} GiB", flush=True)
    print(
        f"  estimated report_outputs total size rough lower bound: "
        f"{(total_native + total_resampled) / 1024**3:.2f} GiB plus maps/figures",
        flush=True,
    )
    if not options.main_panels_only:
        for summary in raster_summaries:
            print(f"  raster: {summary}", flush=True)


def run_report_tables_figures(options: ReportOptions) -> Path:
    """Run stage-13 reporting."""

    global FIGURE_GENERATED_AT, FIGURE_GIT_COMMIT, SAVE_VECTOR_OUTPUTS
    options = _effective_report_options(options)
    FIGURE_GENERATED_AT = datetime.now().isoformat(timespec="seconds")
    FIGURE_GIT_COMMIT = _current_git_commit()
    SAVE_VECTOR_OUTPUTS = bool(options.save_vector)

    if not 0 <= float(options.threshold) <= 1:
        raise ValueError("--threshold must be between 0 and 1.")
    if options.dpi <= 0:
        raise ValueError("--dpi must be a positive integer.")
    if options.max_preview_size <= 0:
        raise ValueError("--max-preview-size must be a positive integer.")
    if options.xyz_stride is not None and options.xyz_stride <= 0:
        raise ValueError("--xyz-stride must be a positive integer.")

    project_config = load_project_config(project_id=options.project_id)
    sites_config = load_sites_config(project_id=options.project_id)
    experiment_matrix = load_experiment_matrix(project_id=options.project_id)
    analysis_config = load_analysis_config(project_id=options.project_id)
    publication_config = load_publication_config(project_id=options.project_id)
    _configure_publication_policy(publication_config)
    feature_configs = load_all_feature_set_configs(project_id=options.project_id)
    model_configs = load_all_model_configs(project_id=options.project_id)
    paths = resolve_project_paths(project_config)
    output_root = paths["output_root"]
    report_root = options.output_dir.expanduser().resolve() if options.output_dir else output_root / "report_outputs"
    dirs = _make_report_dirs(report_root, options)
    if not options.dry_run:
        _write_typography_report(dirs, options)
    print(f"Resolved publication font: {RESOLVED_FONT_FAMILY}", flush=True)

    analysis = _load_analysis_inputs(output_root, analysis_config)
    candidates = _selected_candidates(
        analysis["stage2_final_candidate_comparison.csv"],
        options.candidate_label_filter,
    )
    sites = _selected_sites(experiment_matrix, options.site_filter)
    map_sites = _selected_mapping_sites(
        analysis_config, experiment_matrix, options.site_filter
    )
    train_summary, feature_importance = _load_final_map_records(
        output_root=output_root,
        sites_config=sites_config,
        sites=map_sites,
        candidates=candidates,
    )

    if options.figures_only:
        probability_path = dirs.table_source_maps / "probability_statistics.csv"
        binary_path = dirs.table_source_maps / "binary_statistics.csv"
        if not probability_path.exists() or not binary_path.exists():
            raise FileNotFoundError(
                "--figures-only requires existing report raster-statistics tables: "
                f"{probability_path}; {binary_path}"
            )
        probability_stats = _read_csv(probability_path, "Probability raster statistics")
        binary_stats = _read_csv(binary_path, "Binary raster statistics")
        warnings: list[str] = []
        _write_figures(
            dirs=dirs,
            options=options,
            output_root=output_root,
            analysis=analysis,
            feature_configs=feature_configs,
            train_summary=train_summary,
            feature_importance=feature_importance,
            probability_stats=probability_stats,
            binary_stats=binary_stats,
            warnings=warnings,
        )
        _write_official_composite_sidecars(dirs, options)
        _write_main_panel_sidecars(dirs, options)
        _write_paper_manifest(dirs, options)
        _write_figure_structure_readme(dirs, options)
        _validate_figure_contracts(dirs, options)
        _write_publication_output_manifest(
            dirs=dirs,
            options=options,
            output_root=output_root,
            project_paths=paths,
        )
        print(f"Figure-only report outputs written to: {dirs.figures}", flush=True)
        return report_root

    if options.dry_run:
        _dry_run_report(
            report_root=report_root,
            output_root=output_root,
            sites=sites,
            candidates=candidates,
            train_summary=train_summary,
            options=options,
        )
        return report_root

    warnings: list[str] = []
    if options.main_panels_only:
        probability_stats = pd.DataFrame()
        binary_stats = pd.DataFrame()
        area_summary = pd.DataFrame()
        export_inventory = pd.DataFrame()
        alignment_check = pd.DataFrame()
        file_inventory = pd.DataFrame()
    else:
        probability_stats, binary_stats, area_summary, export_inventory, alignment_check = _process_final_maps(
            train_summary=train_summary,
            dirs=dirs,
            options=options,
            warnings=warnings,
        )
        file_inventory = _expected_file_inventory(
            output_root=output_root,
            dirs=dirs,
            train_summary=train_summary,
            analysis_config=analysis_config,
            options=options,
        )

    if not options.no_tables and not options.main_panels_only:
        _write_tables(
            dirs=dirs,
            options=options,
            output_root=output_root,
            analysis=analysis,
            analysis_config=analysis_config,
            sites_config=sites_config,
            sites=sites,
            feature_configs=feature_configs,
            model_configs=model_configs,
            train_summary=train_summary,
            feature_importance=feature_importance,
            probability_stats=probability_stats,
            binary_stats=binary_stats,
            area_summary=area_summary,
            export_inventory=export_inventory,
            file_inventory=file_inventory,
            target_resolution_m=float(project_config["spatial"]["target_resolution_m"]),
        )

    if not options.no_figures:
        try:
            _write_figures(
                dirs=dirs,
                options=options,
                output_root=output_root,
                analysis=analysis,
                feature_configs=feature_configs,
                train_summary=train_summary,
                feature_importance=feature_importance,
                probability_stats=probability_stats,
                binary_stats=binary_stats,
                warnings=warnings,
            )
        except Exception as exc:
            warnings.append(f"Figure generation failed: {exc}")
            print(f"Warning: figure generation failed: {exc}", flush=True)

    if not options.main_panels_only:
        file_inventory = _expected_file_inventory(
            output_root=output_root,
            dirs=dirs,
            train_summary=train_summary,
            analysis_config=analysis_config,
            options=options,
        )
    if not options.no_diagnostics and not options.main_panels_only:
        _write_diagnostics(
            dirs=dirs,
            options=options,
            file_inventory=file_inventory,
            train_summary=train_summary,
            feature_importance=feature_importance,
            probability_stats=probability_stats,
            binary_stats=binary_stats,
            export_inventory=export_inventory,
            alignment_check=alignment_check,
            warnings=warnings,
            generated_counts={
                "table_outputs_enabled": not options.no_tables,
                "figure_outputs_enabled": not options.no_figures,
                "map_outputs_enabled": not options.no_maps,
                "asc_outputs_enabled": not options.no_asc,
                "grid_outputs_enabled": (options.allow_full_xyz or options.xyz_stride is not None)
                and not options.no_grids,
                "final_map_set_count": int(train_summary.shape[0]),
                "probability_stats_rows": int(probability_stats.shape[0]),
                "binary_stats_rows": int(binary_stats.shape[0]),
                "export_inventory_rows": int(export_inventory.shape[0]),
            },
        )

    if options.generate_captions or options.generate_source_data or options.generate_metadata:
        _write_artifact_sidecars(
            dirs=dirs,
            options=options,
            output_root=output_root,
        )

    if not options.no_figures:
        _write_official_composite_sidecars(dirs, options)
        _write_main_panel_sidecars(dirs, options)
        _write_supplementary_panel_sidecars(dirs, options)

    if not options.tables_only:
        _write_figure_structure_readme(dirs, options)
    _write_table_structure_readme(dirs, options)
    if not options.tables_only:
        _write_report_structure_readme(dirs, options)

    if options.generate_paper_manifest:
        _write_paper_manifest(dirs, options)
        if not options.no_figures or (options.tables_only and dirs.main_composites.exists()):
            _validate_figure_contracts(dirs, options)
        _write_publication_output_manifest(
            dirs=dirs,
            options=options,
            output_root=output_root,
            project_paths=paths,
        )

    print(f"Report outputs written to: {report_root}", flush=True)
    return report_root


def _effective_report_options(options: ReportOptions) -> ReportOptions:
    """Normalize mutually dependent Stage 13 publication modes."""

    if options.figures_only:
        options = replace(
            options,
            dpi=PANEL_DPI,
            save_vector=True,
            main_panels=True,
            generate_paper_manifest=True,
            generate_captions=True,
            generate_source_data=True,
            generate_metadata=True,
        )
    if options.tables_only:
        options = replace(
            options,
            no_figures=True,
            no_tables=False,
            no_maps=True,
            no_asc=True,
            no_diagnostics=True,
            no_grids=True,
            skip_native_asc=True,
            skip_0p5m_asc=True,
            generate_paper_manifest=True,
            generate_captions=True,
            generate_metadata=True,
        )
    if not options.no_figures and not options.tables_only:
        options = replace(options, main_panels=True)
    return options


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate project report tables, figures, maps, ASCII exports, and QC files.",
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
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary_05 outputs. Default: 0.5.",
    )
    parser.add_argument(
        "--output-dir",
        "--report-output-root",
        dest="output_dir",
        default=None,
        help="Optional report output root. Default: 05_outputs/<project_id>/report_outputs.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Overwrite report outputs. Enabled by default.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if a report output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect inputs and estimate outputs without writing files.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Re-render and validate only the official Figure package from existing "
            "analysis/final-map outputs; do not process maps, tables, ASCII, or diagnostics."
        ),
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help=(
            "Generate and validate only the publication Table package from existing "
            "validation, analysis, and final-map outputs."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI. Default: 300.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure outputs.")
    parser.add_argument("--no-tables", action="store_true", help="Skip table outputs.")
    parser.add_argument("--no-maps", action="store_true", help="Skip probability/binary map copy and PNG outputs.")
    parser.add_argument("--no-asc", action="store_true", help="Skip ASC/TXT exports.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Skip diagnostics outputs.")
    parser.add_argument("--no-grids", action="store_true", help="Skip XYZ grid exports.")
    parser.add_argument(
        "--main-panels",
        action="store_true",
        default=False,
        help=(
            "Explicitly request individual main-figure panel outputs; full and "
            "--figures-only publication runs include them automatically."
        ),
    )
    parser.add_argument(
        "--main-panels-only",
        action="store_true",
        default=False,
        help="Generate Main Fig.1-Fig.9 as individual panels instead of composite/supplementary figures.",
    )
    parser.add_argument(
        "--panel-label",
        dest="panel_label",
        action="store_true",
        default=False,
        help="Prefix main panel titles with panel letters such as (a). Disabled by default for Illustrator assembly.",
    )
    parser.add_argument(
        "--no-panel-label",
        dest="panel_label",
        action="store_false",
        help="Do not prefix main panel titles with panel letters.",
    )
    parser.add_argument(
        "--max-preview-size",
        type=int,
        default=2000,
        help="Maximum preview raster dimension for figures. Default: 2000.",
    )
    parser.add_argument(
        "--allow-full-xyz",
        action="store_true",
        help="Write full-resolution XYZ files. This can be extremely large.",
    )
    parser.add_argument(
        "--xyz-stride",
        type=int,
        default=None,
        help="Enable stride-sampled XYZ export with this stride. Recommended: 10.",
    )
    parser.add_argument(
        "--save-vector",
        dest="save_vector",
        action="store_true",
        default=True,
        help="Save PDF vector copies of figures. Enabled by default.",
    )
    parser.add_argument(
        "--no-save-vector",
        dest="save_vector",
        action="store_false",
        help="Do not save PDF vector copies of figures.",
    )
    parser.add_argument("--map-cmap", default="viridis", help="Probability map colormap. Default: viridis.")
    parser.add_argument(
        "--difference-cmap",
        default="RdBu_r",
        help="Difference map diverging colormap. Default: RdBu_r.",
    )
    parser.add_argument(
        "--hist-log-y",
        action="store_true",
        default=False,
        help="Use log y-scale for histogram/distribution figures.",
    )
    parser.add_argument(
        "--skip-native-asc",
        action="store_true",
        default=False,
        help="Skip native-resolution ASC/TXT exports.",
    )
    parser.add_argument(
        "--skip-0p5m-asc",
        action="store_true",
        default=False,
        help="Skip 0.5 m resampled ASC/TXT exports.",
    )
    parser.add_argument(
        "--generate-paper-manifest",
        action="store_true",
        default=True,
        help="Generate paper_manifest CSV files. Enabled by default.",
    )
    parser.add_argument(
        "--generate-captions",
        action="store_true",
        default=True,
        help="Generate figure/table caption sidecar text files. Enabled by default.",
    )
    parser.add_argument(
        "--generate-source-data",
        action="store_true",
        default=True,
        help="Generate figure source-data sidecar CSV files. Enabled by default.",
    )
    parser.add_argument(
        "--generate-metadata",
        action="store_true",
        default=True,
        help="Generate figure/table metadata sidecar text files. Enabled by default.",
    )
    return parser.parse_args()


def main() -> None:
    """Run stage 13 from the command line."""

    args = parse_args()
    options = ReportOptions(
        project_id=str(args.project_id),
        site_filter=args.site,
        candidate_label_filter=args.candidate_label,
        threshold=float(args.threshold),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        dpi=int(args.dpi),
        no_figures=bool(args.no_figures),
        no_tables=bool(args.no_tables),
        no_maps=bool(args.no_maps),
        no_asc=bool(args.no_asc),
        no_diagnostics=bool(args.no_diagnostics),
        no_grids=bool(args.no_grids),
        max_preview_size=int(args.max_preview_size),
        allow_full_xyz=bool(args.allow_full_xyz),
        xyz_stride=args.xyz_stride,
        save_vector=bool(args.save_vector),
        map_cmap=str(args.map_cmap),
        difference_cmap=str(args.difference_cmap),
        hist_log_y=bool(args.hist_log_y),
        skip_native_asc=bool(args.skip_native_asc),
        skip_0p5m_asc=bool(args.skip_0p5m_asc),
        generate_paper_manifest=bool(args.generate_paper_manifest),
        generate_captions=bool(args.generate_captions),
        generate_source_data=bool(args.generate_source_data),
        generate_metadata=bool(args.generate_metadata),
        main_panels=bool(args.main_panels),
        main_panels_only=bool(args.main_panels_only),
        panel_label=bool(args.panel_label),
        figures_only=bool(args.figures_only),
        tables_only=bool(args.tables_only),
    )
    run_report_tables_figures(options)


if __name__ == "__main__":
    main()

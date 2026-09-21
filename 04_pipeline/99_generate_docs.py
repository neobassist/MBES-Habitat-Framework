#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Synchronize project documentation from validated configuration and outputs.

Stage 99 updates only explicitly marked AUTO sections. It preserves MANUAL,
HYBRID, and unmarked content, and never changes scientific pipeline outputs.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from utils.config_io import (
    DEFAULT_PROJECT_ID,
    RuntimePathBinding,
    configured_analysis_candidates,
    configured_final_mapping_policy,
    configured_figure_specs,
    configured_publication_contract,
    configured_representative_runs,
    configured_validation_sources,
    expand_run_matrix,
    expand_spatial_fold_run_matrix,
    get_git_metadata,
    load_all_configs,
    publication_display_labels,
    resolve_project_paths,
    resolve_site_input_paths,
    runtime_path_binding_from_environment,
)


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DISPLAY_PATH = Path("04_pipeline/99_generate_docs.py")
MARKER_RE = re.compile(
    r"^<!--\s*(AUTO|MANUAL|HYBRID):([a-z0-9][a-z0-9-]*):(start|end)\s*-->\s*$"
)
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
TIMESTAMP_RE = re.compile(r"^- Synchronization time: `([^`]+)`$", re.MULTILINE)
HEALTH_TIMESTAMP_RE = re.compile(r"^- Checked at: `([^`]+)`$", re.MULTILINE)
HEALTH_DOCUMENT = "health_report.md"
CHANGELOG_DOCUMENT = "changelog.md"
CHANGELOG_COMMIT_LIMIT = 20

PROJECT_DOCUMENTS = {
    "README": "README.md",
    "pipeline_manual": "pipeline_manual.md",
    "feature_definitions": "feature_definitions.md",
    "experiment_catalog": "experiment_catalog.md",
}

DOCUMENTATION_SWITCHES = {
    "README": "auto_generate_readme",
    "pipeline_manual": "auto_generate_pipeline_manual",
    "feature_definitions": "auto_generate_feature_definitions",
    "experiment_catalog": "auto_generate_experiment_catalog",
}

EXPECTED_AUTO_SECTIONS = {
    "README": (
        "project-metadata",
        "site-summary",
        "configuration-links",
        "pipeline-summary",
        "output-summary",
        "health-snapshot",
        "document-index",
        "provenance",
    ),
    "pipeline_manual": (
        "pipeline-stage-table",
        "pipeline-dependency",
        "cli-reference",
        "rebuild-guidance",
        "output-readers",
        "provenance",
    ),
    "feature_definitions": (
        "feature-groups",
        "feature-inventory",
        "feature-set-matrix",
        "feature-yaml-index",
        "provenance",
    ),
    "experiment_catalog": (
        "experiment-matrix",
        "validation-design",
        "run-counts",
        "directory-layout",
        "output-coverage",
        "representative-validation-comparison",
        "provenance",
    ),
}


@dataclass(frozen=True)
class StageSpec:
    number: str
    script: str
    purpose: str
    inputs: str
    outputs: str
    downstream: str
    cost: str


STAGE_SPECS = (
    StageSpec("01", "01_align.py", "Align/crop MBES rasters and materialize common-valid physical rasters.", "Raw bathymetry, backscatter, crop polygon", "prepared/ and common_valid_rasters/", "Stage 02 and physical-data users", "High"),
    StageSpec("02", "02_base_features.py", "Generate configured aligned base-feature rasters per site.", "prepared/ and project feature settings", "base_features/", "Stage 03", "High"),
    StageSpec("03", "03_feature_sets.py", "Assemble ordered multi-band feature stacks.", "base_features/ and feature-set YAML", "feature_sets/", "Stages 04 and 12", "High"),
    StageSpec("04", "04_sampling.py", "Rasterize labels and create finite ML samples.", "feature_sets/, labels, site settings", "samples/", "Stages 05, 07, 09, and 12", "High"),
    StageSpec("05", "05_train_within_site.py", "Train within-site held-out models and predictions.", "samples/ and model YAML", "within_site/ training artifacts", "Stage 06", "Very high"),
    StageSpec("06", "06_evaluate_within_site.py", "Evaluate within-site held-out predictions.", "within_site predictions", "within_site/ metrics and curves", "Stages 11 and 13", "High"),
    StageSpec("07", "07_train_cross_site.py", "Train directional cross-site transfer models.", "samples/ and model YAML", "cross_site/ training artifacts", "Stage 08", "Very high"),
    StageSpec("08", "08_evaluate_cross_site.py", "Evaluate complete target-site transfer predictions.", "cross_site predictions", "cross_site/ metrics and curves", "Stages 11 and 13", "High"),
    StageSpec("09", "09_train_spatial_block_cv.py", "Train spatial block CV folds.", "samples/ and spatial matrix", "spatial_block_cv/ training artifacts", "Stage 10", "Very high"),
    StageSpec("10", "10_evaluate_spatial_block_cv.py", "Evaluate spatial block CV fold predictions.", "spatial fold predictions", "spatial_block_cv/ metrics and curves", "Stages 11 and 13", "High"),
    StageSpec("11", "11_analyze_validation_results.py", "Integrate validation rankings and select candidates.", "Within/cross/spatial summaries", "analysis_compare/", "Stages 12 and 13", "Medium"),
    StageSpec("12", "12_predict_maps.py", "Retrain selected site models and predict full rasters.", "samples/, feature stacks, candidate comparison", "final_maps/", "Stage 13", "Very high"),
    StageSpec("13", "13_report_tables_figures.py", "Regenerate publication reports from saved results.", "Validation, analysis, and final-map outputs", "report_outputs/", "Publication use and Stage 99 summaries", "High"),
    StageSpec("99", "99_generate_docs.py", "Synchronize marked project documentation.", "YAML, pipeline code, manifests, Git", "Project docs and optional output READMEs", "Project users", "Low"),
)


@dataclass(frozen=True)
class MarkerBlock:
    kind: str
    name: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CheckResult:
    status: str
    category: str
    check_name: str
    severity: str
    expected: str
    actual: str
    message: str
    source_path: str


@dataclass
class SyncContext:
    project_id: str
    framework_root: Path
    runtime_binding: RuntimePathBinding | None
    project_config: dict[str, Any]
    sites_config: dict[str, Any]
    matrix: dict[str, Any]
    analysis_config: dict[str, Any]
    publication_config: dict[str, Any]
    feature_sets: dict[str, dict[str, Any]]
    models: dict[str, dict[str, Any]]
    paths: dict[str, Path]
    git: dict[str, Any]
    manifests: dict[str, list[dict[str, str]]]
    coverage: dict[str, dict[str, int]]
    run_counts: dict[str, int]
    stage_clis: dict[str, list[str]]
    warnings: list[str]
    checks: list[CheckResult]
    report_counts: dict[str, int]
    validation_plans: dict[str, tuple[dict[str, Any], ...]]
    publication_contract: dict[str, int]
    sync_time: str


@dataclass(frozen=True)
class OutputReadmeSpec:
    folder: str
    purpose: str
    created_by: str
    used_by: str
    structure: str
    key_files: tuple[str, ...]
    interpretation: str
    rebuild: str
    preservation: str
    filename: str = "README.md"


OUTPUT_README_SPECS = (
    OutputReadmeSpec(
        "prepared",
        "Common-grid, crop-bounding-box bathymetry and backscatter inputs for each site; physical rasters are not pixel-mask-applied.",
        "Stage 01 (`01_align.py`).",
        "Stage 02 reads aligned rasters and the common valid mask.",
        "`<site_id>/` with aligned configured-frequency rasters, grid metadata, and summaries.",
        ("`depth_<frequency>_aligned.tif`", "`backscatter_<frequency>_aligned.tif`", "`common_valid_mask.tif`", "alignment summary CSV files"),
        "All frequencies within a site share CRS, transform, width, and height. `common_valid_mask.tif` separately defines the intersected analysis footprint.",
        "Rebuild from raw site rasters and crop polygons by rerunning Stage 01.",
        "KEEP. This is the aligned upstream basis for all derived features.",
    ),
    OutputReadmeSpec(
        "common_valid_rasters",
        "Native-resolution physical bathymetry and backscatter with the Stage 01 common-valid mask explicitly applied.",
        "Stage 01 (`01_align.py`).",
        "Physical-data/GIS readers; Stage 02 remains compatible with the historical `prepared/` plus mask access path.",
        "`<site_id>/` directly contains the mask and configured physical variables as GeoTIFF, ASC, and identical TXT files.",
        (
            "`<site_id>/common_valid_mask.{tif,asc,txt}`",
            "`<site_id>/bathymetry_<frequency>.{tif,asc,txt}`",
            "`<site_id>/backscatter_<frequency>.{tif,asc,txt}`",
            "`common_valid_rasters_summary.csv`",
        ),
        "Inside the mask values equal `prepared/` exactly; outside is NoData. ASC/TXT rows run north-to-south on the native projected-metre grid and no ML transform or additional resampling is applied.",
        "Rebuild from existing `prepared/` outputs with Stage 01 `--common-valid-only`, or create it during a fresh Stage 01 run.",
        "KEEP for new runs. Historical runs without this directory remain readable through `prepared/` plus `common_valid_mask.tif`.",
    ),
    OutputReadmeSpec(
        "base_features",
        "Single-band acoustic, frequency-difference, bathymetric, and texture features.",
        "Stage 02 (`02_base_features.py`).",
        "Stage 03 assembles these rasters into configured feature stacks.",
        "`<site_id>/<feature_name>.tif` plus a site-level feature summary.",
        ("Configured single-band feature GeoTIFFs per site", "`base_feature_summary.csv`"),
        "Features retain the Stage 01 grid and use finite values inside the common valid mask.",
        "Rebuild from `prepared/` and `config_project.yaml` by rerunning Stage 02.",
        "KEEP. Rebuilding requires large raster calculations.",
    ),
    OutputReadmeSpec(
        "feature_sets",
        "Configured ordered multi-band feature stacks used by sampling and mapping.",
        "Stage 03 (`03_feature_sets.py`).",
        "Stages 04 and 12 read stack bands in the stored order.",
        "`<site_id>/<feature_set>.tif` with band-name and summary sidecars.",
        ("`<feature_set>.tif`", "`<feature_set>.bands.txt`", "`<feature_set>_summary.csv`"),
        "Band order follows acoustic, frequency-difference, terrain, and texture order in feature-set YAML.",
        "Rebuild from `base_features/` and feature-set YAML by rerunning Stage 03.",
        "KEEP. These stacks are upstream inputs to samples and final maps.",
    ),
    OutputReadmeSpec(
        "samples",
        "Finite labeled sample tables for each site and feature set.",
        "Stage 04 (`04_sampling.py`).",
        "Stages 05, 07, 09, and 12 read the sample CSV files.",
        "`<site_id>/<feature_set>_samples.csv` and matching sample summaries.",
        ("`<feature_set>_samples.csv`", "`<feature_set>_sample_summary.csv`"),
        "Positive labels are clipped to the finite feature area; negatives are sampled from finite background.",
        "Rebuild from `feature_sets/`, labels, and sampling configuration by rerunning Stage 04.",
        "KEEP. Samples are the common input to every model-training design.",
    ),
    OutputReadmeSpec(
        "within_site",
        "Within-site random-stratified training and held-out test evaluation results.",
        "Stages 05 and 06.",
        "Stages 11 and 13 read summaries, curves, confusion matrices, and representative predictions.",
        "`<site>/<feature_set>/<model>/seed_<seed>/` plus model and root summaries.",
        ("`model.joblib`", "`predictions.csv`", "`evaluation_metrics.csv`", "PR/ROC curves", "feature importance and summaries"),
        "Evaluation thresholds use the configured precision-target policy; report maps and representative publication confusion matrices use the configured fixed report threshold.",
        "Evaluation can be rebuilt from predictions; training requires `samples/` and is expensive.",
        "KEEP all splits, models, predictions, metrics, curves, and summaries.",
    ),
    OutputReadmeSpec(
        "cross_site",
        "Directional transfer validation using a complete target-site sample set.",
        "Stages 07 and 08.",
        "Stages 11 and 13 consume direction-aware summaries and representative predictions.",
        "`train_<train_site>_test_<test_site>/<feature_set>/<model>/seed_<seed>/`.",
        ("`model.joblib`", "`predictions.csv`", "`evaluation_metrics.csv`", "PR/ROC curves", "feature importance and summaries"),
        "Transfer directions remain separate. Evaluation thresholds are selected per run; representative publication confusion matrices use the configured fixed report threshold.",
        "Evaluation can be rebuilt from predictions; retraining from both sites' samples is expensive.",
        "KEEP all directional models, predictions, metrics, curves, and summaries.",
    ),
    OutputReadmeSpec(
        "spatial_block_cv",
        "Within-site configured-fold spatial block cross-validation results.",
        "Stages 09 and 10.",
        "Stages 11 and 13 read fold summaries and pooled representative predictions.",
        "`<site>/<feature_set>/<model>/seed_<seed>/fold_<fold>/`.",
        ("fold-level `model.joblib`", "`predictions.csv`", "`evaluation_metrics.csv`", "PR/ROC curves", "root summaries"),
        "A pooled OOF result combines the configured test folds for one configuration. Publication representative runs are selected by `publication.yaml`.",
        "Evaluation can be rebuilt from fold predictions; fold training is very expensive.",
        "KEEP every configured seed and fold for spatial-validation reproducibility.",
    ),
    OutputReadmeSpec(
        "analysis_compare",
        "Integrated Stage 11 validation comparisons, rankings, stability, and candidate selection.",
        "Stage 11 (`11_analyze_validation_results.py`).",
        "Stages 12 and 13 read final-candidate and integrated analysis tables.",
        "Root CSV files grouped by Stage-1 comparisons and Stage-2 integrated analysis.",
        ("`stage2_final_candidate_comparison.csv`", "integrated model/feature statistics", "rank stability and shift tables"),
        "These tables summarize stored validation outputs; they do not replace seed/fold prediction records.",
        "Rebuild from complete Stage 06, 08, and 10 root summaries.",
        "KEEP. Candidate selection and publication reports depend on these tables.",
    ),
    OutputReadmeSpec(
        "final_maps",
        "Final site-specific model artifacts and full-area probability predictions.",
        "Stage 12 (`12_predict_maps.py`).",
        "Stage 13 uses these as upstream inputs for publication maps, statistics, and GIS/ASCII exports.",
        "`<site>/<candidate_label>/` for every selected final candidate.",
        ("`final_model.joblib`", "`prediction_probability.tif`", "`prediction_valid_mask.tif`", "feature importance", "training summary", "optional `prediction_preview.png`"),
        "Stage 12 does not create new validation scores. The optional QC preview carries site/candidate/model context, grid-north and a projected-coordinate scale bar; Stage 13 does not require it. Stage 13 binary products use the configured fixed report threshold.",
        "Rebuild from samples, feature stacks, model YAML, and the final-candidate comparison; this is very expensive.",
        "KEEP all candidate models, rasters, masks, importance values, and summaries. Previews are reproducible optional QC derivatives.",
    ),
)


REPORT_README_SPECS = (
    OutputReadmeSpec(
        "report_outputs",
        "Canonical publication reports regenerated from stored analysis and final-map results.",
        "Stage 13 (`13_report_tables_figures.py`); synchronized documentation markers are maintained by Stage 99.",
        "Project users, publication assembly, GIS workflows, and Stage 99 health checks.",
        "`figures/`, `tables/`, `maps/`, `diagnostics/`, `asciis/`, and `paper_manifest/`.",
        ("Main and Supplementary Figure composites/panels", "official, source-data, and internal/QC tables", "map and ASCII exports", "paper manifests and contract validation"),
        "This directory contains derived report products, not model training or validation execution.",
        "Rebuild from saved Stage 06/08/10/11/12 outputs by rerunning Stage 13.",
        "KEEP official report products. Preserve all native and Stage 13 fixed-contract 0.5 m ASC/TXT exports for Surfer workflows.",
        "README_report_outputs.md",
    ),
    OutputReadmeSpec(
        "report_outputs/figures",
        "Canonical Main and Supplementary publication Figures and vector-preserving panel exports.",
        "Stage 13; synchronized documentation markers are maintained by Stage 99.",
        "Publication assembly and paper manifests.",
        "`main/composites/`, `main/panels/`, `supplementary/composites/`, and `supplementary/panels/`.",
        (
            "PNG 600 dpi",
            "SVG and PDF vector outputs",
            "portable source CSV",
            "publication caption TXT",
            "metadata TXT with canonical Table and Source Data pairing",
        ),
        "Every official composite and panel follows the six-file contract recorded in `paper_manifest/`. Standalone panels are cropped from the rendered composite axes where feasible. Map Figures use grid/map north and projected-coordinate scale bars. Graphics share the Arial, Helvetica, then DejaVu Sans runtime font policy, with the actual resolved family recorded in `paper_manifest/typography.txt`.",
        "Rerun Stage 13 from saved scientific outputs; no model retraining is required.",
        "KEEP official Figure files and sidecars. Legacy flat outputs are not active results.",
        "README_output_structure.md",
    ),
    OutputReadmeSpec(
        "report_outputs/tables",
        "Official Main/Supplementary Tables, canonical source data, and internal/QC tables.",
        "Stage 13; synchronized documentation markers are maintained by Stage 99.",
        "Publication assembly, Figure generation, and reproducibility review.",
        "`main/`, `supplementary/`, `source_data/{validation,figures,maps,inventories}/`, and `internal/{supporting,qc}/`.",
        (
            "official CSV data",
            "caption TXT",
            "metadata TXT",
            "dictionary Markdown",
            "TABLE_INDEX.md and contract validation",
        ),
        "Only Main and Supplementary Tables carry paper numbers; source and internal artifacts are unnumbered. Official tables use a four-file contract.",
        "Rerun Stage 13 from stored validation, analysis, and final-map results.",
        "KEEP official tables and all source/metadata sidecars.",
        "README_table_structure.md",
    ),
    OutputReadmeSpec(
        "report_outputs/paper_manifest",
        "Machine-readable inventory and contract validation for official report artifacts.",
        "Stage 13; synchronized documentation markers are maintained by Stage 99.",
        "Stage 99 health checks and publication file assembly.",
        "CSV manifests for Figures, panels, Tables, mappings, recommended layout, and contract validation.",
        ("`main_figures.csv`", "`supplementary_figures.csv`", "panel manifests", "table manifests", "contract validation CSV files"),
        "Manifest paths are authoritative for official artifact inventory and must remain non-stale and non-zero-byte. Figure rows also resolve canonical paired Table and Source Data identifiers.",
        "Rerun Stage 13 to regenerate manifests from the current official report structure.",
        "KEEP. These files support contract checks and reproducible paper assembly.",
    ),
)


ALL_OUTPUT_README_SPECS = OUTPUT_README_SPECS + REPORT_README_SPECS


class DocumentationError(RuntimeError):
    """Raised when synchronization cannot proceed safely."""


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DocumentationError(f"{name} must be a mapping.")
    return value


def _cell(value: Any) -> str:
    if value is None:
        return "NA"
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _code(value: Any) -> str:
    return f"`{_cell(value)}`"


def _link(label: str, source: Path, target: Path) -> str:
    relative = os.path.relpath(target, source.parent)
    return f"[{label}]({Path(relative).as_posix()})"


def _runtime_attempt_root(context: SyncContext) -> Path:
    binding = context.runtime_binding
    if binding is None:
        raise DocumentationError("No split-root runtime binding is active.")
    output_root = binding.output_root.resolve(strict=False)
    docs_root = binding.docs_root.resolve(strict=False)
    if output_root.name != context.project_id or output_root.parent.name != "05_outputs":
        raise DocumentationError(
            f"Runtime output root does not match the attempt namespace: {output_root}"
        )
    attempt_root = output_root.parent.parent
    expected_docs_root = (
        attempt_root / "06_docs" / "projects" / context.project_id
    ).resolve(strict=False)
    if docs_root != expected_docs_root:
        raise DocumentationError(
            f"Runtime docs root does not match the attempt namespace: {docs_root}"
        )
    return attempt_root


def _bound_display_path(context: SyncContext, value: Path | str) -> str:
    """Render a stable path relative to its explicit Stage 99 authority root."""

    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()

    resolved = path.resolve(strict=False)
    binding = context.runtime_binding
    if binding is None:
        roots = ((context.framework_root.resolve(strict=False), False),)
    else:
        roots = (
            (_runtime_attempt_root(context), False),
            (binding.config_root.resolve(strict=False), True),
            (binding.framework_code_root.resolve(strict=False), False),
        )
    for root, immutable_config_root in roots:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        if immutable_config_root and relative.parts[:1] == ("runs",):
            continue
        return relative.as_posix()
    raise DocumentationError(
        f"Path is outside the bound Stage 99 runtime authorities: {resolved}"
    )


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _marker(kind: str, name: str, body: str) -> str:
    clean = body.rstrip("\n")
    return f"<!-- {kind}:{name}:start -->\n{clean}\n<!-- {kind}:{name}:end -->"


def _parse_markers(text: str, path: Path) -> dict[tuple[str, str], MarkerBlock]:
    lines = text.splitlines(keepends=True)
    blocks: dict[tuple[str, str], MarkerBlock] = {}
    open_marker: tuple[str, str, int] | None = None
    seen_names: set[str] = set()

    for index, line in enumerate(lines):
        match = MARKER_RE.match(line.rstrip("\r\n"))
        if not match:
            continue
        kind, name, boundary = match.groups()
        key = (kind, name)
        if boundary == "start":
            if open_marker is not None:
                raise DocumentationError(
                    f"Nested marker in {path}: {kind}:{name} starts before "
                    f"{open_marker[0]}:{open_marker[1]} ends."
                )
            if name in seen_names:
                raise DocumentationError(f"Duplicate marker name in {path}: {name}")
            seen_names.add(name)
            open_marker = (kind, name, index)
            continue

        if open_marker is None:
            raise DocumentationError(f"Marker end without start in {path}: {kind}:{name}")
        if (open_marker[0], open_marker[1]) != key:
            raise DocumentationError(
                f"Mismatched marker in {path}: expected end for "
                f"{open_marker[0]}:{open_marker[1]}, got {kind}:{name}."
            )
        blocks[key] = MarkerBlock(kind, name, open_marker[2], index)
        open_marker = None

    if open_marker is not None:
        raise DocumentationError(
            f"Marker start without end in {path}: {open_marker[0]}:{open_marker[1]}"
        )
    return blocks


def _replace_auto_sections(
    text: str,
    path: Path,
    replacements: Mapping[str, str],
) -> str:
    lines = text.splitlines(keepends=True)
    blocks = _parse_markers(text, path)
    auto_names = {name for kind, name in blocks if kind == "AUTO"}
    expected = set(replacements)
    missing = sorted(expected - auto_names)
    extra = sorted(auto_names - expected)
    if missing:
        raise DocumentationError(f"Missing AUTO marker(s) in {path}: {', '.join(missing)}")
    if extra:
        raise DocumentationError(f"Unmanaged AUTO marker(s) in {path}: {', '.join(extra)}")

    ordered = sorted(
        (blocks[("AUTO", name)] for name in expected),
        key=lambda block: block.start_line,
        reverse=True,
    )
    for block in ordered:
        body = replacements[block.name].rstrip("\n") + "\n"
        lines[block.start_line + 1 : block.end_line] = body.splitlines(keepends=True)
    return "".join(lines)


def _atomic_write(path: Path, content: str) -> bool:
    encoded = content.encode("utf-8")
    if path.exists() and path.read_bytes() == encoded:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return True


def _unified_diff(path: Path, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _extract_cli_flags(script_path: Path) -> list[str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("--"):
                    flags.add(argument.value)
    return sorted(flags)


def _manifest_inventory(output_root: Path, warnings: list[str]) -> dict[str, list[dict[str, str]]]:
    manifest_root = output_root / "report_outputs" / "paper_manifest"
    names = (
        "main_figures.csv",
        "main_panel_figures.csv",
        "supplementary_figures.csv",
        "supplementary_panel_figures.csv",
        "main_tables.csv",
        "supplementary_tables.csv",
        "source_data_tables.csv",
        "internal_tables.csv",
        "figure_contract_validation.csv",
        "table_contract_validation.csv",
        "source_data_contract_validation.csv",
        "figure_file_mapping.csv",
        "table_file_mapping.csv",
    )
    manifests: dict[str, list[dict[str, str]]] = {}
    for name in names:
        path = manifest_root / name
        rows = _read_csv(path)
        manifests[name] = rows
        if not path.exists():
            warnings.append(f"Optional report manifest is missing: {path}")
            continue
        path_columns = [
            column
            for column in (
                "png_path",
                "svg_path",
                "pdf_path",
                "source_path",
                "caption_path",
                "metadata_path",
                "csv_path",
                "dictionary_path",
            )
            if rows and column in rows[0]
        ]
        for row_index, row in enumerate(rows, start=2):
            for column in path_columns:
                value = row.get(column, "").strip()
                if value.upper() == "NA":
                    continue
                manifest_path = Path(value)
                if value and not manifest_path.is_absolute():
                    manifest_path = output_root / "report_outputs" / manifest_path
                if value and not manifest_path.exists():
                    raise DocumentationError(f"Stale manifest path in {path}:{row_index} ({column}): {value}")
    return manifests


def _build_validation_plans(
    matrix: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Build active validation inventories through the shared execution planners."""

    non_spatial = expand_run_matrix(matrix)
    return {
        "within_site": tuple(
            run for run in non_spatial if run["validation_type"] == "within_site"
        ),
        "cross_site": tuple(
            run for run in non_spatial if run["validation_type"] == "cross_site"
        ),
        "spatial_block_cv": tuple(expand_spatial_fold_run_matrix(matrix)),
    }


def _validation_run_counts(
    plans: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    counts = {name: len(plans[name]) for name in ("within_site", "cross_site", "spatial_block_cv")}
    counts["non_spatial_total"] = counts["within_site"] + counts["cross_site"]
    counts["validation_total"] = counts["non_spatial_total"] + counts["spatial_block_cv"]
    return counts


def _expected_output_paths(context: SyncContext) -> dict[str, list[tuple[Path, Path]]]:
    root = context.paths["output_root"]
    pairs: dict[str, list[tuple[Path, Path]]] = {"within_site": [], "cross_site": [], "spatial_block_cv": []}
    for run in context.validation_plans["within_site"]:
        base = root / "within_site" / run["train_site"] / run["feature_set"] / run["model"] / f"seed_{run['seed']}"
        pairs["within_site"].append((base / "predictions.csv", base / "evaluation_metrics.csv"))
    for run in context.validation_plans["cross_site"]:
        pair_id = f"train_{run['train_site']}_test_{run['test_site']}"
        base = root / "cross_site" / pair_id / run["feature_set"] / run["model"] / f"seed_{run['seed']}"
        pairs["cross_site"].append((base / "predictions.csv", base / "evaluation_metrics.csv"))
    for run in context.validation_plans["spatial_block_cv"]:
        base = root / "spatial_block_cv" / run["site_id"] / run["feature_set"] / run["model_id"] / f"seed_{run['seed']}" / f"fold_{run['fold_id']}"
        pairs["spatial_block_cv"].append((base / "predictions.csv", base / "evaluation_metrics.csv"))
    return pairs


def _coverage(context: SyncContext) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for validation_type, expected_paths in _expected_output_paths(context).items():
        result[validation_type] = _artifact_coverage(expected_paths)
        expected_count = result[validation_type]["expected"]
        prediction_count = result[validation_type]["predictions"]
        evaluation_count = result[validation_type]["evaluations"]
        importance_count = result[validation_type]["feature_importance"]
        if (
            prediction_count != expected_count
            or evaluation_count != expected_count
            or importance_count != expected_count
        ):
            context.warnings.append(
                f"{validation_type} output coverage is incomplete: expected={expected_count}, "
                f"predictions={prediction_count}, evaluations={evaluation_count}, "
                f"feature_importance={importance_count}."
            )
    return result


def _artifact_coverage(expected_paths: Sequence[tuple[Path, Path]]) -> dict[str, int]:
    return {
        "expected": len(expected_paths),
        "predictions": sum(
            prediction.is_file() and prediction.stat().st_size > 0
            for prediction, _ in expected_paths
        ),
        "evaluations": sum(
            evaluation.is_file() and evaluation.stat().st_size > 0
            for _, evaluation in expected_paths
        ),
        "feature_importance": sum(
            (prediction.parent / "feature_importance.csv").is_file()
            and (prediction.parent / "feature_importance.csv").stat().st_size > 0
            for prediction, _ in expected_paths
        ),
        "estimators": sum(
            (prediction.parent / "model.joblib").is_file()
            and (prediction.parent / "model.joblib").stat().st_size > 0
            for prediction, _ in expected_paths
        ),
    }


def _scientific_validation_actual(coverage: Mapping[str, int]) -> int:
    return min(
        coverage["predictions"],
        coverage["evaluations"],
        coverage["feature_importance"],
    )


def _validation_coverage_checks(
    validation_type: str,
    coverage: Mapping[str, int],
    source_path: Path,
) -> tuple[CheckResult, CheckResult]:
    expected = coverage["expected"]
    scientific_actual = _scientific_validation_actual(coverage)
    scientific_severity = "INFO" if scientific_actual == expected else "ERROR"
    return (
        _check_result(
            "Pipeline and validation coverage",
            f"{validation_type} complete runs",
            scientific_severity,
            expected,
            scientific_actual,
            f"predictions={coverage['predictions']}, evaluations={coverage['evaluations']}, "
            f"feature_importance={coverage['feature_importance']}",
            source_path,
        ),
        _check_result(
            "Pipeline and validation coverage",
            f"{validation_type} estimator retention",
            "INFO",
            expected,
            coverage["estimators"],
            "Historical validation estimators retained locally; absence does not affect scientific completeness.",
            source_path,
        ),
    )


def _publication_manifest_counts(
    manifests: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, int]:
    return {
        "main_figures": len(manifests.get("main_figures.csv", [])),
        "supplementary_figures": len(manifests.get("supplementary_figures.csv", [])),
        "main_panels": len(manifests.get("main_panel_figures.csv", [])),
        "supplementary_panels": len(manifests.get("supplementary_panel_figures.csv", [])),
        "main_tables": len(manifests.get("main_tables.csv", [])),
        "supplementary_tables": len(manifests.get("supplementary_tables.csv", [])),
        "source_data": len(manifests.get("source_data_tables.csv", [])),
    }


def _publication_inventory_mismatches(
    expected: Mapping[str, int],
    manifests: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, tuple[int, int]]:
    actual = _publication_manifest_counts(manifests)
    return {
        key: (expected[key], actual[key])
        for key in expected
        if actual.get(key) != expected[key]
    }


def _expected_final_map_directories(
    output_root: Path,
    analysis_config: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> tuple[Path, ...]:
    mapping = configured_final_mapping_policy(analysis_config, matrix)
    candidates = configured_analysis_candidates(analysis_config)
    return tuple(
        output_root / "final_maps" / site_id / candidate["candidate_label"]
        for site_id in mapping["sites"]
        for candidate in candidates
    )


def _check_result(
    category: str,
    check_name: str,
    severity: str,
    expected: Any,
    actual: Any,
    message: str,
    source_path: Path | str,
) -> CheckResult:
    normalized = severity.upper()
    return CheckResult(
        status={"ERROR": "FAIL", "WARNING": "WARNING"}.get(normalized, "PASS"),
        category=category,
        check_name=check_name,
        severity=normalized,
        expected=str(expected),
        actual=str(actual),
        message=message,
        source_path=str(source_path),
    )


def _git_changed_paths(root: Path) -> set[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    paths: set[str] = set()
    entries = result.stdout.split("\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        status = entry[:2]
        path = entry[3:]
        if path:
            paths.add(path)
        if "R" in status or "C" in status:
            if index < len(entries) and entries[index]:
                paths.add(entries[index])
                index += 1
    return paths


def _git_scope_status(context: SyncContext) -> dict[str, Any]:
    changed = _git_changed_paths(context.framework_root)
    reproducibility_metadata = {
        "environment.yml",
        "REPRODUCIBILITY.md",
        "04_pipeline/00_preflight.py",
        "04_pipeline/utils/preflight.py",
    }
    scientific_stage_paths = {
        f"04_pipeline/{stage.script}" for stage in STAGE_SPECS if stage.number not in {"13", "99"}
    }
    scientific_changed = sorted(
        path
        for path in changed
        if (
            path.startswith(("02_projects/", "03_experiments/"))
            and path not in reproducibility_metadata
        )
        or path in scientific_stage_paths
    )
    docs_prefix = f"06_docs/projects/{context.project_id}"
    managed_docs = {
        f"{docs_prefix}/README.md",
        f"{docs_prefix}/pipeline_manual.md",
        f"{docs_prefix}/feature_definitions.md",
        f"{docs_prefix}/experiment_catalog.md",
        f"{docs_prefix}/health_report.md",
        f"{docs_prefix}/changelog.md",
    }
    stage99_scope = managed_docs | {
        ".gitignore",
        "04_pipeline/13_report_tables_figures.py",
        "04_pipeline/99_generate_docs.py",
    } | reproducibility_metadata
    return {
        "changed_paths": changed,
        "scientific_changed": scientific_changed,
        "managed_docs_only": changed.issubset(managed_docs),
        "stage99_scope_only": changed.issubset(stage99_scope),
    }


def _report_output_counts(context: SyncContext) -> dict[str, int]:
    report_root = context.paths["output_root"] / "report_outputs"
    contract = context.publication_contract
    manifest_counts = _publication_manifest_counts(context.manifests)
    return {
        "main_figures_expected": contract["main_figures"],
        "main_figures_manifest": manifest_counts["main_figures"],
        "main_figures_actual": len(list((report_root / "figures/main/composites").glob("*.png"))),
        "main_panels_expected": contract["main_panels"],
        "main_panels_manifest": manifest_counts["main_panels"],
        "main_panels_actual": len(list((report_root / "figures/main/panels").rglob("*.png"))),
        "supplementary_figures_expected": contract["supplementary_figures"],
        "supplementary_figures_manifest": manifest_counts["supplementary_figures"],
        "supplementary_figures_actual": len(list((report_root / "figures/supplementary/composites").glob("*.png"))),
        "supplementary_panels_expected": contract["supplementary_panels"],
        "supplementary_panels_manifest": manifest_counts["supplementary_panels"],
        "supplementary_panels_actual": len(list((report_root / "figures/supplementary/panels").rglob("*.png"))),
        "main_tables_expected": contract["main_tables"],
        "main_tables_manifest": manifest_counts["main_tables"],
        "main_tables_actual": len(list((report_root / "tables/main").glob("*.csv"))),
        "supplementary_tables_expected": contract["supplementary_tables"],
        "supplementary_tables_manifest": manifest_counts["supplementary_tables"],
        "supplementary_tables_actual": len(list((report_root / "tables/supplementary").glob("*.csv"))),
        "source_data_tables_expected": contract["source_data"],
        "source_data_tables_manifest": manifest_counts["source_data"],
        "source_data_tables_actual": len(list((report_root / "tables/source_data").rglob("*.csv"))),
        "internal_tables_expected": len(context.manifests.get("internal_tables.csv", [])),
        "internal_tables_actual": len(list((report_root / "tables/internal").rglob("*.csv"))),
        "map_tif": len(list((report_root / "maps").rglob("*.tif"))),
        "map_png": len(list((report_root / "maps").rglob("*.png"))),
        "map_pdf": len(list((report_root / "maps").rglob("*.pdf"))),
        "asc": len(list((report_root / "asciis").rglob("*.asc"))),
        "txt": len(list((report_root / "asciis").rglob("*.txt"))),
    }


def _split_manifest_ids(value: str) -> list[str]:
    """Split a semicolon-delimited manifest relationship field."""

    text = str(value).strip()
    if not text or text.upper() == "NA":
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def _figure_publication_qa(context: SyncContext) -> list[CheckResult]:
    """Validate portable Figure sidecars, captions, and pairing identifiers."""

    report_root = context.paths["output_root"] / "report_outputs"
    figure_manifest_names = (
        "main_figures.csv",
        "supplementary_figures.csv",
        "main_panel_figures.csv",
        "supplementary_panel_figures.csv",
    )
    composite_manifest_names = ("main_figures.csv", "supplementary_figures.csv")
    forbidden = re.compile(
        r"/(?:Users|home)/[^/\s]+/|/private/tmp/|(?<![A-Za-z0-9_])/tmp/|miniconda|"  # portability-pattern-definition
        r"conda(?:/|\\)|staging|rollback",
        flags=re.IGNORECASE,
    )
    portability_failures: list[str] = []
    for manifest_name in figure_manifest_names:
        for row in context.manifests.get(manifest_name, []):
            for column in ("source_path", "metadata_path"):
                value = row.get(column, "").strip()
                if not value or value.upper() == "NA":
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = report_root / path
                if path.is_file() and forbidden.search(path.read_text(encoding="utf-8", errors="ignore")):
                    portability_failures.append(str(path))

    caption_failures: list[str] = []
    panel_counts = Counter(
        row.get("parent_figure", "")
        for name in ("main_panel_figures.csv", "supplementary_panel_figures.csv")
        for row in context.manifests.get(name, [])
    )
    for manifest_name in composite_manifest_names:
        for row in context.manifests.get(manifest_name, []):
            value = row.get("caption_path", "").strip()
            path = report_root / value
            if not path.is_file() or path.stat().st_size == 0:
                caption_failures.append(f"{row.get('figure_number')}: missing caption")
                continue
            text = path.read_text(encoding="utf-8").strip()
            words = re.findall(r"\b[\w-]+\b", text)
            title_only = len(words) < 25 or text.rstrip(".") == (
                f"{row.get('figure_number')}. {row.get('title')}"
            )
            panel_description_missing = (
                panel_counts.get(row.get("figure_number", ""), 0) > 1
                and not re.search(r"\bPanels?\s*\([a-z](?:-[a-z])?\)", text, flags=re.IGNORECASE)
                and not (
                    re.search(r"\bColumns?\b", text, flags=re.IGNORECASE)
                    and re.search(r"\bRows?\b", text, flags=re.IGNORECASE)
                )
            )
            if title_only or panel_description_missing:
                caption_failures.append(str(path))

    official_table_ids = {
        row.get("table_id", "")
        for name in ("main_tables.csv", "supplementary_tables.csv")
        for row in context.manifests.get(name, [])
    }
    source_data_ids = {
        row.get("table_id", "") for row in context.manifests.get("source_data_tables.csv", [])
    }
    pairing_failures: list[str] = []
    for manifest_name in figure_manifest_names:
        for row in context.manifests.get(manifest_name, []):
            figure_id = row.get("figure_number", "")
            if "paired_table_ids" not in row or "paired_source_data_ids" not in row:
                pairing_failures.append(f"{manifest_name}:{figure_id}: missing pairing columns")
                continue
            table_ids = _split_manifest_ids(row.get("paired_table_ids", ""))
            source_ids = _split_manifest_ids(row.get("paired_source_data_ids", ""))
            missing_tables = sorted(set(table_ids) - official_table_ids)
            missing_sources = sorted(set(source_ids) - source_data_ids)
            if missing_tables or missing_sources or not (table_ids or source_ids):
                pairing_failures.append(
                    f"{manifest_name}:{figure_id}: tables={missing_tables}, sources={missing_sources}"
                )

    system_files = [
        path
        for path in report_root.rglob("*")
        if path.is_file()
        and (
            path.name in {".DS_Store", "Thumbs.db", "desktop.ini"}
            or path.name.startswith("._")
        )
    ]
    return [
        _check_result(
            "Contracts and manifests",
            "Figure sidecar portability failures",
            "INFO" if not portability_failures else "ERROR",
            0,
            len(portability_failures),
            "Official Figure source and metadata sidecars contain no machine-specific paths.",
            report_root / "figures",
        ),
        _check_result(
            "Contracts and manifests",
            "Generic Figure caption failures",
            "INFO" if not caption_failures else "ERROR",
            0,
            len(caption_failures),
            "Official composite captions are non-empty, non-generic, and describe multi-panel Figures.",
            report_root / "figures",
        ),
        _check_result(
            "Contracts and manifests",
            "Figure pairing failures",
            "INFO" if not pairing_failures else "ERROR",
            0,
            len(pairing_failures),
            "Figure pairing fields resolve to canonical official Table or Source Data identifiers.",
            report_root / "paper_manifest",
        ),
        _check_result(
            "Contracts and manifests",
            "Publication system files",
            "INFO" if not system_files else "ERROR",
            0,
            len(system_files),
            "Publication outputs contain no operating-system metadata files.",
            report_root,
        ),
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sidecar_digest(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        return ""
    parts = path.read_text(encoding="ascii").strip().split()
    return parts[0] if parts else ""


def _frozen_authority_checks(context: SyncContext) -> list[CheckResult]:
    """Verify immutable Stage 12 and Stage 13 manifests without recomputation."""

    root = context.framework_root
    output_root = context.paths["output_root"]
    freeze_path = output_root / "final_mapping_freeze/final_mapping_freeze.json"
    freeze_sidecar = freeze_path.with_suffix(".sha256")
    report_root = output_root / "report_outputs"
    publication_path = report_root / "paper_manifest/publication_output_manifest.json"
    publication_sidecar = publication_path.with_suffix(".sha256")
    checks: list[CheckResult] = []

    freeze_errors: list[str] = []
    final_models = 0
    frozen_files = 0
    freeze_enabled = freeze_path.is_file()
    if freeze_enabled:
        actual_digest = _sha256_file(freeze_path)
        if _manifest_sidecar_digest(freeze_sidecar) != actual_digest:
            freeze_errors.append("freeze manifest sidecar hash mismatch")
        try:
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            freeze_errors.append(f"freeze manifest is unreadable: {exc}")
            freeze = {}
        expected_identities = {
            (path.parts[-2], path.parts[-1])
            for path in _expected_final_map_directories(
                output_root, context.analysis_config, context.matrix
            )
        }
        actual_identities: set[tuple[str, str]] = set()
        for branch in freeze.get("branches", []):
            identity = (str(branch.get("site_id", "")), str(branch.get("candidate_role", "")))
            actual_identities.add(identity)
            for name, record in branch.get("files", {}).items():
                path = root / str(record.get("path", ""))
                frozen_files += 1
                if name == "final_model.joblib":
                    final_models += 1
                if not path.is_file():
                    freeze_errors.append(f"missing frozen file: {path}")
                    continue
                if path.stat().st_size != int(record.get("bytes", -1)):
                    freeze_errors.append(f"frozen file size mismatch: {path}")
                elif _sha256_file(path) != record.get("sha256"):
                    freeze_errors.append(f"frozen file hash mismatch: {path}")
            raster = branch.get("raster", {})
            if raster.get("crs") != "EPSG:32652" or not raster.get("mask_equals_common_valid"):
                freeze_errors.append(f"frozen raster contract mismatch: {identity}")
            if not (0.0 <= float(raster.get("probability_min", -1.0)) <= 1.0):
                freeze_errors.append(f"frozen probability minimum invalid: {identity}")
            if not (0.0 <= float(raster.get("probability_max", 2.0)) <= 1.0):
                freeze_errors.append(f"frozen probability maximum invalid: {identity}")
        if actual_identities != expected_identities:
            freeze_errors.append("frozen branch identities do not match configured final mapping")
        if freeze.get("project_id") != context.project_id:
            freeze_errors.append("freeze manifest project identity mismatch")
        if int(freeze.get("final_map_inventory_count", -1)) != frozen_files:
            freeze_errors.append("freeze manifest file count mismatch")

    expected_final_dirs = _expected_final_map_directories(
        output_root, context.analysis_config, context.matrix
    )
    expected_final_models = len(expected_final_dirs)
    final_models = sum(
        (path / "final_model.joblib").is_file()
        and (path / "final_model.joblib").stat().st_size > 0
        for path in expected_final_dirs
    )
    checks.extend((
        _check_result(
            "Frozen authorities", "Final mapping freeze integrity",
            "INFO" if freeze_enabled and not freeze_errors else ("ERROR" if freeze_errors else "DEFERRED"),
            0 if freeze_enabled else "optional", len(freeze_errors) if freeze_enabled else "not configured",
            "; ".join(freeze_errors) if freeze_errors else (
                "Frozen Stage 12 files and candidate identities match their manifest."
                if freeze_enabled else "No final-mapping freeze manifest is configured for this project."
            ),
            freeze_path,
        ),
        _check_result(
            "Pipeline and validation coverage", "Final estimator retention",
            "INFO" if final_models == expected_final_models else "ERROR",
            expected_final_models, final_models,
            "Final site-candidate estimators are required independently of validation-estimator retention.",
            output_root / "final_maps",
        ),
    ))

    publication_errors: list[str] = []
    payload_count = 0
    publication_frozen = False
    if not publication_path.is_file():
        publication_errors.append("publication output manifest is missing")
    else:
        actual_digest = _sha256_file(publication_path)
        if _manifest_sidecar_digest(publication_sidecar) != actual_digest:
            publication_errors.append("publication manifest sidecar hash mismatch")
        try:
            publication = json.loads(publication_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            publication_errors.append(f"publication manifest is unreadable: {exc}")
            publication = {}
        if publication.get("project_id") != context.project_id:
            publication_errors.append("publication manifest project identity mismatch")
        publication_frozen = bool(
            freeze_enabled
            and publication.get("authorities", {}).get("final_mapping_freeze", {}).get("exists")
        )
        authority_paths = {
            "analysis_config": context.paths["experiment_root"] / "analysis.yaml",
            "candidate_authority": output_root / "analysis_compare/stage2_final_candidate_comparison.csv",
            "final_mapping_freeze": freeze_path,
            "publication_config": context.paths["experiment_root"] / "publication.yaml",
        }
        if publication_frozen:
            for name, path in authority_paths.items():
                record = publication.get("authorities", {}).get(name, {})
                if not path.is_file() or record.get("sha256") != _sha256_file(path):
                    publication_errors.append(f"publication authority mismatch: {name}")
        listed_paths: set[str] = set()
        for record in publication.get("output_files", []):
            relative = Path(str(record.get("path", "")))
            path = (report_root / relative).resolve()
            try:
                path.relative_to(report_root.resolve())
            except ValueError:
                publication_errors.append(f"publication payload escapes package root: {relative}")
                continue
            listed_paths.add(relative.as_posix())
            payload_count += 1
            if not path.is_file():
                publication_errors.append(f"missing publication payload: {relative}")
                continue
            if publication_frozen:
                if path.stat().st_size != int(record.get("size_bytes", -1)):
                    publication_errors.append(f"publication payload size mismatch: {relative}")
                elif _sha256_file(path) != record.get("sha256"):
                    publication_errors.append(f"publication payload hash mismatch: {relative}")
        if int(publication.get("output_file_count", -1)) != payload_count:
            publication_errors.append("publication payload count mismatch")
        actual_paths = {
            path.relative_to(report_root).as_posix()
            for path in report_root.rglob("*") if path.is_file()
        }
        manifest_paths = {
            publication_path.relative_to(report_root).as_posix(),
            publication_sidecar.relative_to(report_root).as_posix(),
        }
        if publication_frozen and actual_paths != listed_paths | manifest_paths:
            publication_errors.append("official report inventory differs from publication manifest")

    checks.append(_check_result(
        "Frozen authorities", "Publication output manifest integrity",
        "INFO" if publication_frozen and not publication_errors else ("ERROR" if publication_errors else "DEFERRED"),
        0 if publication_frozen else "optional", len(publication_errors) if publication_frozen else "not frozen",
        "; ".join(publication_errors) if publication_errors else (
            "All official publication payloads and authority hashes match the publication manifest."
            if publication_frozen else "Publication payload hashing is deferred until a final-mapping freeze authority is configured."
        ),
        publication_path,
    ))
    return checks


def _collect_health_checks(context: SyncContext) -> list[CheckResult]:
    checks: list[CheckResult] = []
    project_root = context.paths["project_root"]
    experiment_root = context.paths["experiment_root"]
    output_root = context.paths["output_root"]
    docs_root = context.paths["docs_root"]
    report_root = output_root / "report_outputs"

    for label, path in (
        ("Project configuration", project_root / "config_project.yaml"),
        ("Site configuration", project_root / "config_sites.yaml"),
        ("Experiment matrix", experiment_root / "experiment_matrix.yaml"),
        ("Analysis configuration", experiment_root / "analysis.yaml"),
        ("Publication configuration", experiment_root / "publication.yaml"),
    ):
        checks.append(_check_result("Configuration health", label, "INFO", "present and parseable", "present and parseable", "Validated by the shared Stage 99 configuration loader.", path))

    feature_expected = len(context.matrix["feature_sets"])
    model_expected = len(context.matrix["models"])
    checks.extend((
        _check_result("Configuration health", "Feature-set YAML coverage", "INFO", feature_expected, len(context.feature_sets), "Feature-set definitions match the experiment matrix.", experiment_root / "feature_sets"),
        _check_result("Configuration health", "Model YAML coverage", "INFO", model_expected, len(context.models), "Model definitions match the experiment matrix.", experiment_root / "models"),
        _check_result("Configuration health", "Site ID consistency", "INFO", sorted(context.matrix["sites"]), sorted(context.sites_config["sites"]), "Site IDs agree across project inputs.", project_root / "config_sites.yaml"),
        _check_result("Configuration health", "Validation enable consistency", "INFO", "project config == experiment matrix", "consistent", "Within-site, cross-site, and spatial validation enable flags agree.", experiment_root / "experiment_matrix.yaml"),
        _check_result("Configuration health", "Expected run count consistency", "INFO", context.run_counts["validation_total"], context.matrix["run_plan"]["validation_expected_runs_total"], "Recomputed validation units match the declared full total.", experiment_root / "experiment_matrix.yaml"),
        _check_result(
            "Configuration health",
            "Threshold interpretation",
            "INFO",
            f"precision target {context.project_config['analysis']['precision_target']} for evaluation; p >= {context.project_config['analysis']['probability_threshold_default']} for maps/configured representative publication comparisons",
            "distinct policies configured",
            "Evaluation and fixed reporting thresholds are documented separately.",
            project_root / "config_project.yaml",
        ),
        _check_result("Configuration health", "Deprecated compatibility total", "INFORMATION", "non-spatial compatibility value only", context.matrix["run_plan"].get("total_expected_runs"), "run_plan.total_expected_runs is retained for compatibility and is not the full validation-unit count.", experiment_root / "experiment_matrix.yaml"),
    ))

    for folder in ("prepared", "base_features", "feature_sets", "samples", "within_site", "cross_site", "spatial_block_cv", "analysis_compare", "final_maps", "report_outputs"):
        path = output_root / folder
        severity = "INFO" if path.is_dir() else "ERROR"
        checks.append(_check_result("Pipeline and validation coverage", f"{folder} output", severity, "directory present", "present" if path.is_dir() else "missing", f"Active Stage output root: {folder}/", path))

    common_valid_root = output_root / "common_valid_rasters"
    checks.append(_check_result(
        "Pipeline and validation coverage",
        "common_valid_rasters output",
        "INFO" if common_valid_root.is_dir() else "INFORMATION",
        "present for new Stage 01 runs; optional for historical runs",
        "present" if common_valid_root.is_dir() else "historical compatibility path",
        "Stage 01 common-valid physical exports; historical prepared-plus-mask runs remain supported.",
        common_valid_root,
    ))
    for folder in ("export_asc", "figures", "tables", "habitat_maps"):
        path = output_root / folder
        checks.append(_check_result(
            "Pipeline and validation coverage",
            f"obsolete root {folder}",
            "ERROR" if path.exists() else "INFO",
            "absent",
            "present" if path.exists() else "absent",
            "Fresh lifecycle stages must not create obsolete report roots.",
            path,
        ))

    expected_paths = _expected_output_paths(context)
    for validation_type in expected_paths:
        checks.extend(
            _validation_coverage_checks(
                validation_type,
                context.coverage[validation_type],
                output_root / validation_type,
            )
        )

    total_expected = sum(context.coverage[name]["expected"] for name in ("within_site", "cross_site", "spatial_block_cv"))
    total_actual = sum(_scientific_validation_actual(context.coverage[name]) for name in ("within_site", "cross_site", "spatial_block_cv"))
    checks.append(_check_result("Pipeline and validation coverage", "Total validation units", "INFO" if total_actual == total_expected else "ERROR", total_expected, total_actual, "Complete prediction/evaluation units across all enabled validation designs.", output_root))
    total_estimators = sum(context.coverage[name]["estimators"] for name in ("within_site", "cross_site", "spatial_block_cv"))
    checks.append(_check_result("Pipeline and validation coverage", "Total estimator retention", "INFO", total_expected, total_estimators, "Historical validation estimators retained locally; reported independently from scientific completeness.", output_root))

    count_pairs = (
        ("Main Figure composites", "main_figures_expected", "main_figures_actual"),
        ("Main Figure panels", "main_panels_expected", "main_panels_actual"),
        ("Supplementary Figure composites", "supplementary_figures_expected", "supplementary_figures_actual"),
        ("Supplementary Figure panels", "supplementary_panels_expected", "supplementary_panels_actual"),
        ("Main Tables", "main_tables_expected", "main_tables_actual"),
        ("Supplementary Tables", "supplementary_tables_expected", "supplementary_tables_actual"),
        ("Source data tables", "source_data_tables_expected", "source_data_tables_actual"),
        ("Internal/QC tables", "internal_tables_expected", "internal_tables_actual"),
    )
    for label, expected_key, actual_key in count_pairs:
        expected = context.report_counts[expected_key]
        actual = context.report_counts[actual_key]
        checks.append(_check_result("Report outputs", label, "INFO" if actual == expected else "ERROR", expected, actual, "Publication-config expectation and filesystem count agree." if actual == expected else "Publication-config expectation and filesystem count differ.", report_root))

    for label, key in (
        ("Main Figure manifest rows", "main_figures"),
        ("Main panel manifest rows", "main_panels"),
        ("Supplementary Figure manifest rows", "supplementary_figures"),
        ("Supplementary panel manifest rows", "supplementary_panels"),
        ("Main Table manifest rows", "main_tables"),
        ("Supplementary Table manifest rows", "supplementary_tables"),
        ("Source Data manifest rows", "source_data_tables"),
    ):
        expected = context.report_counts[f"{key}_expected"]
        actual = context.report_counts[f"{key}_manifest"]
        checks.append(_check_result(
            "Contracts and manifests",
            label,
            "INFO" if actual == expected else "ERROR",
            expected,
            actual,
            "Publication-config expectation matches the generated manifest inventory.",
            report_root / "paper_manifest",
        ))

    expected_final_map_dirs = _expected_final_map_directories(
        output_root, context.analysis_config, context.matrix
    )
    actual_final_map_sets = sum(path.is_dir() for path in expected_final_map_dirs)
    expected_final_map_sets = len(expected_final_map_dirs)
    checks.append(_check_result(
        "Pipeline and validation coverage",
        "Configured final-map sets",
        "INFO" if actual_final_map_sets == expected_final_map_sets else "ERROR",
        expected_final_map_sets,
        actual_final_map_sets,
        "Configured analysis candidate-site final-map directories are present.",
        output_root / "final_maps",
    ))
    expected_map_files = expected_final_map_sets * 2
    expected_ascii_files = expected_final_map_sets * 2 * 2
    for label, key, expected in (
        ("Map GeoTIFFs", "map_tif", expected_map_files),
        ("Map PNGs", "map_png", expected_map_files),
        ("Map PDFs", "map_pdf", expected_map_files),
        ("ASC outputs", "asc", expected_ascii_files),
        ("TXT outputs", "txt", expected_ascii_files),
    ):
        actual = context.report_counts[key]
        checks.append(_check_result("Report outputs", label, "INFO" if actual == expected else "ERROR", expected, actual, "Count derived from active final-map sets and report export policy.", report_root))

    for kind, manifest_name in (
        ("Figure", "figure_contract_validation.csv"),
        ("Table", "table_contract_validation.csv"),
        ("Source-data", "source_data_contract_validation.csv"),
    ):
        rows = context.manifests.get(manifest_name, [])
        failures = sum(row.get("status", "").lower() != "ok" for row in rows)
        checks.append(_check_result("Contracts and manifests", f"{kind} contract failures", "INFO" if failures == 0 else "ERROR", 0, failures, f"Validated {len(rows)} artifact contract rows.", report_root / "paper_manifest" / manifest_name))

    official_manifests = ("main_figures.csv", "supplementary_figures.csv", "main_tables.csv", "supplementary_tables.csv")
    missing_paths = 0
    missing_sidecars = 0
    zero_byte = 0
    duplicate_ids = 0
    for name in official_manifests:
        rows = context.manifests.get(name, [])
        identifier_column = "figure_number" if "figures" in name else "table_number"
        identifiers = [row.get(identifier_column, "") for row in rows]
        duplicate_ids += len(identifiers) - len(set(identifiers))
        for row in rows:
            for column in (
                "png_path",
                "svg_path",
                "pdf_path",
                "source_path",
                "caption_path",
                "metadata_path",
                "csv_path",
                "dictionary_path",
            ):
                value = row.get(column, "").strip()
                if value.upper() == "NA":
                    continue
                if not value:
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = report_root / path
                if not path.exists():
                    missing_paths += 1
                    if column in {"source_path", "caption_path", "metadata_path", "dictionary_path"}:
                        missing_sidecars += 1
                elif path.is_file() and path.stat().st_size == 0:
                    zero_byte += 1
    checks.extend((
        _check_result("Contracts and manifests", "Missing manifest paths", "INFO" if missing_paths == 0 else "ERROR", 0, missing_paths, "Required official artifact paths referenced by manifests.", report_root / "paper_manifest"),
        _check_result("Contracts and manifests", "Stale manifest paths", "INFO" if missing_paths == 0 else "ERROR", 0, missing_paths, "No official manifest path points to a missing artifact.", report_root / "paper_manifest"),
        _check_result("Contracts and manifests", "Missing required sidecars", "INFO" if missing_sidecars == 0 else "ERROR", 0, missing_sidecars, "Source, caption, and metadata sidecars referenced by official manifests.", report_root / "paper_manifest"),
        _check_result("Contracts and manifests", "Zero-byte official artifacts", "INFO" if zero_byte == 0 else "ERROR", 0, zero_byte, "Official artifacts must be non-empty.", report_root),
        _check_result("Contracts and manifests", "Duplicate official identifiers", "INFO" if duplicate_ids == 0 else "ERROR", 0, duplicate_ids, "Identifiers are unique within each official manifest.", report_root / "paper_manifest"),
    ))
    checks.extend(_figure_publication_qa(context))
    checks.extend(_frozen_authority_checks(context))

    canonical_docs = [docs_root / filename for filename in PROJECT_DOCUMENTS.values()]
    for path in canonical_docs:
        severity = "INFO"
        message = "Document exists and marker structure is valid."
        if not path.is_file() or path.stat().st_size == 0:
            severity, message = "ERROR", "Required project document is missing or empty."
        else:
            try:
                _parse_markers(path.read_text(encoding="utf-8"), path)
            except DocumentationError as exc:
                severity, message = "ERROR", str(exc)
        checks.append(_check_result("Documentation health", path.name, severity, "present with valid markers", "valid" if severity == "INFO" else "invalid", message, path))

    health_path = docs_root / HEALTH_DOCUMENT
    checks.append(_check_result("Documentation health", HEALTH_DOCUMENT, "INFO" if health_path.is_file() and health_path.stat().st_size > 0 else "ERROR", "present", "present" if health_path.is_file() and health_path.stat().st_size > 0 else "missing", "Fully generated Stage 99 health report." if health_path.exists() else "Health report has not been initialized.", health_path))

    broken_links = []
    for path in canonical_docs:
        if path.is_file() and path.stat().st_size > 0:
            broken_links.extend(_check_links(path, path.read_text(encoding="utf-8")))
    checks.append(_check_result("Documentation health", "Broken Markdown links", "INFO" if not broken_links else "ERROR", 0, len(broken_links), "Required project-document links resolve.", docs_root))

    for spec in ALL_OUTPUT_README_SPECS:
        path = output_root / spec.folder / spec.filename
        marker_managed = path.is_file() and "<!-- AUTO:folder-guide:start -->" in path.read_text(encoding="utf-8")
        severity = "INFO" if marker_managed else "DEFERRED"
        checks.append(_check_result("Documentation health", f"{spec.folder} README markers", severity, "marker-managed", "marker-managed" if marker_managed else "not converted", "Operational README exists but marker conversion is optional." if not marker_managed else "Operational README marker is present.", path))

    report_readmes = (
        report_root / "README_report_outputs.md",
        report_root / "figures" / "README_output_structure.md",
        report_root / "tables" / "README_table_structure.md",
    )
    report_readme_missing = sum(not path.is_file() or path.stat().st_size == 0 for path in report_readmes)
    checks.append(_check_result("Documentation health", "Report README consistency", "INFO" if report_readme_missing == 0 else "ERROR", 0, report_readme_missing, "Stage 13-owned report guides are present and non-empty.", report_root))

    git_scope = _git_scope_status(context)
    if context.git.get("git_dirty"):
        severity = "INFORMATION" if git_scope["stage99_scope_only"] else "WARNING"
        checks.append(_check_result("Project identity", "Repository working tree", severity, "clean", "dirty", "Uncommitted changes are confined to the authorized report/documentation/reproducibility-maintenance scope." if severity == "INFORMATION" else "Uncommitted changes are recorded in provenance and should be reviewed before the final commit.", context.framework_root))
    else:
        checks.append(_check_result("Project identity", "Repository working tree", "INFO", "clean", "clean", "Git working tree is clean.", context.framework_root))
    checks.extend((
        _check_result("Project identity", "Scientific/configuration sources", "INFO" if not git_scope["scientific_changed"] else "WARNING", "clean", "clean" if not git_scope["scientific_changed"] else ", ".join(git_scope["scientific_changed"]), "Tracked YAML and Stage 01-12 scientific sources are unchanged." if not git_scope["scientific_changed"] else "Tracked scientific/configuration sources contain changes.", context.framework_root),
        _check_result("Project identity", "Managed documentation changes only", "INFORMATION", "informational", "yes" if git_scope["managed_docs_only"] else "no", "No: the Stage 99 generator implementation itself is also modified." if not git_scope["managed_docs_only"] else "All tracked changes are managed documentation files.", context.paths["docs_root"]),
        _check_result("Project identity", "Report/reproducibility maintenance scope only", "INFO" if git_scope["stage99_scope_only"] else "WARNING", "yes", "yes" if git_scope["stage99_scope_only"] else "no", "All changes are confined to Stage 13/99, documentation, reproducibility infrastructure, and ignore rules.", context.framework_root),
    ))
    checks.append(_check_result("Project identity", "Exact Git tag", "INFORMATION", "optional", context.git.get("git_tag") or "none", "An absent exact tag is informational and does not reduce project health.", context.framework_root / ".git"))
    grids_path = report_root / "grids"
    checks.append(_check_result("Report outputs", "Optional grids", "INFO" if grids_path.exists() else "DEFERRED", "optional", "present" if grids_path.exists() else "absent", "XYZ grids are optional and are omitted by the default report build.", grids_path))
    changelog_path = docs_root / CHANGELOG_DOCUMENT
    changelog_ready = changelog_path.is_file() and changelog_path.stat().st_size > 0
    checks.append(_check_result(
        "Documentation health",
        "Git changelog",
        "INFO" if changelog_ready else "ERROR",
        "present",
        "present" if changelog_ready else "missing",
        f"Fully generated Git history containing up to {CHANGELOG_COMMIT_LIMIT} recent commits.",
        changelog_path,
    ))
    return checks


def _validate_configuration(context: SyncContext) -> None:
    project = _as_mapping(context.project_config.get("project"), "project")
    if not project.get("id"):
        raise DocumentationError("project.id is required.")
    if project["id"] != context.project_id:
        raise DocumentationError(f"Requested project {context.project_id} does not match project.id={project['id']}")
    matrix_id = _as_mapping(context.matrix["experiment_matrix"], "experiment_matrix").get("project_id")
    if matrix_id != context.project_id:
        raise DocumentationError(f"experiment_matrix.project_id={matrix_id!r} does not match {context.project_id!r}.")

    runtime_binding = context.runtime_binding
    if runtime_binding is None:
        expected_paths = {
            "framework_root": context.framework_root,
            "data_root": context.framework_root / "01_data",
            "project_root": context.framework_root / "02_projects" / context.project_id,
            "experiment_root": context.framework_root / "03_experiments" / context.project_id,
            "pipeline_root": context.framework_root / "04_pipeline",
            "output_root": context.framework_root / "05_outputs" / context.project_id,
            "docs_root": context.framework_root / "06_docs" / "projects" / context.project_id,
        }
    else:
        expected_paths = {
            "framework_root": runtime_binding.framework_code_root,
            "data_root": runtime_binding.data_root,
            "project_root": runtime_binding.config_root / "02_projects" / context.project_id,
            "experiment_root": runtime_binding.config_root / "03_experiments" / context.project_id,
            "pipeline_root": runtime_binding.framework_code_root / "04_pipeline",
            "output_root": runtime_binding.output_root,
            "docs_root": runtime_binding.docs_root,
        }
    for key, expected in expected_paths.items():
        actual = context.paths[key]
        if actual != expected.resolve():
            raise DocumentationError(f"paths.{key} resolves to {actual}, expected {expected.resolve()}.")
        if not actual.exists():
            if key in {"output_root", "docs_root"} and actual.parent.exists():
                context.warnings.append(f"Configured {key} does not exist yet but can be created: {actual}")
            else:
                raise DocumentationError(f"Required configured path does not exist: {actual}")

    documentation = _as_mapping(context.project_config.get("documentation"), "documentation")
    for switch in DOCUMENTATION_SWITCHES.values():
        if not isinstance(documentation.get(switch), bool):
            raise DocumentationError(f"documentation.{switch} must be a bool.")
    for site_id in context.matrix["sites"]:
        resolve_site_input_paths(
            context.sites_config["sites"][site_id],
            runtime_binding=context.runtime_binding,
        )

    counts = _validation_run_counts(context.validation_plans)
    declared = _as_mapping(context.matrix.get("run_plan"), "run_plan")
    comparisons = {
        "within_site_expected_runs": counts["within_site"],
        "cross_site_expected_runs": counts["cross_site"],
        "spatial_block_cv_expected_fold_runs": counts["spatial_block_cv"],
        "non_spatial_expected_runs_total": counts["non_spatial_total"],
        "validation_expected_runs_total": counts["validation_total"],
        "total_expected_runs": counts["non_spatial_total"],
    }
    for key, expected in comparisons.items():
        if declared.get(key) != expected:
            raise DocumentationError(f"run_plan.{key}={declared.get(key)!r}, recomputed={expected}.")
    context.run_counts = counts


def _build_context(project_id: str) -> SyncContext:
    warnings: list[str] = []
    runtime_binding = runtime_path_binding_from_environment()
    configs = load_all_configs(project_id, runtime_binding=runtime_binding)
    project_config = configs["project"]
    sites_config = configs["sites"]
    matrix = configs["experiment_matrix"]
    analysis_config = configs["analysis"]
    publication_config = configs["publication"]
    feature_sets = configs["feature_sets"]
    models = configs["models"]
    paths = resolve_project_paths(project_config, runtime_binding=runtime_binding)
    framework_root = paths["framework_root"]
    validation_plans = _build_validation_plans(matrix)
    stage_clis: dict[str, list[str]] = {}
    for stage in STAGE_SPECS:
        path = paths["pipeline_root"] / stage.script
        if not path.exists():
            raise DocumentationError(f"Pipeline script not found: {path}")
        stage_clis[stage.number] = _extract_cli_flags(path)
    context = SyncContext(
        project_id=project_id,
        framework_root=framework_root,
        runtime_binding=runtime_binding,
        project_config=project_config,
        sites_config=sites_config,
        matrix=matrix,
        analysis_config=analysis_config,
        publication_config=publication_config,
        feature_sets=feature_sets,
        models=models,
        paths=paths,
        git=get_git_metadata(framework_root),
        manifests={},
        coverage={},
        run_counts={},
        stage_clis=stage_clis,
        warnings=warnings,
        checks=[],
        report_counts={},
        validation_plans=validation_plans,
        publication_contract=configured_publication_contract(publication_config),
        sync_time=datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat(),
    )
    _validate_configuration(context)
    context.manifests = _manifest_inventory(paths["output_root"], warnings)
    _validate_report_readmes(context)
    context.coverage = _coverage(context)
    if context.git.get("git_dirty") and not _git_scope_status(context)["stage99_scope_only"]:
        warnings.append("Git working tree is dirty; synchronized provenance will record dirty=true.")
    context.report_counts = _report_output_counts(context)
    context.checks = _collect_health_checks(context)
    return context


def _validate_report_readmes(context: SyncContext) -> None:
    report_root = context.paths["output_root"] / "report_outputs"
    readmes = (
        report_root / "README_report_outputs.md",
        report_root / "figures" / "README_output_structure.md",
        report_root / "tables" / "README_table_structure.md",
    )
    for path in readmes:
        if not path.exists():
            context.warnings.append(f"Stage 13 report README is missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise DocumentationError(f"Stage 13 report README is empty: {path}")
        if "report_archives" in text or "report_outputs_old" in text:
            raise DocumentationError(f"Stage 13 report README contains a deleted legacy path: {path}")


def _manifest_counts(context: SyncContext) -> dict[str, int]:
    return {
        "Main Figures": context.publication_contract["main_figures"],
        "Supplementary Figures": context.publication_contract["supplementary_figures"],
        "Main Tables": context.publication_contract["main_tables"],
        "Supplementary Tables": context.publication_contract["supplementary_tables"],
    }


def _health_summary(context: SyncContext) -> tuple[str, int, int]:
    errors = sum(check.severity == "ERROR" for check in context.checks)
    warnings = sum(check.severity == "WARNING" for check in context.checks)
    status = "FAIL" if errors else ("PASS WITH WARNINGS" if warnings else "PASS")
    return status, errors, warnings


def _runtime_health_document(
    context: SyncContext,
    *,
    attempt_id: str,
    workflow_sha256: str,
    bundle_sha256: str,
) -> dict[str, Any]:
    """Return the narrow B4 terminal-health contract from existing checks."""

    status, errors, warnings = _health_summary(context)
    error_summaries = [
        f"{check.category}: {check.check_name}: {check.message}"
        for check in context.checks
        if check.severity == "ERROR"
    ]
    warning_summaries = [
        f"{check.category}: {check.check_name}: {check.message}"
        for check in context.checks
        if check.severity == "WARNING"
    ]
    return {
        "schema_version": 1,
        "project_id": context.project_id,
        "attempt_id": attempt_id,
        "stage_id": "99",
        "workflow_sha256": workflow_sha256,
        "bundle_sha256": bundle_sha256,
        "health_status": status,
        "error_count": errors,
        "warning_count": warnings,
        "errors": error_summaries,
        "warnings": warning_summaries,
    }


def _write_runtime_health_result(
    path: Path,
    context: SyncContext,
    *,
    attempt_id: str,
    workflow_sha256: str,
    bundle_sha256: str,
) -> None:
    document = _runtime_health_document(
        context,
        attempt_id=attempt_id,
        workflow_sha256=workflow_sha256,
        bundle_sha256=bundle_sha256,
    )
    _atomic_write(path, json.dumps(document, sort_keys=True, indent=2) + "\n")


def _checks_table(context: SyncContext, category: str) -> str:
    rows = (
        (check.check_name, check.status, check.expected, check.actual, check.message)
        for check in context.checks
        if check.category == category
    )
    return _table(("Check", "Status", "Expected", "Actual", "Message"), rows)


def _dashboard_rows(
    context: SyncContext,
    names: Sequence[str],
    labels: Mapping[str, str] | None = None,
) -> list[tuple[str, str, str, str]]:
    by_name = {check.check_name: check for check in context.checks}
    display_labels = labels or {}
    rows: list[tuple[str, str, str, str]] = []
    for name in names:
        check = by_name.get(name)
        if check is not None:
            rows.append((display_labels.get(name, check.check_name), check.status, check.expected, check.actual))
    return rows


def _render_health_report(context: SyncContext, checked_at: str) -> str:
    status, errors, warnings = _health_summary(context)
    def display_source(check: CheckResult) -> str:
        return _bound_display_path(context, check.source_path)

    warning_lines = [f"- **{check.check_name}:** {check.message} (actual: `{check.actual}`; source: `{display_source(check)}`)" for check in context.checks if check.severity == "WARNING"]
    information_lines = [f"- **{check.check_name}** [{check.severity.title()}]: {check.message} (actual: `{check.actual}`; source: `{display_source(check)}`)" for check in context.checks if check.severity in {"INFORMATION", "DEFERRED"}]
    error_lines = [f"- **{check.check_name}:** {check.message} (actual: `{check.actual}`; source: `{display_source(check)}`)" for check in context.checks if check.severity == "ERROR"]
    project = context.project_config["project"]
    analysis = context.project_config["analysis"]
    git_scope = _git_scope_status(context)
    validation_rows = _dashboard_rows(context, (
        "within_site complete runs",
        "cross_site complete runs",
        "spatial_block_cv complete runs",
        "Total validation units",
    ), {
        "within_site complete runs": "Within-site",
        "cross_site complete runs": "Cross-site",
        "spatial_block_cv complete runs": "Spatial block CV",
    })
    output_rows = _dashboard_rows(context, (
        "Main Figure composites",
        "Main Figure panels",
        "Supplementary Figure composites",
        "Supplementary Figure panels",
        "Main Tables",
        "Supplementary Tables",
        "Source data tables",
        "Internal/QC tables",
        "Map GeoTIFFs",
        "ASC outputs",
        "TXT outputs",
    ))
    contract_rows = _dashboard_rows(context, (
        "Figure contract failures",
        "Table contract failures",
        "Source-data contract failures",
        "Missing manifest paths",
        "Stale manifest paths",
        "Missing required sidecars",
        "Zero-byte official artifacts",
        "Duplicate official identifiers",
        "Figure sidecar portability failures",
        "Generic Figure caption failures",
        "Figure pairing failures",
        "Publication system files",
    ))
    pipeline_rows = _dashboard_rows(context, (
        "prepared output",
        "common_valid_rasters output",
        "base_features output",
        "feature_sets output",
        "samples output",
        "analysis_compare output",
        "final_maps output",
        "report_outputs output",
    ), {
        "prepared output": "Prepared rasters",
        "common_valid_rasters output": "Common-valid physical rasters",
        "base_features output": "Base features",
        "feature_sets output": "Feature sets",
        "samples output": "Samples",
        "analysis_compare output": "Integrated analysis",
        "final_maps output": "Final maps",
        "report_outputs output": "Report outputs",
    })
    documentation_rows = _dashboard_rows(context, (
        "README.md",
        "pipeline_manual.md",
        "feature_definitions.md",
        "experiment_catalog.md",
        HEALTH_DOCUMENT,
        "Git changelog",
        "Broken Markdown links",
        "Report README consistency",
    ))
    return f"""# {context.project_id} Health Report

This file is fully generated by Stage 99.
Manual edits will be overwritten.

## Overall status

- Status: **{status}**
- Errors: `{errors}`
- Actionable warnings: `{warnings}`
- Checked at: `{checked_at}`

## Project

- Project ID: `{context.project_id}`
- Project title: {str(project['title']).strip()}
- Generator: `{SCRIPT_DISPLAY_PATH}`

No semantic, date-based, or health-derived project version is generated.
The checked source revision is the repository HEAD at health-check time. A later commit containing this generated report normally has a different hash and does not by itself require immediate resynchronization.

## Validation summary

{_table(('Validation', 'Status', 'Expected', 'Actual'), validation_rows)}

Evaluation threshold: precision-target policy (`precision_target={analysis['precision_target']}`). Binary maps/reports and configured representative publication comparisons: fixed `p >= {analysis['probability_threshold_default']}`.

## Output summary

{_table(('Output', 'Status', 'Expected', 'Actual'), output_rows)}

## Contract summary

{_table(('Contract', 'Status', 'Expected', 'Actual'), contract_rows)}

## Pipeline summary

{_table(('Stage output', 'Status', 'Expected', 'Actual'), pipeline_rows)}

## Git summary

| Field | Value |
|---|---|
| Branch | `{context.git.get('git_branch') or 'unknown'}` |
| Checked source revision | `{context.git.get('git_hash') or 'unknown'}` |
| Exact tag | `{context.git.get('git_tag') or 'none'}` |
| Working tree | `{'dirty' if context.git.get('git_dirty') else 'clean'}` |
| Scientific/configuration sources | `{'clean' if not git_scope['scientific_changed'] else 'dirty'}` |
| Checked time | `{checked_at}` |

## Documentation summary

{_table(('Document/check', 'Status', 'Expected', 'Actual'), documentation_rows)}

## Information and deferred items

{os.linesep.join(information_lines) if information_lines else 'None.'}

## Warnings

{os.linesep.join(warning_lines) if warning_lines else 'None.'}

## Errors

{os.linesep.join(error_lines) if error_lines else 'None.'}
"""


def _render_health_sync(context: SyncContext, existing: str) -> str:
    match = HEALTH_TIMESTAMP_RE.search(existing)
    stable_time = match.group(1) if match else context.sync_time
    candidate = _render_health_report(context, stable_time)
    if candidate == existing:
        return existing
    return _render_health_report(context, context.sync_time)


def _git_changelog_entries(context: SyncContext) -> list[tuple[str, str, str, str]]:
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                f"-{CHANGELOG_COMMIT_LIMIT}",
                "--date=short",
                "--pretty=format:%h%x1f%an%x1f%ad%x1f%s%x1e",
            ],
            cwd=context.framework_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise DocumentationError(f"Unable to read Git history for changelog: {exc}") from exc
    entries: list[tuple[str, str, str, str]] = []
    for record in result.stdout.split("\x1e"):
        fields = record.strip().split("\x1f")
        if len(fields) == 4:
            entries.append(tuple(fields))
    return entries


def _render_changelog(context: SyncContext) -> str:
    entries = _git_changelog_entries(context)
    return "\n".join((
        f"# {context.project_id} Git Changelog",
        "",
        "This file is fully generated by Stage 99.",
        "Manual edits will be overwritten.",
        "",
        f"Checked source revision: `{context.git.get('git_hash') or 'unknown'}`",
        "",
        f"## Recent {len(entries)} commits",
        "",
        _table(("Commit", "Author", "Date", "Message"), entries),
        "",
    ))


def _provenance_body(context: SyncContext, sync_time: str) -> str:
    return "\n".join(
        (
            f"- Git commit: `{context.git.get('git_hash') or 'NA'}`",
            f"- Git branch: `{context.git.get('git_branch') or 'NA'}`",
            f"- Exact Git tag: `{context.git.get('git_tag') or 'none'}`",
            f"- Working tree dirty: `{str(bool(context.git.get('git_dirty'))).lower()}`",
            f"- Synchronization time: `{sync_time}`",
            f"- Synchronizer: `{SCRIPT_DISPLAY_PATH}`",
        )
    )


def _configured_frequency_description(
    sites_config: Mapping[str, Any], site_order: Sequence[str]
) -> str:
    parts = []
    for site_id in site_order:
        frequencies = sites_config["sites"][site_id]["frequencies"]
        parts.append(f"{site_id}: " + "/".join(str(value) for value in frequencies) + " kHz")
    return "; ".join(parts)


def _base_feature_names(feature_sets: Mapping[str, Mapping[str, Any]], feature_order: Sequence[str]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for feature_set_id in feature_order:
        for _, feature in _flatten_features(feature_sets[feature_set_id]):
            if feature not in seen:
                seen.add(feature)
                names.append(feature)
    return tuple(names)


def _representative_validation_description(
    publication_config: Mapping[str, Any],
    report_threshold: float | None = None,
) -> str:
    labels = publication_display_labels(publication_config)
    figures = {item["id"]: item for item in configured_figure_specs(publication_config)}
    rows = []
    for run in configured_representative_runs(publication_config):
        figure = figures[run["figure_id"]]
        rows.append((
            figure["number"],
            labels.get(run["target_site"], run["target_site"]),
            labels.get(run["train_site"], run["train_site"]),
            labels.get(run["model"], run["model"]),
            labels.get(run["feature_set"], run["feature_set"]),
            run["seed"],
            ", ".join(str(value) for value in run["spatial_folds"]),
            run["spatial_aggregation_method"],
        ))
    threshold_text = (
        f" Configured representative confusion matrices use fixed `p >= {report_threshold:g}`."
        if report_threshold is not None else ""
    )
    return "\n".join((
        "Publication representative validation runs are configured in `publication.yaml`:",
        "",
        _table(
            ("Figure", "Target site", "Transfer training site", "Model", "Feature set", "Seed", "Spatial folds", "Spatial aggregation"),
            rows,
        ),
        "",
        "Within-site uses the held-out test subset; cross-site uses the complete configured target-site dataset."
        + threshold_text,
        "No averaging across models, feature sets, or seeds is added by Stage 99; no bootstrap or multi-run confidence interval is introduced.",
    ))


def _stage_purpose(context: SyncContext, stage: StageSpec) -> str:
    if stage.number != "02":
        return stage.purpose
    feature_count = len(_base_feature_names(context.feature_sets, context.matrix["feature_sets"]))
    frequencies = _configured_frequency_description(context.sites_config, context.matrix["sites"])
    return f"Generate {feature_count} aligned base-feature rasters per site from configured frequencies ({frequencies})."


def _readme_sections(context: SyncContext, path: Path, sync_time: str) -> dict[str, str]:
    project = context.project_config["project"]
    analysis = context.project_config["analysis"]
    spatial = context.project_config["spatial"]
    site_rows = []
    for site_id in context.matrix["sites"]:
        site = context.sites_config["sites"][site_id]
        environment = site.get("environment", {})
        habitat = site.get("habitat", {})
        site_rows.append(
            (
                _code(site_id),
                site["site_name"],
                site["survey_date"],
                ", ".join(f"{value} kHz" for value in site["frequencies"]),
                habitat.get("dominant_type", "NA"),
                environment.get("depth_range_m", {}).get("min", "NA"),
                environment.get("depth_range_m", {}).get("max", "NA"),
            )
        )
    config_links = [
        ("Project config", context.paths["project_root"] / "config_project.yaml"),
        ("Site config", context.paths["project_root"] / "config_sites.yaml"),
        ("Experiment matrix", context.paths["experiment_root"] / "experiment_matrix.yaml"),
        ("Analysis config", context.paths["experiment_root"] / "analysis.yaml"),
        ("Publication config", context.paths["experiment_root"] / "publication.yaml"),
        ("Pipeline", context.paths["pipeline_root"]),
        ("Outputs", context.paths["output_root"]),
        ("Project docs", context.paths["docs_root"]),
    ]
    counts = _manifest_counts(context)
    report_root = context.paths["output_root"] / "report_outputs"
    report_guides = (
        report_root / "README_report_outputs.md",
        report_root / "figures" / "README_output_structure.md",
        report_root / "tables" / "README_table_structure.md",
    )
    health_status, health_errors, health_warnings = _health_summary(context)
    total_actual = sum(
        _scientific_validation_actual(context.coverage[name])
        for name in ("within_site", "cross_site", "spatial_block_cv")
    )
    figure_contract_failures = sum(
        row.get("status", "").lower() != "ok"
        for row in context.manifests.get("figure_contract_validation.csv", [])
    )
    table_contract_failures = sum(
        row.get("status", "").lower() != "ok"
        for row in context.manifests.get("table_contract_validation.csv", [])
    )
    return {
        "project-metadata": "\n".join(
            (
                _table(
                    ("Field", "Value"),
                    (
                        ("Project ID", _code(project["id"])),
                        ("Title", project["title"]),
                        ("Short title", project.get("short_title", "NA")),
                        ("Type / status", f"{project['type']} / {project['status']}"),
                        ("Target journal", project.get("journal_target", "NA")),
                        ("Lead author", project.get("lead_author", "NA")),
                        ("Task", analysis["task_type"]),
                        ("Classes", f"positive={analysis['positive_class_name']}; negative={analysis['negative_class_name']}"),
                        ("Spatial framework", f"{spatial['crs_out']}; {spatial['target_resolution_m']} m target resolution"),
                    ),
                ),
                "",
                str(project.get("description", "")).strip(),
            )
        ),
        "site-summary": _table(("Site ID", "Site", "Survey date", "Frequencies", "Habitat", "Min depth (m)", "Max depth (m)"), site_rows),
        "configuration-links": "\n".join(
            f"- {_link(label, path, target)}: `{_bound_display_path(context, target)}`"
            for label, target in config_links
        ),
        "pipeline-summary": _table(
            ("Stage", "Script", "Purpose"),
            ((stage.number, _link(stage.script, path, context.paths["pipeline_root"] / stage.script), _stage_purpose(context, stage)) for stage in STAGE_SPECS),
        ),
        "output-summary": "\n".join(
            (
                _table(("Official artifact group", "Manifest rows"), counts.items()),
                "",
                f"Official report root: `{_bound_display_path(context, context.paths['output_root'] / 'report_outputs')}`",
                "",
                "Stage 13 structure guides: " + ", ".join(_link(guide.name, path, guide) for guide in report_guides),
                "",
                f"Expected validation units are {context.run_counts['within_site']:,} within-site, {context.run_counts['cross_site']:,} cross-site, and {context.run_counts['spatial_block_cv']:,} spatial fold-runs ({context.run_counts['validation_total']:,} total). The deprecated {context.matrix['run_plan']['total_expected_runs']:,} total is non-spatial only.",
            )
        ),
        "health-snapshot": "\n".join(
            (
                "This is an automated technical-completeness snapshot, not an assessment of scientific quality or submission readiness.",
                "",
                f"- Overall status: **{health_status}** ({health_errors} errors; {health_warnings} actionable {'warning' if health_warnings == 1 else 'warnings'})",
                f"- Validation coverage: `{total_actual} / {context.run_counts['validation_total']}`",
                f"- Main/Supplementary figures: `{counts['Main Figures']} / {counts['Supplementary Figures']}`",
                f"- Main/Supplementary tables: `{counts['Main Tables']} / {counts['Supplementary Tables']}`",
                f"- Figure contract failures: `{figure_contract_failures}`",
                f"- Table contract failures: `{table_contract_failures}`",
                f"- Full report: {_link(HEALTH_DOCUMENT, path, context.paths['docs_root'] / HEALTH_DOCUMENT)}",
            )
        ),
        "document-index": "\n".join(
            f"- {_link(filename, path, context.paths['docs_root'] / filename)}"
            for filename in (*PROJECT_DOCUMENTS.values(), HEALTH_DOCUMENT, CHANGELOG_DOCUMENT)
        ),
        "provenance": _provenance_body(context, sync_time),
    }


def _pipeline_sections(context: SyncContext, path: Path, sync_time: str) -> dict[str, str]:
    stage_rows = (
        (
            stage.number,
            _link(stage.script, path, context.paths["pipeline_root"] / stage.script),
            _stage_purpose(context, stage),
            stage.inputs,
            stage.outputs,
            stage.downstream,
            stage.cost,
        )
        for stage in STAGE_SPECS
    )
    cli_rows = []
    for stage in STAGE_SPECS:
        flags = context.stage_clis[stage.number]
        cli_rows.append((stage.number, _code(stage.script), ", ".join(_code(flag) for flag in flags) if flags else "None"))
    return {
        "pipeline-stage-table": _table(("Stage", "Script", "Purpose", "Input", "Output", "Downstream", "Approx. cost"), stage_rows),
        "pipeline-dependency": """```mermaid
flowchart LR
    RAW[Raw site data] --> S01[Stage 01 prepared]
    S01 --> CV[Stage 01 common-valid physical rasters]
    S01 --> S02[Stage 02 base features]
    S02 --> S03[Stage 03 feature stacks]
    S03 --> S04[Stage 04 samples]
    S04 --> S05[Stage 05 within training]
    S05 --> S06[Stage 06 within evaluation]
    S04 --> S07[Stage 07 cross-site training]
    S07 --> S08[Stage 08 cross-site evaluation]
    S04 --> S09[Stage 09 spatial training]
    S09 --> S10[Stage 10 spatial evaluation]
    S06 --> S11[Stage 11 integrated analysis]
    S08 --> S11
    S10 --> S11
    S11 --> S12[Stage 12 final maps]
    S11 --> S13[Stage 13 reports]
    S12 --> S13
    S13 --> S99[Stage 99 documentation sync]
```""",
        "cli-reference": "\n".join(
            (
                "Only flags discovered from the active `argparse` calls are listed.",
                "",
                _table(("Stage", "Script", "Available long options"), cli_rows),
                "",
                "Run commands from the repository root with the project environment.",
            )
        ),
        "rebuild-guidance": _table(
            ("Deleted output", "Minimum rebuild", "Cost note"),
            (
                ("prepared/", "Stage 01 onward", "High raster cost; all downstream artifacts depend on it"),
                ("common_valid_rasters/", "Stage 01 --common-valid-only", "Physical exports can be rebuilt from prepared rasters without retraining"),
                ("base_features/", "Stage 02 onward", "High raster cost"),
                ("feature_sets/", "Stage 03 onward", "High raster I/O"),
                ("samples/", "Stage 04 onward", "All model designs must be retrained"),
                ("within/cross/spatial evaluation files", "Stages 06/08/10", "Can use stored predictions without retraining"),
                ("analysis_compare/", "Stage 11 onward", "Medium; uses saved summaries"),
                ("final_maps/", "Stage 12 onward", "Very high; retraining and raster prediction"),
                ("report_outputs/", "Stage 13", "Uses saved analyses/maps; preserves ASC/TXT policy"),
                ("Project AUTO docs", "Stage 99", "Low; MANUAL content is preserved"),
            ),
        ),
        "output-readers": _table(
            ("Output root", "Created by", "Principal readers"),
            ((spec.folder + "/", spec.created_by, spec.used_by) for spec in OUTPUT_README_SPECS),
        ),
        "provenance": _provenance_body(context, sync_time),
    }


def _flatten_features(feature_config: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for group in ("acoustic", "frequency_difference", "terrain", "texture"):
        for feature in feature_config.get("features", {}).get(group, []):
            rows.append((group, str(feature)))
    return rows


def _feature_sections(context: SyncContext, path: Path, sync_time: str) -> dict[str, str]:
    settings = context.project_config["base_features"]
    group_rows = (
        ("Acoustic intensity", "`a<frequency>_norm`", "Depth-quantile robust normalization: `(x - median) / max(p90 - p10, eps)`", settings["acoustic_normalization"]["method"]),
        ("Frequency-difference indices", "`NDI_*_norm`", "`(high - low) / (abs(high) + abs(low))`", "Stage 02 `compute_ndi`"),
        ("Bathymetric features", "`dpth_lite_*`, `dpth_rel_*`", "Robust global depth and local median/IQR relative depth", f"{settings['depth_lite']['method']}; {settings['relative_depth']['method']}"),
        ("Texture features", "`a*_low`, `a*_high`, `a*_log1p`", "Gaussian low/high and log-transformed Laplacian channels with robust scaling", settings["texture"]["method"]),
    )
    memberships: dict[str, list[str]] = {}
    groups: dict[str, str] = {}
    for feature_set_id in context.matrix["feature_sets"]:
        for group, feature in _flatten_features(context.feature_sets[feature_set_id]):
            memberships.setdefault(feature, []).append(feature_set_id)
            groups[feature] = group
    inventory_rows = (
        (_code(feature), groups[feature], ", ".join(_code(item) for item in memberships[feature]))
        for feature in memberships
    )
    set_rows = []
    for feature_set_id in context.matrix["feature_sets"]:
        config = context.feature_sets[feature_set_id]
        metadata = config["feature_set"]
        ordered = [feature for _, feature in _flatten_features(config)]
        set_rows.append(
            (
                _code(feature_set_id),
                metadata["type"],
                ", ".join(str(value) for value in metadata["frequencies"]),
                metadata["reference_frequency"],
                metadata["feature_count"],
                ", ".join(_code(value) for value in ordered),
            )
        )
    yaml_rows = []
    for feature_set_id in context.matrix["feature_sets"]:
        yaml_path = context.paths["experiment_root"] / "feature_sets" / f"{feature_set_id}.yaml"
        yaml_rows.append((_code(feature_set_id), _link(yaml_path.name, path, yaml_path)))
    return {
        "feature-groups": _table(("Feature group", "Naming pattern", "Implemented calculation", "Configured method"), group_rows),
        "feature-inventory": "\n".join((f"The {len(context.matrix['feature_sets'])} configured feature-set definitions reference {len(memberships)} unique base features.", "", _table(("Feature", "YAML group", "Used by feature sets"), inventory_rows))),
        "feature-set-matrix": _table(("Feature set", "Type", "Frequencies (kHz)", "Reference (kHz)", "Count", "Ordered bands"), set_rows),
        "feature-yaml-index": _table(("Feature set", "Definition"), yaml_rows),
        "provenance": _provenance_body(context, sync_time),
    }


def _experiment_sections(context: SyncContext, path: Path, sync_time: str) -> dict[str, str]:
    matrix = context.matrix
    model_rows = []
    for model_id in matrix["models"]:
        model = context.models[model_id]["model"]
        model_rows.append((_code(model_id), model["display_name"], model["type"], model["task"]))
    validation = matrix["validation"]
    validation_rows = (
        ("Within-site", validation["within_site"]["enable"], validation["within_site"]["split_method"], "Held-out stratified subset within each site"),
        ("Cross-site", validation["cross_site"]["enable"], "ordered site pairs", "; ".join(f"{pair['train']} -> {pair['test']}" for pair in validation["cross_site"]["pairs"])),
        ("Spatial block CV", validation["spatial_block_cv"]["enable"], validation["spatial_block_cv"]["split_method"], f"{validation['spatial_block_cv']['n_splits']} folds; {validation['spatial_block_cv']['block_size_m']} m blocks"),
    )
    count_rows = []
    labels = {
        "within_site": "Within-site seed runs",
        "cross_site": "Cross-site seed runs",
        "spatial_block_cv": "Spatial fold-runs",
    }
    for key in ("within_site", "cross_site", "spatial_block_cv"):
        coverage = context.coverage[key]
        count_rows.append((labels[key], context.run_counts[key], coverage["predictions"], coverage["evaluations"]))
    count_rows.extend(
        (
            ("Non-spatial total", context.run_counts["non_spatial_total"], "NA", "NA"),
            ("All validation units", context.run_counts["validation_total"], "NA", "NA"),
            ("Deprecated `total_expected_runs`", matrix["run_plan"]["total_expected_runs"], "NA", "Within + cross only"),
        )
    )
    return {
        "experiment-matrix": "\n".join(
            (
                _table(
                    ("Dimension", "Configured values"),
                    (
                        ("Sites", ", ".join(_code(value) for value in matrix["sites"])),
                        ("Feature sets", ", ".join(_code(value) for value in matrix["feature_sets"])),
                        ("Models", ", ".join(_code(value) for value in matrix["models"])),
                        ("Seeds", ", ".join(_code(value) for value in matrix["seeds"])),
                    ),
                ),
                "",
                _table(("Model ID", "Display name", "Type", "Task"), model_rows),
            )
        ),
        "validation-design": "\n".join(
            (
                _table(("Validation", "Enabled", "Method", "Interpretation"), validation_rows),
                "",
                f"Evaluation metrics use the configured precision-target policy (`precision_target={float(context.project_config['analysis']['precision_target']):.2f}`). Binary maps/reports and configured representative publication confusion matrices use fixed `p >= {context.project_config['analysis']['probability_threshold_default']}`; these thresholds are not interchangeable.",
            )
        ),
        "run-counts": "\n".join((
            _table(("Unit", "Expected", "Prediction files observed", "Evaluation files observed / note"), count_rows),
            "",
            "Formulas: within = sites x feature sets x models x seeds; cross = ordered pairs x feature sets x models x seeds; spatial = sites x spatial feature sets x spatial models x seeds x folds.",
        )),
        "directory-layout": """```text
within_site/<site>/<feature_set>/<model>/seed_<seed>/
cross_site/train_<train>_test_<test>/<feature_set>/<model>/seed_<seed>/
spatial_block_cv/<site>/<feature_set>/<model>/seed_<seed>/fold_<fold>/
prepared/<site>/
common_valid_rasters/<site>/
base_features/<site>/
analysis_compare/
final_maps/<site>/<candidate>/
report_outputs/
```""",
        "output-coverage": "\n".join((
            "- Stage 11 reads the configured validation root summaries (" + ", ".join(source["id"] for source in configured_validation_sources(context.analysis_config)) + ") and writes integrated rankings, stability, shifts, and candidate comparisons.",
            "- Stage 12 reads samples, feature stacks, model YAML, and `stage2_final_candidate_comparison.csv` to retrain final site models and predict maps.",
            "- Stage 12 keeps its preview optional (`--no-preview`) and can refresh only that QC graphic from frozen probability rasters with `--preview-only`.",
            "- Stage 13 reads validation, Stage 11 analysis, and Stage 12 final maps to generate reports without retraining models; it does not require Stage 12 previews.",
            "- Stage 13 map Figures use grid/map north and projected-metre scale bars. Stage 12/13 graphics share an Arial, Helvetica, DejaVu Sans fallback policy and report the family actually resolved at runtime.",
            "- Report manifests currently register " + ", ".join(f"{count} {name}" for name, count in _manifest_counts(context).items()) + ".",
        )),
        "representative-validation-comparison": _representative_validation_description(
            context.publication_config,
            float(context.project_config["analysis"]["probability_threshold_default"]),
        ),
        "provenance": _provenance_body(context, sync_time),
    }


SECTION_RENDERERS: dict[str, Callable[[SyncContext, Path, str], dict[str, str]]] = {
    "README": _readme_sections,
    "pipeline_manual": _pipeline_sections,
    "feature_definitions": _feature_sections,
    "experiment_catalog": _experiment_sections,
}


def _manual_placeholder(name: str) -> str:
    return f"<!-- Add human-authored {name.replace('-', ' ')} here. Stage 99 preserves this block. -->"


def _document_skeleton(name: str, context: SyncContext, path: Path, sync_time: str) -> str:
    sections = SECTION_RENDERERS[name](context, path, sync_time)
    title = {
        "README": f"# {context.project_config['project']['short_title']}",
        "pipeline_manual": "# Pipeline manual",
        "feature_definitions": "# Feature definitions",
        "experiment_catalog": "# Experiment catalog",
    }[name]
    layouts: dict[str, list[tuple[str, str, str]]] = {
        "README": [
            ("AUTO", "project-metadata", "Project metadata"),
            ("MANUAL", "research-context", "Research context"),
            ("AUTO", "site-summary", "Study sites"),
            ("AUTO", "configuration-links", "Configuration and locations"),
            ("AUTO", "pipeline-summary", "Pipeline summary"),
            ("AUTO", "output-summary", "Official outputs"),
            ("AUTO", "health-snapshot", "Automated health snapshot"),
            ("MANUAL", "project-status-notes", "Project status notes"),
            ("AUTO", "document-index", "Document index"),
            ("AUTO", "provenance", "Synchronization provenance"),
        ],
        "pipeline_manual": [
            ("AUTO", "pipeline-stage-table", "Pipeline stages"),
            ("AUTO", "pipeline-dependency", "Dependency graph"),
            ("AUTO", "cli-reference", "CLI reference"),
            ("AUTO", "rebuild-guidance", "Rebuild guidance"),
            ("AUTO", "output-readers", "Output readers"),
            ("MANUAL", "operational-notes", "Operational notes"),
            ("MANUAL", "known-caveats", "Known caveats"),
            ("AUTO", "provenance", "Synchronization provenance"),
        ],
        "feature_definitions": [
            ("AUTO", "feature-groups", "Feature groups and calculations"),
            ("AUTO", "feature-inventory", "Base-feature inventory"),
            ("AUTO", "feature-set-matrix", "Feature-set matrix"),
            ("AUTO", "feature-yaml-index", "Feature YAML index"),
            ("HYBRID", "feature-interpretation", "Feature interpretation"),
            ("MANUAL", "scientific-feature-notes", "Scientific feature notes"),
            ("AUTO", "provenance", "Synchronization provenance"),
        ],
        "experiment_catalog": [
            ("AUTO", "experiment-matrix", "Experiment matrix"),
            ("AUTO", "validation-design", "Validation design"),
            ("AUTO", "run-counts", "Expected and observed runs"),
            ("AUTO", "directory-layout", "Experiment directory layout"),
            ("AUTO", "output-coverage", "Output coverage and downstream use"),
            ("AUTO", "representative-validation-comparison", "Representative validation comparison"),
            ("MANUAL", "experiment-rationale", "Experiment rationale"),
            ("MANUAL", "interpretation-notes", "Interpretation notes"),
            ("MANUAL", "candidate-selection-notes", "Candidate-selection notes"),
            ("AUTO", "provenance", "Synchronization provenance"),
        ],
    }
    parts = [title]
    for kind, section_name, heading in layouts[name]:
        body = sections[section_name] if kind == "AUTO" else _manual_placeholder(section_name)
        parts.extend(("", f"## {heading}", "", _marker(kind, section_name, body)))
    return "\n".join(parts).rstrip() + "\n"


def _existing_sync_time(text: str, fallback: str) -> str:
    match = TIMESTAMP_RE.search(text)
    return match.group(1) if match else fallback


def _render_sync(name: str, context: SyncContext, path: Path, existing: str) -> str:
    stable_time = _existing_sync_time(existing, context.sync_time)
    sections = SECTION_RENDERERS[name](context, path, stable_time)
    candidate = _replace_auto_sections(existing, path, sections)
    if candidate == existing:
        return existing
    sections = SECTION_RENDERERS[name](context, path, context.sync_time)
    return _replace_auto_sections(existing, path, sections)


def _output_readme_body(spec: OutputReadmeSpec) -> str:
    key_files = "\n".join(f"- {value}" for value in spec.key_files)
    return f"""# {spec.folder}

## Purpose
{spec.purpose}

## Created by
{spec.created_by}

## Used by
{spec.used_by}

## Directory structure
{spec.structure}

## Key files
{key_files}

## Interpretation notes
{spec.interpretation}

## Rebuild dependency
{spec.rebuild}

## Preservation policy
{spec.preservation}"""


def _output_readme_skeleton(spec: OutputReadmeSpec, existing: str = "") -> str:
    if existing.strip():
        body = existing.rstrip("\n")
    else:
        body = _output_readme_body(spec)
    return "\n".join(
        (
            _marker("AUTO", "folder-guide", body),
            "",
            _marker("MANUAL", "local-notes", _manual_placeholder("local-notes")),
            "",
        )
    )


def _render_output_readme_sync(spec: OutputReadmeSpec, path: Path, existing: str) -> str:
    return _replace_auto_sections(existing, path, {"folder-guide": _output_readme_body(spec)})


def _check_links(path: Path, text: str) -> list[str]:
    broken: list[str] = []
    for raw_target in LINK_RE.findall(text):
        target = raw_target.strip().strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        resolved = Path(target) if Path(target).is_absolute() else path.parent / target
        if not resolved.exists():
            broken.append(f"{path}: {raw_target}")
    return broken


def _validate_document_markers(
    name: str,
    path: Path,
    warnings: list[str],
    allow_empty: bool,
) -> None:
    if not path.exists() or path.stat().st_size == 0:
        message = f"Project document is empty and requires --initialize: {path}"
        if allow_empty:
            warnings.append(message)
            return
        raise DocumentationError(message)
    text = path.read_text(encoding="utf-8")
    blocks = _parse_markers(text, path)
    auto_names = {marker_name for kind, marker_name in blocks if kind == "AUTO"}
    expected = set(EXPECTED_AUTO_SECTIONS[name])
    if auto_names != expected:
        raise DocumentationError(
            f"AUTO marker set mismatch in {path}: missing={sorted(expected-auto_names)}, extra={sorted(auto_names-expected)}"
        )


def _selected_documents(values: Sequence[str] | None) -> list[str]:
    if not values:
        return [*PROJECT_DOCUMENTS, "health_report", "changelog"]
    aliases: dict[str, str] = {}
    for key, filename in PROJECT_DOCUMENTS.items():
        aliases[key.lower()] = key
        aliases[filename.lower()] = key
        aliases[Path(filename).stem.lower()] = key
    aliases["health_report"] = "health_report"
    aliases[HEALTH_DOCUMENT.lower()] = "health_report"
    aliases["changelog"] = "changelog"
    aliases[CHANGELOG_DOCUMENT.lower()] = "changelog"
    selected: list[str] = []
    for raw in values:
        key = aliases.get(raw.strip().lower())
        if key is None:
            raise DocumentationError(
                f"Unknown --only target {raw!r}. Use one of: {', '.join((*PROJECT_DOCUMENTS, 'health_report', 'changelog'))}"
            )
        if key not in selected:
            selected.append(key)
    return selected


def _process_health_report(
    context: SyncContext,
    selected: Sequence[str],
    action: str,
    dry_run: bool,
) -> int:
    if "health_report" not in selected:
        return 0
    path = context.paths["docs_root"] / HEALTH_DOCUMENT
    before = path.read_text(encoding="utf-8") if path.exists() else ""
    if action == "check":
        return 0
    if action == "initialize" and before:
        print(f"Unchanged: {path}", flush=True)
        return 0
    after = _render_health_sync(context, before)
    return int(_apply_change(path, before, after, dry_run))


def _process_changelog(
    context: SyncContext,
    selected: Sequence[str],
    action: str,
    dry_run: bool,
) -> int:
    if "changelog" not in selected:
        return 0
    path = context.paths["docs_root"] / CHANGELOG_DOCUMENT
    before = path.read_text(encoding="utf-8") if path.exists() else ""
    if action == "check":
        return 0
    if action == "initialize" and before:
        print(f"Unchanged: {path}", flush=True)
        return 0
    after = _render_changelog(context)
    return int(_apply_change(path, before, after, dry_run))


def _apply_change(path: Path, before: str, after: str, dry_run: bool) -> bool:
    if before == after:
        print(f"Unchanged: {path}", flush=True)
        return False
    if dry_run:
        print(_unified_diff(path, before, after), end="", flush=True)
        return True
    changed = _atomic_write(path, after)
    print(f"Synchronized: {path}", flush=True)
    return changed


def _process_project_documents(
    context: SyncContext,
    selected: Sequence[str],
    action: str,
    dry_run: bool,
) -> tuple[int, list[str]]:
    plans: list[tuple[Path, str, str]] = []
    broken_links: list[str] = []
    documentation = context.project_config["documentation"]
    for name in selected:
        if name in {"health_report", "changelog"}:
            continue
        if not documentation.get(DOCUMENTATION_SWITCHES[name], False):
            context.warnings.append(f"Documentation target is disabled and was skipped: {name}")
            continue
        path = context.paths["docs_root"] / PROJECT_DOCUMENTS[name]
        before = path.read_text(encoding="utf-8") if path.exists() else ""
        if action == "check":
            _validate_document_markers(name, path, context.warnings, allow_empty=True)
            if before:
                broken_links.extend(_check_links(path, before))
            continue
        if action == "initialize":
            if not before:
                after = _document_skeleton(name, context, path, context.sync_time)
            else:
                _validate_document_markers(name, path, context.warnings, allow_empty=False)
                after = before
        else:
            _validate_document_markers(name, path, context.warnings, allow_empty=False)
            after = _render_sync(name, context, path, before)
        # New skeletons may cross-link siblings created by this transaction.
        # Existing documents must still be checked during initialize.
        newly_created_skeleton = action == "initialize" and not before
        if not newly_created_skeleton:
            broken_links.extend(_check_links(path, after))
        plans.append((path, before, after))
    if broken_links:
        return 0, broken_links
    changed = sum(
        int(_apply_change(path, before, after, dry_run))
        for path, before, after in plans
    )
    return changed, broken_links


def _process_output_readmes(
    context: SyncContext,
    action: str,
    dry_run: bool,
) -> tuple[int, list[str]]:
    plans: list[tuple[Path, str, str]] = []
    broken_links: list[str] = []
    for spec in ALL_OUTPUT_README_SPECS:
        folder = context.paths["output_root"] / spec.folder
        path = folder / spec.filename
        if not folder.exists():
            context.warnings.append(f"Output folder does not exist; README skipped: {folder}")
            continue
        before = path.read_text(encoding="utf-8") if path.exists() else ""
        if action == "check":
            if not before:
                context.warnings.append(f"Output README is missing or empty: {path}")
                continue
            blocks = _parse_markers(before, path)
            if ("AUTO", "folder-guide") not in blocks:
                context.warnings.append(
                    f"Output README is not marker-managed; use --initialize --sync-output-readmes to convert it safely: {path}"
                )
            broken_links.extend(_check_links(path, before))
            continue
        if action == "initialize":
            blocks = _parse_markers(before, path) if before else {}
            after = before if ("AUTO", "folder-guide") in blocks else _output_readme_skeleton(spec, before)
        else:
            if not before:
                context.warnings.append(f"Output README requires initialization and was skipped: {path}")
                continue
            blocks = _parse_markers(before, path)
            if ("AUTO", "folder-guide") not in blocks:
                context.warnings.append(f"Output README lacks markers and was not changed: {path}")
                continue
            after = _render_output_readme_sync(spec, path, before)
        broken_links.extend(_check_links(path, after))
        plans.append((path, before, after))
    if broken_links:
        return 0, broken_links
    changed = sum(
        int(_apply_change(path, before, after, dry_run))
        for path, before, after in plans
    )
    return changed, broken_links


def _hash_protected_sections(path: Path) -> dict[str, str]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    blocks = _parse_markers(text, path)
    hashes: dict[str, str] = {}
    for (kind, name), block in blocks.items():
        if kind in {"MANUAL", "HYBRID"}:
            body = "".join(lines[block.start_line + 1 : block.end_line]).encode("utf-8")
            hashes[f"{kind}:{name}"] = hashlib.sha256(body).hexdigest()
    return hashes


def _print_summary(context: SyncContext) -> None:
    status, errors, warnings = _health_summary(context)
    checks = {check.check_name: check for check in context.checks}

    def value(name: str) -> str:
        check = checks[name]
        return f"{check.actual}/{check.expected}"

    print("=" * 60, flush=True)
    print("Project", flush=True)
    print(f"  {context.project_id}", flush=True)
    print("Health", flush=True)
    print(f"  {status} | errors={errors} | warnings={warnings}", flush=True)
    print("Validation", flush=True)
    print(f"  Within {value('within_site complete runs')} | Cross {value('cross_site complete runs')}", flush=True)
    print(f"  Spatial {value('spatial_block_cv complete runs')} | Total {value('Total validation units')}", flush=True)
    print("Local estimator retention", flush=True)
    print(f"  Within {value('within_site estimator retention')} | Cross {value('cross_site estimator retention')}", flush=True)
    print(f"  Spatial {value('spatial_block_cv estimator retention')} | Total {value('Total estimator retention')}", flush=True)
    print("Frozen authorities", flush=True)
    print(f"  Final estimators {value('Final estimator retention')}", flush=True)
    print(
        "  Final mapping/publication manifest errors "
        f"{checks['Final mapping freeze integrity'].actual}/"
        f"{checks['Publication output manifest integrity'].actual}",
        flush=True,
    )
    print("Outputs", flush=True)
    print(f"  Figures main/supp {value('Main Figure composites')} / {value('Supplementary Figure composites')}", flush=True)
    print(f"  Tables main/supp {value('Main Tables')} / {value('Supplementary Tables')}", flush=True)
    print(f"  ASCII ASC/TXT {checks['ASC outputs'].actual}/{checks['TXT outputs'].actual}", flush=True)
    print("Contracts", flush=True)
    print(f"  Figure/Table failures {checks['Figure contract failures'].actual}/{checks['Table contract failures'].actual}", flush=True)
    print(f"  Missing manifest paths {checks['Missing manifest paths'].actual}", flush=True)
    print(
        "  Figure portability/caption/pairing "
        f"{checks['Figure sidecar portability failures'].actual}/"
        f"{checks['Generic Figure caption failures'].actual}/"
        f"{checks['Figure pairing failures'].actual}",
        flush=True,
    )
    print("Git", flush=True)
    print(f"  Branch {context.git.get('git_branch') or 'unknown'} | Commit {context.git.get('git_short_hash') or 'unknown'}", flush=True)
    print(f"  Tag {context.git.get('git_tag') or 'none'} | Tree {'dirty' if context.git.get('git_dirty') else 'clean'}", flush=True)
    document_names = ("README.md", "pipeline_manual.md", "feature_definitions.md", "experiment_catalog.md", HEALTH_DOCUMENT, "Git changelog")
    ready = sum(checks[name].status == "PASS" for name in document_names)
    print("Documentation", flush=True)
    print(f"  {ready}/{len(document_names)} required project documents ready", flush=True)
    for check in context.checks:
        if check.severity == "WARNING":
            print(f"  Warning: {check.check_name}: {check.message}", flush=True)
        elif check.severity == "ERROR":
            print(f"  Error: {check.check_name}: {check.message}", flush=True)
    print("=" * 60, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize marked project documentation from validated configuration and outputs."
    )
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID, help="Project identifier.")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--initialize", action="store_true", help="Initialize empty project documents with safe markers.")
    actions.add_argument("--check", action="store_true", help="Validate sources, markers, links, and consistency without writing.")
    actions.add_argument("--sync", action="store_true", help="Synchronize existing AUTO sections.")
    actions.add_argument("--summary", action="store_true", help="Print a concise read-only project health summary.")
    parser.add_argument("--dry-run", action="store_true", help="Print unified diffs without writing. Defaults to sync planning when no action is given.")
    parser.add_argument("--only", action="append", default=None, help="Restrict to one document; repeat for multiple documents.")
    parser.add_argument("--sync-output-readmes", action="store_true", help="Check or synchronize marker-managed active output READMEs.")
    parser.add_argument("--fail-on-warning", action="store_true", help="Return an error when validation emits any warning.")
    parser.add_argument(
        "--runtime-health-result",
        type=Path,
        help="B4-only path for structured terminal-health JSON.",
    )
    parser.add_argument("--runtime-attempt-id", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-workflow-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-bundle-sha256", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runtime_values = (
        args.runtime_health_result,
        args.runtime_attempt_id,
        args.runtime_workflow_sha256,
        args.runtime_bundle_sha256,
    )
    if any(value is not None for value in runtime_values) and not all(
        value is not None for value in runtime_values
    ):
        raise DocumentationError(
            "B4 runtime health output requires result path, attempt, workflow, and bundle identities."
        )
    if args.initialize:
        action = "initialize"
    elif args.check:
        action = "check"
    elif args.sync:
        action = "sync"
    elif args.summary:
        action = "summary"
    elif args.dry_run:
        action = "sync"
    else:
        raise DocumentationError("Choose --initialize, --check, --sync, --summary, or --dry-run.")

    context = _build_context(args.project_id)
    if action == "summary":
        _print_summary(context)
        return 0
    selected = _selected_documents(args.only)
    selected_project_documents = [name for name in selected if name in PROJECT_DOCUMENTS]
    protected_before = {
        name: _hash_protected_sections(context.paths["docs_root"] / PROJECT_DOCUMENTS[name])
        for name in selected_project_documents
    }
    preflight_only = bool(args.dry_run or action == "check" or args.fail_on_warning)
    changed, broken_links = _process_project_documents(
        context,
        selected,
        action,
        preflight_only,
    )
    changed += _process_health_report(context, selected, action, preflight_only)
    changed += _process_changelog(context, selected, action, preflight_only)
    manage_output_readmes = args.sync_output_readmes or action in {"check", "sync"}
    if manage_output_readmes:
        output_changed, output_broken = _process_output_readmes(
            context,
            action,
            preflight_only,
        )
        changed += output_changed
        broken_links.extend(output_broken)

    if broken_links:
        raise DocumentationError("Broken Markdown link(s):\n- " + "\n- ".join(sorted(set(broken_links))))
    if args.runtime_health_result is not None and action == "initialize":
        # A B4 attempt owns a new docs namespace. Initialize it once, then
        # reuse the ordinary sync path so the terminal snapshot reflects the
        # newly materialized documents rather than their pre-initialize state.
        context = _build_context(args.project_id)
        sync_changed, sync_broken = _process_project_documents(
            context,
            selected,
            "sync",
            False,
        )
        changed += sync_changed
        changed += _process_health_report(context, selected, "sync", False)
        changed += _process_changelog(context, selected, "sync", False)
        if manage_output_readmes:
            output_changed, output_broken = _process_output_readmes(
                context,
                "sync",
                False,
            )
            changed += output_changed
            sync_broken.extend(output_broken)
        if sync_broken:
            raise DocumentationError(
                "Broken Markdown link(s):\n- " + "\n- ".join(sorted(set(sync_broken)))
            )
    for warning in context.warnings:
        print(f"WARNING: {warning}", flush=True)
    health_status, health_errors, health_warnings = _health_summary(context)
    print(
        f"Health summary: status={health_status}, errors={health_errors}, warnings={health_warnings}",
        flush=True,
    )
    if health_warnings:
        for check in context.checks:
            if check.severity == "WARNING":
                print(
                    f"Health warning: {check.check_name}: {check.message}",
                    flush=True,
                )
    if args.fail_on_warning and (context.warnings or health_warnings):
        raise DocumentationError(
            f"Validation produced {len(context.warnings) + health_warnings} warning(s)."
        )

    if args.fail_on_warning and action != "check" and not args.dry_run:
        changed, broken_links = _process_project_documents(
            context,
            selected,
            action,
            False,
        )
        changed += _process_health_report(context, selected, action, False)
        changed += _process_changelog(context, selected, action, False)
        if manage_output_readmes:
            output_changed, output_broken = _process_output_readmes(
                context,
                action,
                False,
            )
            changed += output_changed
            broken_links.extend(output_broken)
        if broken_links:
            raise DocumentationError("Broken Markdown link(s):\n- " + "\n- ".join(sorted(set(broken_links))))

    if not args.dry_run and action != "check":
        for name in selected_project_documents:
            path = context.paths["docs_root"] / PROJECT_DOCUMENTS[name]
            if protected_before[name] and protected_before[name] != _hash_protected_sections(path):
                raise DocumentationError(f"Protected MANUAL/HYBRID content changed unexpectedly: {path}")

    warning_count = len(context.warnings)
    print(
        f"Documentation {action} complete: selected={len(selected)}, changed={changed}, "
        f"broken_links=0, warnings={warning_count}, dry_run={bool(args.dry_run or action == 'check')}",
        flush=True,
    )
    if args.runtime_health_result is not None:
        _write_runtime_health_result(
            args.runtime_health_result,
            context,
            attempt_id=args.runtime_attempt_id,
            workflow_sha256=args.runtime_workflow_sha256,
            bundle_sha256=args.runtime_bundle_sha256,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DocumentationError, FileNotFoundError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(1) from exc

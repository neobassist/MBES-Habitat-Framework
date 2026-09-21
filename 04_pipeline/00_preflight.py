#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Check project readiness before Stage 01 without changing scientific data."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.preflight import (
    CheckResult,
    FRAMEWORK_ROOT,
    atomic_write_if_changed,
    build_manifest_rows,
    check_environment,
    check_portability,
    config_input_specs,
    json_text,
    limitation_checks,
    load_and_validate_configs,
    overall_status,
    project_paths,
    render_manifest,
    render_text_summary,
    summary_dict,
    validate_rasters,
    validate_references,
    validate_frequency_design,
    validate_site_inputs,
    validate_vectors,
    verify_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only MBES project readiness and reproducibility preflight.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--project-config", type=Path, help="Explicit project YAML for isolated tests.")
    parser.add_argument("--sites-config", type=Path, help="Explicit site YAML for isolated tests.")
    parser.add_argument("--experiment-matrix", type=Path, help="Explicit matrix YAML for isolated tests.")
    parser.add_argument("--analysis-config", type=Path, help="Explicit analysis YAML for isolated tests.")
    parser.add_argument("--publication-config", type=Path, help="Explicit publication YAML for isolated tests.")
    parser.add_argument("--manifest", type=Path, help="Input manifest path; defaults to the project directory.")
    parser.add_argument("--write-manifest", action="store_true", help="Atomically create or refresh the input manifest.")
    parser.add_argument("--skip-manifest-check", action="store_true", help="Skip canonical manifest comparison for temporary test fixtures.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    parser.add_argument("--output", type=Path, help="Explicitly write the rendered report; default is terminal-only.")
    return parser


def run(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    paths = project_paths(
        args.project_id,
        FRAMEWORK_ROOT,
        args.project_config,
        args.sites_config,
        args.experiment_matrix,
        args.analysis_config,
        args.publication_config,
    )
    manifest_path = args.manifest or paths["manifest"]
    configs, checks = load_and_validate_configs(paths, args.project_id)
    rows: list[dict] = []
    if len(configs) == 5:
        feature_configs, model_configs, reference_checks = validate_references(configs, paths)
        checks.extend(reference_checks)
        checks.extend(validate_frequency_design(configs, feature_configs))
        site_specs, site_checks = validate_site_inputs(configs, FRAMEWORK_ROOT)
        checks.extend(site_checks)
        config_specs = config_input_specs(args.project_id, paths, feature_configs, model_configs)
        specs = [*config_specs, *site_specs]
        raster_metadata, raster_checks = validate_rasters(specs)
        vector_metadata, vector_checks = validate_vectors(specs, configs)
        checks.extend(raster_checks)
        checks.extend(vector_checks)
        rows = build_manifest_rows(specs, raster_metadata, vector_metadata)
        if args.write_manifest:
            blocking = [check for check in checks if check.status == "FAIL"]
            if blocking:
                checks.append(
                    CheckResult(
                        category="Input manifest",
                        check_name="manifest_write",
                        status="FAIL",
                        message=(
                            "Input manifest was not written because configuration or "
                            "input validation failed."
                        ),
                        expected="no blocking validation failures",
                        actual=len(blocking),
                        source_path=str(manifest_path),
                    )
                )
            else:
                changed = atomic_write_if_changed(manifest_path, render_manifest(rows))
                checks.append(
                    CheckResult(
                        category="Input manifest",
                        check_name="manifest_write",
                        status="PASS",
                        message=(
                            f"Input manifest {'updated' if changed else 'already current'}: "
                            f"{manifest_path}"
                        ),
                        actual="changed" if changed else "unchanged",
                        source_path=str(manifest_path),
                    )
                )
        if not args.skip_manifest_check:
            checks.extend(verify_manifest(manifest_path, rows))
    checks.extend(check_environment())
    checks.extend(check_portability())
    checks.extend(limitation_checks())
    return summary_dict(args.project_id, checks), rows


def main() -> int:
    args = build_parser().parse_args()
    summary, _ = run(args)
    rendered = json_text(summary) if args.json else render_text_summary(summary)
    print(rendered, end="")
    if args.output:
        atomic_write_if_changed(args.output, rendered)
    return 1 if summary["overall"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

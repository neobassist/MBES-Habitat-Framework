#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read-only project readiness checks and input-manifest support."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from utils.config_io import (
    RuntimePathBinding,
    configured_frequencies,
    configured_cross_site_pairs,
    configured_feature_sets,
    configured_ndi_pairs,
    cross_site_expected_run_count,
    generated_base_feature_names,
    resolve_repo_path,
    runtime_path_binding_from_environment,
    spatial_expected_fold_run_count,
    validate_analysis_config,
    validate_publication_config,
    validate_experiment_matrix,
    validate_project_config,
    validate_sites_config,
)


FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_COLUMNS = [
    "project_id",
    "site_id",
    "input_role",
    "artifact_id",
    "configured_path",
    "repository_relative_path",
    "file_type",
    "required",
    "exists",
    "file_size_bytes",
    "sha256",
    "asset_class",
    "frequency_khz",
    "expected_band_count",
    "observed_band_count",
    "crs",
    "provenance",
    "notes",
]
SHAPEFILE_COMPONENTS = {
    ".shp": True,
    ".shx": True,
    ".dbf": True,
    ".prj": True,
    ".cpg": False,
}
SUPPORTED_MODEL_TYPES = {
    "sklearn_random_forest",
    "xgboost_classifier",
    "lightgbm_classifier",
    "catboost_classifier",
}
PORTABILITY_PATTERN = re.compile(
    r"/(?:Users|home)/[^/\s]+|/private/tmp(?:/|\b)|/tmp(?:/|\b)|"
    r"(?:mini|ana)conda(?:3)?(?:/|\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CheckResult:
    category: str
    check_name: str
    status: str
    message: str
    expected: Any = None
    actual: Any = None
    source_path: str = ""


@dataclass(frozen=True)
class DependencySpec:
    distribution: str
    module: str
    requirement: str
    used_by: str


DEPENDENCIES = (
    DependencySpec("numpy", "numpy", "required", "Stages 01-13"),
    DependencySpec("pandas", "pandas", "required", "Stages 04, 11-13 and I/O helpers"),
    DependencySpec("scipy", "scipy", "required", "Stages 02 and 11"),
    DependencySpec("scikit-learn", "sklearn", "required", "Training, evaluation, and Stage 13"),
    DependencySpec("rasterio", "rasterio", "required", "Raster stages 01-04 and 12-13"),
    DependencySpec("geopandas", "geopandas", "required", "Vector input in Stages 01 and 04"),
    DependencySpec("shapely", "shapely", "required", "GeoPandas geometry operations"),
    DependencySpec("pyproj", "pyproj", "required", "CRS operations through GeoPandas/rasterio"),
    DependencySpec("fiona", "fiona", "required", "Vector file driver used by GeoPandas"),
    DependencySpec("matplotlib", "matplotlib", "required", "Stages 12 and 13"),
    DependencySpec("joblib", "joblib", "required", "Model serialization"),
    DependencySpec("PyYAML", "yaml", "required", "All YAML configuration I/O"),
    DependencySpec("lightgbm", "lightgbm", "required", "Configured LightGBM runs"),
    DependencySpec("xgboost", "xgboost", "required", "Configured XGBoost runs"),
    DependencySpec("catboost", "catboost", "required", "Configured CatBoost runs"),
)


def _result(
    category: str,
    check_name: str,
    status: str,
    message: str,
    *,
    expected: Any = None,
    actual: Any = None,
    source_path: str | Path = "",
) -> CheckResult:
    return CheckResult(
        category=category,
        check_name=check_name,
        status=status,
        message=message,
        expected=expected,
        actual=actual,
        source_path=str(source_path),
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative(path: Path, framework_root: Path = FRAMEWORK_ROOT) -> str:
    try:
        return path.resolve().relative_to(framework_root.resolve()).as_posix()
    except ValueError:
        return ""


def manifest_portable_path(
    path: Path,
    framework_root: Path = FRAMEWORK_ROOT,
    runtime_binding: RuntimePathBinding | None = None,
) -> str:
    """Return one logical locator for legacy or split-root manifest evidence."""

    binding = runtime_binding or runtime_path_binding_from_environment()
    if binding is None:
        return repository_relative(path, framework_root)

    resolved = path.resolve(strict=False)
    authorities = (
        (binding.data_root, PurePosixPath(binding.data_locator)),
        (binding.config_root, PurePosixPath()),
        (binding.framework_code_root, PurePosixPath()),
    )
    for root, prefix in authorities:
        try:
            suffix = resolved.relative_to(root.resolve(strict=False))
        except ValueError:
            continue
        logical = prefix.joinpath(*suffix.parts)
        if logical.is_absolute() or ".." in logical.parts:
            raise ValueError(f"Manifest locator is not portable: {logical}")
        return logical.as_posix()
    raise ValueError(f"Manifest input has no bound runtime authority root: {path}")


def load_yaml_unique(path: Path) -> dict[str, Any]:
    """Load YAML while rejecting duplicate mapping keys before normal validation."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ImportError("PyYAML is required for preflight configuration checks.") from exc

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(
                    f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}."
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return loaded


def project_paths(
    project_id: str,
    framework_root: Path = FRAMEWORK_ROOT,
    project_config: Path | None = None,
    sites_config: Path | None = None,
    experiment_matrix: Path | None = None,
    analysis_config: Path | None = None,
    publication_config: Path | None = None,
) -> dict[str, Path]:
    runtime_binding = runtime_path_binding_from_environment()
    config_root = runtime_binding.config_root if runtime_binding is not None else framework_root
    return {
        "project": project_config
        or config_root / "02_projects" / project_id / "config_project.yaml",
        "sites": sites_config
        or config_root / "02_projects" / project_id / "config_sites.yaml",
        "matrix": experiment_matrix
        or config_root / "03_experiments" / project_id / "experiment_matrix.yaml",
        "analysis": analysis_config
        or config_root / "03_experiments" / project_id / "analysis.yaml",
        "publication": publication_config
        or config_root / "03_experiments" / project_id / "publication.yaml",
        "feature_dir": config_root / "03_experiments" / project_id / "feature_sets",
        "model_dir": config_root / "03_experiments" / project_id / "models",
        "manifest": config_root / "02_projects" / project_id / "input_manifest.csv",
    }


def load_and_validate_configs(paths: Mapping[str, Path], project_id: str) -> tuple[dict[str, Any], list[CheckResult]]:
    checks: list[CheckResult] = []
    loaded: dict[str, Any] = {}
    validators = {
        "project": validate_project_config,
        "sites": validate_sites_config,
        "matrix": validate_experiment_matrix,
    }
    for key in ("project", "sites", "matrix"):
        path = paths[key]
        if not path.is_file():
            checks.append(_result("Configuration", key, "FAIL", f"Required configuration is missing: {path}", source_path=path))
            continue
        try:
            loaded[key] = load_yaml_unique(path)
            validators[key](loaded[key])
            checks.append(_result("Configuration", key, "PASS", f"Parsed and validated {path.name}.", source_path=path))
        except Exception as exc:
            checks.append(_result("Configuration", key, "FAIL", f"Unable to validate {path.name}: {exc}", source_path=path))

    analysis_path = paths["analysis"]
    if not analysis_path.is_file():
        checks.append(
            _result(
                "Configuration",
                "analysis",
                "FAIL",
                f"Required configuration is missing: {analysis_path}",
                source_path=analysis_path,
            )
        )
    else:
        try:
            loaded["analysis"] = load_yaml_unique(analysis_path)
            validate_analysis_config(
                loaded["analysis"],
                loaded.get("matrix"),
                expected_project_id=project_id,
            )
            checks.append(
                _result(
                    "Configuration",
                    "analysis",
                    "PASS",
                    "Parsed and validated analysis.yaml.",
                    source_path=analysis_path,
                )
            )
        except Exception as exc:
            checks.append(
                _result(
                    "Configuration",
                    "analysis",
                    "FAIL",
                    f"Unable to validate analysis.yaml: {exc}",
                    source_path=analysis_path,
                )
            )

    publication_path = paths["publication"]
    if not publication_path.is_file():
        checks.append(
            _result(
                "Configuration",
                "publication",
                "FAIL",
                f"Required configuration is missing: {publication_path}",
                source_path=publication_path,
            )
        )
    else:
        try:
            loaded["publication"] = load_yaml_unique(publication_path)
            validate_publication_config(
                loaded["publication"],
                loaded.get("matrix"),
                loaded.get("analysis"),
                expected_project_id=project_id,
            )
            checks.append(
                _result(
                    "Configuration",
                    "publication",
                    "PASS",
                    "Parsed and validated publication.yaml.",
                    source_path=publication_path,
                )
            )
        except Exception as exc:
            checks.append(
                _result(
                    "Configuration",
                    "publication",
                    "FAIL",
                    f"Unable to validate publication.yaml: {exc}",
                    source_path=publication_path,
                )
            )

    if len(loaded) != 5:
        return loaded, checks

    configured_id = loaded["project"].get("project", {}).get("id")
    matrix_id = loaded["matrix"].get("experiment_matrix", {}).get("project_id")
    analysis_id = loaded["analysis"].get("project_id")
    publication_id = loaded["publication"].get("project_id")
    if configured_id == project_id == matrix_id == analysis_id == publication_id:
        checks.append(_result("Configuration", "project_id", "PASS", "Project identifiers agree across the CLI and YAML files.", expected=project_id, actual=configured_id))
    else:
        checks.append(_result("Configuration", "project_id", "FAIL", "Project identifiers do not agree.", expected=project_id, actual={"project": configured_id, "matrix": matrix_id, "analysis": analysis_id, "publication": publication_id}))
    return loaded, checks


def _duplicates(values: Sequence[Any]) -> list[Any]:
    return sorted({value for value in values if values.count(value) > 1}, key=str)


def validate_references(
    configs: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[CheckResult]]:
    checks: list[CheckResult] = []
    sites = configs["sites"]["sites"]
    matrix = configs["matrix"]
    matrix_sites = list(matrix["sites"])
    feature_ids = list(matrix["feature_sets"])
    model_ids = list(matrix["models"])
    seeds = list(matrix["seeds"])

    for label, values in (("site IDs", matrix_sites), ("feature-set IDs", feature_ids), ("model IDs", model_ids), ("seeds", seeds)):
        duplicates = _duplicates(values)
        checks.append(_result("Experiment references", f"unique_{label.replace(' ', '_')}", "FAIL" if duplicates else "PASS", f"Duplicate {label}: {duplicates}" if duplicates else f"No duplicate {label}.", actual=duplicates))

    site_diff = sorted(set(matrix_sites) ^ set(sites))
    checks.append(_result("Experiment references", "site_ids", "FAIL" if site_diff else "PASS", f"Site references differ: {site_diff}" if site_diff else "Experiment site IDs match config_sites.yaml.", expected=sorted(sites), actual=sorted(matrix_sites)))

    invalid_seeds = [seed for seed in seeds if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0]
    checks.append(_result("Experiment references", "seeds", "FAIL" if invalid_seeds else "PASS", f"Invalid seeds: {invalid_seeds}" if invalid_seeds else "Seeds are unique non-negative integers.", actual=invalid_seeds))

    feature_configs: dict[str, dict[str, Any]] = {}
    for feature_id in feature_ids:
        path = paths["feature_dir"] / f"{feature_id}.yaml"
        if not path.is_file():
            checks.append(_result("Experiment references", "feature_yaml", "FAIL", f"Feature-set YAML is missing for {feature_id!r}.", source_path=path))
            continue
        try:
            config = load_yaml_unique(path)
            feature_header = config.get("feature_set", {})
            actual_id = feature_header.get("id")
            if actual_id != feature_id:
                raise ValueError(f"expected id {feature_id!r}, found {actual_id!r}")
            groups = config.get("features")
            if not isinstance(groups, Mapping):
                raise ValueError("features must be a mapping")
            members = [item for values in groups.values() for item in values]
            if feature_header.get("feature_count") != len(members):
                raise ValueError(
                    "feature_count does not match flattened feature membership"
                )
            feature_configs[feature_id] = config
            checks.append(_result("Experiment references", "feature_yaml", "PASS", f"Feature-set {feature_id!r} is defined.", source_path=path))
        except Exception as exc:
            checks.append(_result("Experiment references", "feature_yaml", "FAIL", f"Invalid feature-set YAML for {feature_id!r}: {exc}", source_path=path))

    model_configs: dict[str, dict[str, Any]] = {}
    for model_id in model_ids:
        path = paths["model_dir"] / f"{model_id}.yaml"
        if not path.is_file():
            checks.append(_result("Experiment references", "model_yaml", "FAIL", f"Model YAML is missing for {model_id!r}.", source_path=path))
            continue
        try:
            config = load_yaml_unique(path)
            model = config.get("model", {})
            if model.get("id") != model_id:
                raise ValueError(f"expected id {model_id!r}, found {model.get('id')!r}")
            if model.get("type") not in SUPPORTED_MODEL_TYPES:
                raise ValueError(f"unsupported model type {model.get('type')!r}")
            model_configs[model_id] = config
            checks.append(_result("Experiment references", "model_yaml", "PASS", f"Model {model_id!r} is defined with a supported type.", source_path=path))
        except Exception as exc:
            checks.append(_result("Experiment references", "model_yaml", "FAIL", f"Invalid model YAML for {model_id!r}: {exc}", source_path=path))

    try:
        cross_pairs = configured_cross_site_pairs(matrix)
        checks.append(
            _result(
                "Experiment references",
                "cross_site_pairs",
                "PASS",
                "Cross-site pairs are ordered, directional, unique, and reference configured sites.",
                expected="configured directional pairs",
                actual=[{"train": train, "test": test} for train, test in cross_pairs],
                source_path=paths["matrix"],
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        cross_pairs = ()
        checks.append(
            _result(
                "Experiment references",
                "cross_site_pairs",
                "FAIL",
                f"Invalid cross-site pair configuration: {exc}",
                expected="non-empty unique directional pairs between configured sites",
                actual=str(exc),
                source_path=paths["matrix"],
            )
        )

    spatial = matrix["validation"]["spatial_block_cv"]
    n_splits = spatial.get("n_splits")
    checks.append(_result("Experiment references", "spatial_folds", "PASS" if isinstance(n_splits, int) and not isinstance(n_splits, bool) and n_splits >= 2 else "FAIL", "Spatial fold count is valid." if isinstance(n_splits, int) and not isinstance(n_splits, bool) and n_splits >= 2 else "Spatial fold count must be an integer >= 2.", expected=">=2", actual=n_splits))
    stage1 = spatial.get("stage1", {})
    unknown_stage1_models = sorted(set(stage1.get("models", [])) - set(model_ids))
    try:
        spatial_feature_sets = configured_feature_sets(
            matrix,
            available_feature_set_ids=tuple(feature_configs),
        )
        feature_error = ""
    except (KeyError, TypeError, ValueError) as exc:
        spatial_feature_sets = ()
        feature_error = str(exc)
    spatial_reference_errors = {
        "feature_sets": feature_error,
        "models": unknown_stage1_models,
    }
    has_spatial_reference_errors = bool(feature_error or unknown_stage1_models)
    checks.append(_result("Experiment references", "spatial_stage1_references", "FAIL" if has_spatial_reference_errors else "PASS", "Spatial feature sets and stage1 models reference configured definitions." if not has_spatial_reference_errors else f"Invalid spatial references: {spatial_reference_errors}", actual=spatial_reference_errors))

    unavailable_frequencies = []
    for feature_id, config in feature_configs.items():
        required_frequencies = set(config["feature_set"].get("frequencies", []))
        for site_id, site in sites.items():
            missing = sorted(required_frequencies - set(site["frequencies"]))
            if missing:
                unavailable_frequencies.append(
                    f"{feature_id}@{site_id}:{missing}"
                )
    checks.append(
        _result(
            "Experiment references",
            "feature_frequency_coverage",
            "FAIL" if unavailable_frequencies else "PASS",
            (
                f"Feature sets require unavailable site frequencies: {unavailable_frequencies}"
                if unavailable_frequencies
                else "Every feature set's declared frequencies are available at every experiment site."
            ),
            actual=unavailable_frequencies,
        )
    )

    expected_counts = {
        "within_site_expected_runs": len(matrix_sites) * len(feature_ids) * len(model_ids) * len(seeds),
        "cross_site_expected_runs": (
            cross_site_expected_run_count(matrix) if cross_pairs else 0
        ),
        "spatial_block_cv_expected_fold_runs": (
            spatial_expected_fold_run_count(matrix) if spatial_feature_sets else 0
        ),
    }
    run_plan = matrix.get("run_plan", {})
    count_mismatches = {
        key: {"expected": value, "configured": run_plan.get(key)}
        for key, value in expected_counts.items()
        if run_plan.get(key) is not None and int(run_plan[key]) != value
    }
    checks.append(
        _result(
            "Experiment references",
            "expected_run_counts",
            "FAIL" if count_mismatches else "PASS",
            (
                f"Expected run-count mismatches: {count_mismatches}"
                if count_mismatches
                else f"Configured run counts match the matrix: {expected_counts}."
            ),
            expected=expected_counts,
            actual={key: run_plan.get(key) for key in expected_counts},
        )
    )
    return feature_configs, model_configs, checks


def validate_frequency_design(
    configs: Mapping[str, Any],
    feature_configs: Mapping[str, Mapping[str, Any]],
) -> list[CheckResult]:
    """Validate Stage 02 frequency generation against sites and feature YAML."""

    checks: list[CheckResult] = []
    sites = configs["sites"]["sites"]
    base_features = configs["project"]["base_features"]
    try:
        ndi_pairs = configured_ndi_pairs(base_features)
        checks.append(
            _result(
                "Frequency design",
                "ndi_pair_schema",
                "PASS",
                f"Configured {len(ndi_pairs)} explicit NDI pair(s) with stable feature names.",
                actual=[dict(pair) for pair in ndi_pairs],
            )
        )
    except Exception as exc:
        return [
            _result(
                "Frequency design",
                "ndi_pair_schema",
                "FAIL",
                f"Invalid NDI configuration: {exc}",
                source_path="config_project.yaml:base_features.ndi_pairs",
            )
        ]

    for site_id, site in sites.items():
        try:
            frequencies = configured_frequencies(site)
            checks.append(
                _result(
                    "Frequency design",
                    f"{site_id}_frequencies",
                    "PASS",
                    f"Site {site_id} defines unique positive frequencies: {list(frequencies)}.",
                    actual=list(frequencies),
                    source_path=f"config_sites.yaml:sites.{site_id}.frequencies",
                )
            )
        except Exception as exc:
            checks.append(
                _result(
                    "Frequency design",
                    f"{site_id}_frequencies",
                    "FAIL",
                    f"Invalid site frequency configuration: {exc}",
                    source_path=f"config_sites.yaml:sites.{site_id}.frequencies",
                )
            )
            continue

        available = set(frequencies)
        unavailable_pairs = [
            dict(pair)
            for pair in ndi_pairs
            if pair["high_frequency"] not in available
            or pair["low_frequency"] not in available
        ]
        checks.append(
            _result(
                "Frequency design",
                f"{site_id}_ndi_coverage",
                "FAIL" if unavailable_pairs else "PASS",
                (
                    f"NDI pairs reference unavailable frequencies at {site_id}: {unavailable_pairs}."
                    if unavailable_pairs
                    else f"Every configured NDI pair is available at site {site_id}."
                ),
                expected="all NDI frequencies available",
                actual=unavailable_pairs,
                source_path=f"config_sites.yaml:sites.{site_id}.frequencies",
            )
        )

        generated = set(generated_base_feature_names(frequencies, ndi_pairs))
        missing_features: list[str] = []
        for feature_set_id, feature_config in feature_configs.items():
            groups = feature_config.get("features", {})
            for members in groups.values():
                for feature_name in members:
                    if feature_name not in generated:
                        missing_features.append(f"{feature_set_id}:{feature_name}")
        checks.append(
            _result(
                "Frequency design",
                f"{site_id}_feature_generation",
                "FAIL" if missing_features else "PASS",
                (
                    f"Feature YAML references unavailable Stage 02 outputs at {site_id}: {missing_features}."
                    if missing_features
                    else f"All feature YAML members can be generated for site {site_id}."
                ),
                expected="all feature members generated",
                actual=missing_features,
                source_path="feature-set YAML",
            )
        )
    return checks


def _site_input_paths(
    site: Mapping[str, Any],
    framework_root: Path = FRAMEWORK_ROOT,
) -> tuple[dict[int, Path], dict[int, Path], Path, Path]:
    frequencies = [int(value) for value in site["frequencies"]]
    paths = site["paths"]
    files = site["input_files"]
    bathy_dir = resolve_repo_path(paths["bathy_dir"], framework_root)
    sonar_dir = resolve_repo_path(paths["sonar_dir"], framework_root)
    bathy = {frequency: bathy_dir / str(files["bathymetry"].get(frequency, files["bathymetry"].get(str(frequency), ""))) for frequency in frequencies}
    backscatter = {frequency: sonar_dir / str(files["backscatter"].get(frequency, files["backscatter"].get(str(frequency), ""))) for frequency in frequencies}
    return (
        bathy,
        backscatter,
        resolve_repo_path(paths["label_path"], framework_root),
        resolve_repo_path(paths["crop_shp"], framework_root),
    )


def validate_site_inputs(
    configs: Mapping[str, Any],
    framework_root: Path = FRAMEWORK_ROOT,
) -> tuple[list[dict[str, Any]], list[CheckResult]]:
    checks: list[CheckResult] = []
    specs: list[dict[str, Any]] = []
    for site_id, site in configs["sites"]["sites"].items():
        frequencies = list(site["frequencies"])
        duplicate_frequencies = _duplicates(frequencies)
        checks.append(_result("Site inputs", "frequency_ids", "FAIL" if duplicate_frequencies else "PASS", f"Site {site_id} has duplicate frequencies: {duplicate_frequencies}" if duplicate_frequencies else f"Site {site_id} frequency definitions are unique.", source_path=site_id))
        try:
            bathy, backscatter, label_path, crop_path = _site_input_paths(
                site,
                framework_root,
            )
        except Exception as exc:
            checks.append(_result("Site inputs", "path_resolution", "FAIL", f"Unable to resolve inputs for site {site_id}: {exc}", source_path=site_id))
            continue
        for role, items in (("bathymetry", bathy), ("backscatter", backscatter)):
            for frequency, path in items.items():
                specs.append({"project_id": configs["project"]["project"]["id"], "site_id": site_id, "input_role": role, "artifact_id": f"{site_id}:{role}:{frequency}khz", "path": path, "file_type": "GeoTIFF", "required": True, "asset_class": "raster", "frequency_khz": frequency, "expected_band_count": 1, "provenance": "config_sites.yaml", "notes": "Raw Stage 01 input."})
                if path.is_file():
                    checks.append(_result("Site inputs", f"{site_id}_{role}_{frequency}", "PASS", f"Site {site_id} {frequency} kHz {role} raster exists.", source_path=path))
                else:
                    checks.append(_result("Site inputs", f"{site_id}_{role}_{frequency}", "FAIL", f"Site {site_id} requires {frequency} kHz {role} raster, but the configured file does not exist.", source_path=path))
        for role, path in (("habitat_labels", label_path), ("crop_boundary", crop_path)):
            if path.suffix.lower() == ".shp":
                for suffix, required in SHAPEFILE_COMPONENTS.items():
                    component = path.with_suffix(suffix)
                    specs.append({"project_id": configs["project"]["project"]["id"], "site_id": site_id, "input_role": role, "artifact_id": f"{site_id}:{role}:{suffix[1:]}", "path": component, "file_type": f"Shapefile {suffix}", "required": required, "asset_class": "vector", "frequency_khz": "", "expected_band_count": "", "provenance": "config_sites.yaml", "notes": "Required Shapefile component." if required else "Optional encoding component."})
                    status = "PASS" if component.is_file() or not required else "FAIL"
                    message = f"Site {site_id} {role} component {suffix} {'exists' if component.is_file() else 'is optional and absent' if not required else 'is missing'} ."
                    checks.append(_result("Site inputs", f"{site_id}_{role}_{suffix[1:]}", status, message.replace("  .", "."), source_path=component))
            else:
                specs.append({"project_id": configs["project"]["project"]["id"], "site_id": site_id, "input_role": role, "artifact_id": f"{site_id}:{role}", "path": path, "file_type": path.suffix.lower().lstrip(".").upper(), "required": True, "asset_class": "vector", "frequency_khz": "", "expected_band_count": "", "provenance": "config_sites.yaml", "notes": "Raw vector input."})
                checks.append(_result("Site inputs", f"{site_id}_{role}", "PASS" if path.is_file() else "FAIL", f"Site {site_id} {role} {'exists' if path.is_file() else 'is missing'}.", source_path=path))
    return specs, checks


def validate_rasters(specs: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[CheckResult]]:
    metadata: dict[str, dict[str, Any]] = {}
    checks: list[CheckResult] = []
    try:
        import rasterio
    except Exception as exc:
        return metadata, [_result("Raster metadata", "raster_reader", "FAIL", f"Raster metadata cannot be inspected because rasterio is unavailable: {exc}")]
    for spec in specs:
        if spec["asset_class"] != "raster" or not Path(spec["path"]).is_file():
            continue
        path = Path(spec["path"])
        try:
            with rasterio.open(path) as source:
                info = {"observed_band_count": source.count, "crs": str(source.crs) if source.crs else "", "width": source.width, "height": source.height, "nodata": source.nodata}
            metadata[spec["artifact_id"]] = info
            problems = []
            if info["observed_band_count"] != int(spec["expected_band_count"]): problems.append(f"band count={info['observed_band_count']}")
            if not info["crs"]: problems.append("CRS is missing")
            if info["width"] <= 0 or info["height"] <= 0: problems.append(f"invalid dimensions={info['width']}x{info['height']}")
            checks.append(_result("Raster metadata", spec["artifact_id"], "FAIL" if problems else "PASS", f"Raster validation failed: {', '.join(problems)}" if problems else f"Readable single-band raster; CRS={info['crs']}, size={info['width']}x{info['height']}, nodata={info['nodata']!r}.", source_path=path))
        except Exception as exc:
            checks.append(_result("Raster metadata", spec["artifact_id"], "FAIL", f"Raster cannot be read: {exc}", source_path=path))
    return metadata, checks


def validate_vectors(specs: Sequence[Mapping[str, Any]], configs: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[CheckResult]]:
    metadata: dict[str, dict[str, Any]] = {}
    checks: list[CheckResult] = []
    try:
        import geopandas as gpd
    except Exception as exc:
        return metadata, [_result("Vector metadata", "vector_reader", "FAIL", f"Vector metadata cannot be inspected because geopandas is unavailable: {exc}")]
    main_specs = [spec for spec in specs if spec["asset_class"] == "vector" and Path(spec["path"]).suffix.lower() in {".shp", ".gpkg", ".geojson"} and Path(spec["path"]).is_file()]
    for spec in main_specs:
        path = Path(spec["path"])
        try:
            frame = gpd.read_file(path)
            geometry_types = sorted(set(frame.geometry.geom_type.dropna().astype(str)))
            info = {"crs": str(frame.crs) if frame.crs else "", "feature_count": len(frame), "geometry_types": geometry_types}
            metadata[spec["artifact_id"]] = info
            problems = []
            if not info["crs"]: problems.append("CRS is missing")
            if info["feature_count"] <= 0: problems.append("feature count is zero")
            role = spec["input_role"]
            if role == "habitat_labels":
                if not set(geometry_types) <= {"Point"}:
                    problems.append(
                        f"Stage 04 requires Point geometry, found {geometry_types}"
                    )
                label = configs["sites"]["sites"][spec["site_id"]]["label"]
                field = label["positive_field"]
                if field not in frame.columns: problems.append(f"positive label field {field!r} is absent")
                elif label["positive_value"] not in set(frame[field].dropna().tolist()): problems.append(f"positive value {label['positive_value']!r} is absent from {field!r}")
            if role == "crop_boundary" and not set(geometry_types) <= {"Polygon", "MultiPolygon"}:
                problems.append(f"crop boundary must be Polygon geometry, found {geometry_types}")
            checks.append(_result("Vector metadata", spec["artifact_id"], "FAIL" if problems else "PASS", f"Vector validation failed: {', '.join(problems)}" if problems else f"Readable vector; CRS={info['crs']}, features={info['feature_count']}, geometry={geometry_types}.", source_path=path))
        except Exception as exc:
            checks.append(_result("Vector metadata", spec["artifact_id"], "FAIL", f"Vector cannot be read: {exc}", source_path=path))
    return metadata, checks


def config_input_specs(project_id: str, paths: Mapping[str, Path], feature_ids: Iterable[str], model_ids: Iterable[str]) -> list[dict[str, Any]]:
    specs = []
    fixed = (
        ("environment_spec", FRAMEWORK_ROOT / "environment.yml"),
        ("config_project", paths["project"]),
        ("config_sites", paths["sites"]),
        ("experiment_matrix", paths["matrix"]),
        ("analysis_config", paths["analysis"]),
        ("publication_config", paths["publication"]),
    )
    for role, path in fixed:
        specs.append({"project_id": project_id, "site_id": "", "input_role": role, "artifact_id": f"config:{role}", "path": path, "file_type": "YAML", "required": True, "asset_class": "config", "frequency_khz": "", "expected_band_count": "", "provenance": "repository configuration", "notes": "Pipeline configuration input."})
    for kind, ids, directory in (("feature_set_config", feature_ids, paths["feature_dir"]), ("model_config", model_ids, paths["model_dir"])):
        for item_id in ids:
            path = directory / f"{item_id}.yaml"
            specs.append({"project_id": project_id, "site_id": "", "input_role": kind, "artifact_id": f"config:{kind}:{item_id}", "path": path, "file_type": "YAML", "required": True, "asset_class": "config", "frequency_khz": "", "expected_band_count": "", "provenance": "experiment configuration", "notes": "Referenced by experiment_matrix.yaml."})
    return specs


def build_manifest_rows(specs: Sequence[Mapping[str, Any]], raster_metadata: Mapping[str, Mapping[str, Any]], vector_metadata: Mapping[str, Mapping[str, Any]], framework_root: Path = FRAMEWORK_ROOT) -> list[dict[str, Any]]:
    runtime_binding = runtime_path_binding_from_environment()
    rows = []
    for spec in specs:
        path = Path(spec["path"])
        exists = path.is_file()
        metadata = raster_metadata.get(spec["artifact_id"], vector_metadata.get(spec["artifact_id"], {}))
        relative_path = manifest_portable_path(
            path,
            framework_root,
            runtime_binding,
        )
        rows.append({
            "project_id": spec["project_id"], "site_id": spec["site_id"], "input_role": spec["input_role"], "artifact_id": spec["artifact_id"],
            # Absolute paths are runtime-only. Manifests retain portable logical
            # locators across the explicitly bound data, config, and code roots.
            "configured_path": relative_path, "repository_relative_path": relative_path, "file_type": spec["file_type"],
            "required": str(bool(spec["required"])).lower(), "exists": str(exists).lower(), "file_size_bytes": path.stat().st_size if exists else "",
            "sha256": sha256_file(path) if exists else "", "asset_class": spec["asset_class"], "frequency_khz": spec["frequency_khz"],
            "expected_band_count": spec["expected_band_count"], "observed_band_count": metadata.get("observed_band_count", ""), "crs": metadata.get("crs", ""),
            "provenance": spec["provenance"], "notes": spec["notes"],
        })
    return sorted(rows, key=lambda row: (row["asset_class"], row["site_id"], row["input_role"], str(row["frequency_khz"]), row["artifact_id"]))


def render_manifest(rows: Sequence[Mapping[str, Any]]) -> str:
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def atomic_write_if_changed(path: Path, content: str) -> bool:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try: os.unlink(temp_name)
        except FileNotFoundError: pass
        raise
    return True


def verify_manifest(path: Path, current_rows: Sequence[Mapping[str, Any]]) -> list[CheckResult]:
    if not path.is_file():
        return [_result("Input manifest", "manifest_exists", "FAIL", f"Input manifest is missing: {path}", source_path=path)]
    with path.open("r", encoding="utf-8", newline="") as stream:
        recorded = {row["artifact_id"]: row for row in csv.DictReader(stream)}
    current = {str(row["artifact_id"]): row for row in current_rows}
    checks = []
    missing_ids = sorted(set(current) - set(recorded))
    stale_ids = sorted(set(recorded) - set(current))
    checks.append(_result("Input manifest", "artifact_ids", "FAIL" if missing_ids or stale_ids else "PASS", f"Manifest ID mismatch: missing={missing_ids}, stale={stale_ids}" if missing_ids or stale_ids else f"Manifest contains the expected {len(current)} input artifacts."))
    mismatches = []
    for artifact_id in sorted(set(current) & set(recorded)):
        for field in ("exists", "file_size_bytes", "sha256"):
            if str(current[artifact_id][field]) != str(recorded[artifact_id][field]):
                mismatches.append(f"{artifact_id}:{field}")
    checks.append(_result("Input manifest", "checksums", "FAIL" if mismatches else "PASS", f"Manifest mismatches: {mismatches}" if mismatches else "All recorded input sizes and SHA-256 checksums match.", actual=mismatches, source_path=path))
    return checks


def check_environment(dependencies: Sequence[DependencySpec] = DEPENDENCIES) -> list[CheckResult]:
    checks = []
    for dependency in dependencies:
        try:
            module = import_module(dependency.module)
            try: installed = version(dependency.distribution)
            except PackageNotFoundError: installed = getattr(module, "__version__", "unknown")
            checks.append(_result("Environment", dependency.distribution, "PASS", f"{dependency.distribution} {installed} is importable ({dependency.used_by}).", expected=dependency.requirement, actual=installed))
        except Exception as exc:
            status = "FAIL" if dependency.requirement == "required" else "WARN"
            checks.append(_result("Environment", dependency.distribution, status, f"{dependency.distribution} is {dependency.requirement} but cannot be imported: {type(exc).__name__}: {exc}", expected=dependency.requirement, actual="missing"))
    gdalinfo = shutil.which("gdalinfo")
    checks.append(_result("Environment", "gdalinfo", "PASS" if gdalinfo else "WARN", f"GDAL command-line tools are available at {gdalinfo}." if gdalinfo else "gdalinfo is not on PATH. Pipeline raster I/O uses rasterio, but the native GDAL tool is useful for diagnostics.", expected="diagnostic tool", actual=gdalinfo or "missing"))
    return checks


def check_portability(framework_root: Path = FRAMEWORK_ROOT) -> list[CheckResult]:
    roots = [framework_root / "02_projects", framework_root / "03_experiments", framework_root / "04_pipeline", framework_root / "src/project_compiler"]
    findings: list[str] = []
    external_provenance: list[str] = []
    for root in roots:
        if not root.exists(): continue
        for path in sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".md"}):
            try: text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError: continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if "portability-test-fixture" in line:
                    continue
                if path.name == "preflight.py" and "Users|home" in line:
                    continue
                if "portability-pattern-definition" in line:
                    continue
                if PORTABILITY_PATTERN.search(line):
                    location = f"{repository_relative(path, framework_root)}:{line_number}"
                    if "source_project:" in line:
                        external_provenance.append(location)
                    else:
                        findings.append(location)
    status = "WARN" if findings else "PASS"
    message = f"Machine-specific path text occurs at {len(findings)} location(s): {', '.join(findings[:20])}{' ...' if len(findings) > 20 else ''}" if findings else "No machine-specific path text was found in configured scan roots."
    return [
        _result("Portability", "machine_specific_paths", status, message, expected=0, actual=len(findings), source_path=framework_root),
        _result(
            "Portability",
            "external_provenance_paths",
            "PASS",
            f"Recorded {len(external_provenance)} machine-specific external provenance locator(s); these are not runtime inputs: {', '.join(external_provenance) or 'none'}.",
            expected="informational",
            actual=len(external_provenance),
            source_path=framework_root / "02_projects",
        ),
    ]


def limitation_checks() -> list[CheckResult]:
    limitations: tuple[tuple[str, str, str], ...] = ()
    return [_result("Framework limitations", name, "WARN", message, source_path=FRAMEWORK_ROOT / path) for name, message, path in limitations]


def overall_status(checks: Sequence[CheckResult]) -> str:
    if any(check.status == "FAIL" for check in checks): return "FAIL"
    if any(check.status == "WARN" for check in checks): return "WARN"
    return "PASS"


def summary_dict(project_id: str, checks: Sequence[CheckResult]) -> dict[str, Any]:
    categories = {}
    for category in sorted({check.category for check in checks}):
        selected = [check for check in checks if check.category == category]
        categories[category] = {"status": overall_status(selected), "pass": sum(c.status == "PASS" for c in selected), "warn": sum(c.status == "WARN" for c in selected), "fail": sum(c.status == "FAIL" for c in selected)}
    return {"project_id": project_id, "overall": overall_status(checks), "pass": sum(c.status == "PASS" for c in checks), "warn": sum(c.status == "WARN" for c in checks), "fail": sum(c.status == "FAIL" for c in checks), "categories": categories, "checks": [asdict(check) for check in checks]}


def render_text_summary(summary: Mapping[str, Any]) -> str:
    lines = ["=" * 60, "Project Preflight", "=" * 60, f"Project: {summary['project_id']}"]
    for category, values in summary["categories"].items():
        lines.append(f"{category}: {values['status']} (pass={values['pass']}, warn={values['warn']}, fail={values['fail']})")
    lines.extend([f"Overall: {summary['overall']}", "=" * 60])
    notable = [check for check in summary["checks"] if check["status"] in {"WARN", "FAIL"}]
    if notable:
        lines.append("Details")
        for check in notable:
            lines.append(f"{check['status']}: [{check['category']}] {check['check_name']}: {check['message']}")
    return "\n".join(lines) + "\n"


def json_text(summary: Mapping[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

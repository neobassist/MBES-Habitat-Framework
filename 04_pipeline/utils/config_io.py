#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Configuration I/O helpers for the MBES Habitat Framework.

The functions in this module are intentionally strict about the configured
project YAML layout. Missing files, empty YAML documents, and malformed sections
raise clear exceptions so pipeline stages fail early.
"""

from __future__ import annotations

from itertools import product
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Mapping, NamedTuple, Sequence


DEFAULT_PROJECT_ID = "MER_2026_DB"
FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_BINDING_ENV_VAR = "MBES_RUNTIME_BINDING_FILE"


class RuntimePathBinding(NamedTuple):
    """Explicit split roots used by materialized-project stage adapters."""

    config_root: Path
    data_root: Path
    data_locator: str
    framework_code_root: Path
    output_root: Path
    docs_root: Path


def load_runtime_path_binding(path: str | Path) -> RuntimePathBinding:
    """Load one explicit B4 child-process root binding."""

    supplied_path = _as_path(path)
    if supplied_path.is_symlink():
        raise ValueError("Runtime binding file must not be a symlink.")
    binding_path = supplied_path.resolve(strict=True)
    if not binding_path.is_file():
        raise ValueError("Runtime binding file must be a physical regular file.")
    try:
        document = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Runtime binding file is invalid: {binding_path}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Runtime binding file must be a schema-version-1 mapping.")
    roots = document.get("roots")
    if not isinstance(roots, dict):
        raise ValueError("Runtime binding file has no roots mapping.")
    data_locator = document.get("data_locator")
    if not isinstance(data_locator, str):
        raise ValueError("Runtime binding file has no string data_locator.")
    binding = RuntimePathBinding(
        config_root=Path(roots.get("config_root", "")),
        data_root=Path(roots.get("data_root", "")),
        data_locator=data_locator,
        framework_code_root=Path(roots.get("framework_code_root", "")),
        output_root=Path(roots.get("output_root", "")),
        docs_root=Path(roots.get("docs_root", "")),
    )
    _validate_runtime_binding(binding)
    return binding


def runtime_path_binding_from_environment() -> RuntimePathBinding | None:
    """Return the child-scoped B4 binding without mutating process globals."""

    path = os.environ.get(RUNTIME_BINDING_ENV_VAR)
    if path is None:
        return None
    if not path:
        raise ValueError(f"{RUNTIME_BINDING_ENV_VAR} must not be empty.")
    return load_runtime_path_binding(path)


def _effective_runtime_binding(
    runtime_binding: RuntimePathBinding | None,
) -> RuntimePathBinding | None:
    return (
        runtime_binding
        if runtime_binding is not None
        else runtime_path_binding_from_environment()
    )


def _portable_locator(value: str | Path, field_name: str) -> PurePosixPath:
    text = str(value)
    if not text or "\\" in text:
        raise ValueError(f"{field_name} must be a non-empty relative POSIX path.")
    locator = PurePosixPath(text)
    if (
        locator.is_absolute()
        or locator in {PurePosixPath("."), PurePosixPath("..")}
        or ".." in locator.parts
    ):
        raise ValueError(f"{field_name} must not be absolute or contain traversal.")
    return locator


def _require_yaml() -> Any:
    """Import PyYAML with a clear dependency error."""

    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyYAML is required for YAML configuration I/O. "
            "Install it with `pip install pyyaml` or a project environment file."
        ) from exc
    return yaml


def _as_path(path: str | Path) -> Path:
    """Return ``path`` as an expanded ``Path``."""

    return Path(path).expanduser()


def _project_root(framework_root: str | Path | None = None) -> Path:
    """Return the repository root used for config discovery."""

    if framework_root is None:
        return FRAMEWORK_ROOT
    return _as_path(framework_root).resolve()


def _config_root(
    framework_root: str | Path | None,
    runtime_binding: RuntimePathBinding | None,
) -> Path:
    runtime_binding = _effective_runtime_binding(runtime_binding)
    if runtime_binding is None:
        return _project_root(framework_root)
    if not isinstance(runtime_binding, RuntimePathBinding):
        raise TypeError("runtime_binding must be RuntimePathBinding or None.")
    _validate_runtime_binding(runtime_binding)
    if framework_root is not None:
        raise ValueError("framework_root and runtime_binding are mutually exclusive authorities.")
    return runtime_binding.config_root


def _validate_runtime_binding(runtime_binding: RuntimePathBinding) -> None:
    for name in (
        "config_root",
        "data_root",
        "framework_code_root",
        "output_root",
        "docs_root",
    ):
        if not isinstance(getattr(runtime_binding, name), Path):
            raise TypeError(f"RuntimePathBinding.{name} must be pathlib.Path.")
        if not getattr(runtime_binding, name).is_absolute():
            raise ValueError(f"RuntimePathBinding.{name} must be absolute.")
    _portable_locator(runtime_binding.data_locator, "RuntimePathBinding.data_locator")


def resolve_runtime_data_path(
    path: str | Path,
    runtime_binding: RuntimePathBinding,
    *,
    strict: bool = False,
) -> Path:
    """Resolve one materialized input locator against the explicit project data root."""

    if not isinstance(runtime_binding, RuntimePathBinding):
        raise TypeError("runtime_binding must be RuntimePathBinding.")
    _validate_runtime_binding(runtime_binding)
    locator = _portable_locator(path, "path")
    prefix = PurePosixPath(runtime_binding.data_locator)
    try:
        suffix = locator.relative_to(prefix)
    except ValueError as exc:
        raise ValueError(
            f"Input locator is outside the configured data authority: {path!s}"
        ) from exc
    resolved = runtime_binding.data_root.joinpath(*suffix.parts).resolve(strict=strict)
    try:
        resolved.relative_to(runtime_binding.data_root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"Input path escapes the bound data root: {path!s}") from exc
    return resolved


def resolve_repo_path(
    path: str | Path,
    framework_root: str | Path | None = None,
    *,
    strict: bool = False,
) -> Path:
    """Resolve a canonical locator independently of the current directory.

    Relative locators are interpreted from the repository root discovered from
    this module (or an explicitly supplied test root). Absolute locators remain
    supported for legacy configurations and genuine external inputs.
    """

    runtime_binding = runtime_path_binding_from_environment()
    if runtime_binding is not None:
        if (
            framework_root is not None
            and _project_root(framework_root) != runtime_binding.framework_code_root
        ):
            raise ValueError("Explicit framework_root conflicts with active runtime binding.")
        return _resolve_bound_locator(path, runtime_binding, strict=strict)

    root = _project_root(framework_root)
    locator = _as_path(path)
    if locator.is_absolute():
        return locator.resolve(strict=strict)

    resolved = (root / locator).resolve(strict=strict)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Repository-relative path escapes the repository root: {path!s}"
        ) from exc
    return resolved


def _resolve_bound_locator(
    path: str | Path,
    runtime_binding: RuntimePathBinding,
    *,
    strict: bool,
) -> Path:
    """Resolve a legacy repository locator through explicit split-root authority."""

    locator = _portable_locator(path, "path")
    parts = locator.parts
    data_prefix = PurePosixPath(runtime_binding.data_locator)
    try:
        data_suffix = locator.relative_to(data_prefix)
    except ValueError:
        data_suffix = None
    if data_suffix is not None:
        root = runtime_binding.data_root
        suffix = data_suffix.parts
    elif parts[0] in {"02_projects", "03_experiments"}:
        root = runtime_binding.config_root
        suffix = parts
    elif parts[0] == "04_pipeline":
        root = runtime_binding.framework_code_root
        suffix = parts
    elif len(parts) >= 2 and parts[0] == "05_outputs":
        if parts[1] != runtime_binding.output_root.name:
            raise ValueError("Output locator belongs to another project.")
        root = runtime_binding.output_root
        suffix = parts[2:]
    elif len(parts) >= 3 and parts[:2] == ("06_docs", "projects"):
        if parts[2] != runtime_binding.docs_root.name:
            raise ValueError("Documentation locator belongs to another project.")
        root = runtime_binding.docs_root
        suffix = parts[3:]
    else:
        raise ValueError(f"Runtime locator has no bound authority root: {path!s}")
    resolved = root.joinpath(*suffix).resolve(strict=strict)
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"Runtime locator escapes its authority root: {path!s}") from exc
    return resolved


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Validate that ``value`` is a mapping and return it."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _require_keys(mapping: Mapping[str, Any], keys: list[str], name: str) -> None:
    """Raise ``ValueError`` when required ``keys`` are absent."""

    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{name} is missing required key(s): {', '.join(missing)}.")


def _require_list(value: Any, name: str) -> list[Any]:
    """Validate that ``value`` is a list and return it."""

    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list, got {type(value).__name__}.")
    return value


def _require_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is a bool and return it."""

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value).__name__}.")
    return value


def _require_numeric(value: Any, name: str) -> int | float:
    """Validate that ``value`` is numeric and return it."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}.")
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary.

    Parameters
    ----------
    path:
        YAML file path.

    Returns
    -------
    dict[str, Any]
        Parsed YAML document.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is empty or the root YAML object is not a mapping.
    """

    yaml = _require_yaml()
    yaml_path = _as_path(path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    if not yaml_path.is_file():
        raise ValueError(f"YAML path is not a file: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if data is None:
        raise ValueError(f"YAML file is empty: {yaml_path}")
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML root must be a mapping in {yaml_path}, got {type(data).__name__}."
        )

    return data


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    """Save a mapping to a YAML file.

    Parent directories are created when needed. The function refuses non-mapping
    input because every project configuration file currently uses a mapping root.
    """

    yaml = _require_yaml()
    _require_mapping(data, "data")

    yaml_path = _as_path(path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with yaml_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            dict(data),
            stream,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def load_project_config(
    project_id: str = DEFAULT_PROJECT_ID,
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, Any]:
    """Load and validate ``02_projects/<project_id>/config_project.yaml``."""

    path = _config_root(framework_root, runtime_binding) / "02_projects" / project_id / "config_project.yaml"
    config = load_yaml(path)
    validate_project_config(config)
    return config


def load_sites_config(
    project_id: str = DEFAULT_PROJECT_ID,
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, Any]:
    """Load and validate ``02_projects/<project_id>/config_sites.yaml``."""

    path = _config_root(framework_root, runtime_binding) / "02_projects" / project_id / "config_sites.yaml"
    config = load_yaml(path)
    validate_sites_config(config)
    return config


def load_experiment_matrix(
    project_id: str = DEFAULT_PROJECT_ID,
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, Any]:
    """Load and validate ``03_experiments/<project_id>/experiment_matrix.yaml``."""

    path = (
        _config_root(framework_root, runtime_binding)
        / "03_experiments"
        / project_id
        / "experiment_matrix.yaml"
    )
    config = load_yaml(path)
    validate_experiment_matrix(config)
    return config


def load_analysis_config(
    project_id: str = DEFAULT_PROJECT_ID,
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, Any]:
    """Load and validate ``03_experiments/<project_id>/analysis.yaml``."""

    path = _config_root(framework_root, runtime_binding) / "03_experiments" / project_id / "analysis.yaml"
    config = load_yaml(path)
    matrix = load_experiment_matrix(
        project_id=project_id,
        framework_root=framework_root,
        runtime_binding=runtime_binding,
    )
    validate_analysis_config(config, matrix, expected_project_id=project_id)
    return config


FIGURE_BUILDER_IDS = frozenset(
    {
        "study_areas",
        "workflow_validation",
        "mbes_feature_definitions",
        "feature_set_comparison",
        "model_comparison",
        "feature_model_heatmap",
        "final_candidate_comparison",
        "main_feature_importance",
        "predicted_habitat_maps",
        "within_site_validation",
        "cross_site_validation",
        "spatial_cv_validation",
        "integrated_comparison",
        "ranking_stability",
        "final_candidate_analysis",
        "supplementary_feature_importance",
        "representative_validation",
    }
)
TABLE_BUILDER_IDS = frozenset(
    {
        "publication_site_summary",
        "key_validation_performance",
        "final_candidate_selection_summary",
        "primary_habitat_summary",
        "feature_set_definitions",
        "model_config_summary",
        "validation_publication_summary",
        "rank_stability_shift_summary",
        "final_candidate_map_statistics",
        "portable_feature_importance",
        "representative_validation_table",
    }
)
SOURCE_DATA_BUILDER_IDS = frozenset(
    {"validation_source", "figure_source", "map_source", "inventory_source"}
)


def load_publication_config(
    project_id: str = DEFAULT_PROJECT_ID,
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, Any]:
    """Load and cross-validate ``publication.yaml`` for one project."""

    path = _config_root(framework_root, runtime_binding) / "03_experiments" / project_id / "publication.yaml"
    config = load_yaml(path)
    matrix = load_experiment_matrix(
        project_id=project_id,
        framework_root=framework_root,
        runtime_binding=runtime_binding,
    )
    analysis = load_analysis_config(
        project_id=project_id,
        framework_root=framework_root,
        runtime_binding=runtime_binding,
    )
    validate_publication_config(
        config,
        matrix,
        analysis,
        expected_project_id=project_id,
    )
    return config


def load_feature_set_config(
    feature_set_id: str,
    project_id: str = DEFAULT_PROJECT_ID,
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, Any]:
    """Load and validate one feature-set YAML file."""

    path = (
        _config_root(framework_root, runtime_binding)
        / "03_experiments"
        / project_id
        / "feature_sets"
        / f"{feature_set_id}.yaml"
    )
    config = load_yaml(path)
    _validate_feature_set_config(config, feature_set_id)
    return config


def load_all_feature_set_configs(
    project_id: str = DEFAULT_PROJECT_ID,
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, dict[str, Any]]:
    """Load all feature-set configs listed in the experiment matrix."""

    matrix = load_experiment_matrix(
        project_id=project_id,
        framework_root=framework_root,
        runtime_binding=runtime_binding,
    )
    feature_set_ids = _require_list(matrix.get("feature_sets"), "experiment_matrix.feature_sets")

    configs: dict[str, dict[str, Any]] = {}
    for feature_set_id in feature_set_ids:
        if not isinstance(feature_set_id, str):
            raise ValueError("experiment_matrix.feature_sets must contain only strings.")
        configs[feature_set_id] = load_feature_set_config(
            feature_set_id,
            project_id=project_id,
            framework_root=framework_root,
            runtime_binding=runtime_binding,
        )
    return configs


def load_model_config(
    model_id: str,
    project_id: str = DEFAULT_PROJECT_ID,
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, Any]:
    """Load and validate one model YAML file."""

    path = (
        _config_root(framework_root, runtime_binding)
        / "03_experiments"
        / project_id
        / "models"
        / f"{model_id}.yaml"
    )
    config = load_yaml(path)
    _validate_model_config(config, model_id)
    return config


def load_all_model_configs(
    project_id: str = DEFAULT_PROJECT_ID,
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, dict[str, Any]]:
    """Load all model configs listed in the experiment matrix."""

    matrix = load_experiment_matrix(
        project_id=project_id,
        framework_root=framework_root,
        runtime_binding=runtime_binding,
    )
    model_ids = _require_list(matrix.get("models"), "experiment_matrix.models")

    configs: dict[str, dict[str, Any]] = {}
    for model_id in model_ids:
        if not isinstance(model_id, str):
            raise ValueError("experiment_matrix.models must contain only strings.")
        configs[model_id] = load_model_config(
            model_id,
            project_id=project_id,
            framework_root=framework_root,
            runtime_binding=runtime_binding,
        )
    return configs


def load_all_configs(
    project_id: str = DEFAULT_PROJECT_ID,
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, Any]:
    """Load project, site, experiment, feature-set, and model configs together."""

    return {
        "project": load_project_config(
            project_id=project_id,
            framework_root=framework_root,
            runtime_binding=runtime_binding,
        ),
        "sites": load_sites_config(
            project_id=project_id,
            framework_root=framework_root,
            runtime_binding=runtime_binding,
        ),
        "experiment_matrix": load_experiment_matrix(
            project_id=project_id,
            framework_root=framework_root,
            runtime_binding=runtime_binding,
        ),
        "analysis": load_analysis_config(
            project_id=project_id,
            framework_root=framework_root,
            runtime_binding=runtime_binding,
        ),
        "publication": load_publication_config(
            project_id=project_id,
            framework_root=framework_root,
            runtime_binding=runtime_binding,
        ),
        "feature_sets": load_all_feature_set_configs(
            project_id=project_id,
            framework_root=framework_root,
            runtime_binding=runtime_binding,
        ),
        "models": load_all_model_configs(
            project_id=project_id,
            framework_root=framework_root,
            runtime_binding=runtime_binding,
        ),
    }


SUPPORTED_ANALYSIS_SOURCES = ("within_site", "cross_site", "spatial_block_cv")


def _publication_root(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _require_mapping(config.get("publication"), "publication_config.publication")


def _ordered_publication_items(
    config: Mapping[str, Any],
    section: str,
    *,
    parent: str = "publication",
) -> tuple[dict[str, Any], ...]:
    """Return one ordered publication inventory after common ID/order checks."""

    root = _publication_root(config)
    container = root if parent == "publication" else _require_mapping(root.get(parent), f"publication.{parent}")
    values = _require_list(container.get(section), f"publication.{parent}.{section}" if parent != "publication" else f"publication.{section}")
    seen_ids: set[str] = set()
    seen_orders: set[Any] = set()
    items: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        name = f"publication.{section}[{index}]"
        item = dict(_require_mapping(value, name))
        _require_keys(item, ["id", "order"], name)
        item_id = item["id"]
        order = item["order"]
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{name}.id must be a non-empty string.")
        if "/" in item_id or "\\" in item_id or ".." in Path(item_id).parts:
            raise ValueError(f"{name}.id must be a portable identifier without path traversal.")
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError(f"{name}.order must be a positive integer.")
        if item_id in seen_ids:
            raise ValueError(f"{name}.id duplicates {item_id!r}.")
        order_key: Any = (item.get("role"), order) if section in {"figures", "tables"} else order
        if order_key in seen_orders:
            raise ValueError(f"{name}.order duplicates order {order}.")
        seen_ids.add(item_id)
        seen_orders.add(order_key)
        items.append(item)
    return tuple(sorted(items, key=lambda item: item["order"]))


def configured_publication_display(
    config: Mapping[str, Any], section: str
) -> tuple[dict[str, Any], ...]:
    """Return a validated display registry in publication order."""

    return _ordered_publication_items(config, section, parent="display")


def configured_figure_specs(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return _ordered_publication_items(config, "figures")


def configured_table_specs(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return _ordered_publication_items(config, "tables")


def configured_source_data_specs(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return _ordered_publication_items(config, "source_data")


def configured_representative_runs(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return representative validation runs in declared YAML order."""

    root = _publication_root(config)
    values = _require_list(root.get("representative_runs"), "publication.representative_runs")
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        name = f"publication.representative_runs[{index}]"
        item = dict(_require_mapping(value, name))
        _require_keys(
            item,
            [
                "id", "figure_id", "table_id", "target_site", "train_site", "feature_set",
                "model", "seed", "spatial_folds", "spatial_aggregation_method",
            ],
            name,
        )
        if not isinstance(item["id"], str) or not item["id"]:
            raise ValueError(f"{name}.id must be a non-empty string.")
        if item["id"] in seen:
            raise ValueError(f"{name}.id duplicates {item['id']!r}.")
        if item["target_site"] == item["train_site"]:
            raise ValueError(f"{name} train_site and target_site must differ.")
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{name}.seed must be a non-negative integer.")
        folds = _require_list(item["spatial_folds"], f"{name}.spatial_folds")
        if not folds or any(isinstance(fold, bool) or not isinstance(fold, int) or fold < 0 for fold in folds):
            raise ValueError(f"{name}.spatial_folds must contain non-negative integers.")
        if len(set(folds)) != len(folds):
            raise ValueError(f"{name}.spatial_folds contains duplicates.")
        if not isinstance(item["spatial_aggregation_method"], str) or not item["spatial_aggregation_method"].strip():
            raise ValueError(f"{name}.spatial_aggregation_method must not be empty.")
        seen.add(item["id"])
        runs.append(item)
    return tuple(runs)


def publication_display_labels(config: Mapping[str, Any]) -> dict[str, str]:
    """Return all publication labels keyed by upstream identifier."""

    display = _require_mapping(_publication_root(config).get("display"), "publication.display")
    labels = dict(_require_mapping(display.get("labels", {}), "publication.display.labels"))
    for section in ("sites", "candidates", "feature_sets", "models", "validations", "feature_groups"):
        for item in configured_publication_display(config, section):
            label = item.get("label")
            if not isinstance(label, str) or not label:
                raise ValueError(f"publication.display.{section} item {item['id']!r} requires label.")
            labels[item["id"]] = label
    return {str(key): str(value) for key, value in labels.items()}


def configured_publication_contract(config: Mapping[str, Any]) -> dict[str, int]:
    """Return declared publication counts after checking parsed inventories."""

    root = _publication_root(config)
    contract = dict(_require_mapping(root.get("contract"), "publication.contract"))
    required = [
        "main_figures", "supplementary_figures", "main_panels", "supplementary_panels",
        "main_tables", "supplementary_tables", "source_data",
    ]
    _require_keys(contract, required, "publication.contract")
    if any(isinstance(contract[key], bool) or not isinstance(contract[key], int) or contract[key] < 0 for key in required):
        raise ValueError("publication.contract counts must be non-negative integers.")
    figures = configured_figure_specs(config)
    tables = configured_table_specs(config)
    calculated = {
        "main_figures": sum(item.get("role") == "main" for item in figures),
        "supplementary_figures": sum(item.get("role") == "supplementary" for item in figures),
        "main_panels": sum(len(item.get("panels", [])) for item in figures if item.get("role") == "main"),
        "supplementary_panels": sum(len(item.get("panels", [])) for item in figures if item.get("role") == "supplementary"),
        "main_tables": sum(item.get("role") == "main" for item in tables),
        "supplementary_tables": sum(item.get("role") == "supplementary" for item in tables),
        "source_data": len(configured_source_data_specs(config)),
    }
    mismatch = {key: (contract[key], value) for key, value in calculated.items() if contract[key] != value}
    if mismatch:
        raise ValueError(f"publication.contract count mismatch: {mismatch}.")
    return {key: int(contract[key]) for key in required}


def validate_publication_config(
    config: Mapping[str, Any],
    experiment_matrix: Mapping[str, Any],
    analysis_config: Mapping[str, Any],
    *,
    expected_project_id: str | None = None,
) -> None:
    """Validate publication policy and all upstream references."""

    root = _require_mapping(config, "publication_config")
    _require_keys(root, ["schema_version", "project_id", "publication"], "publication_config")
    if root["schema_version"] != 1:
        raise ValueError(f"Unsupported publication schema_version: {root['schema_version']!r}.")
    if expected_project_id is not None and root["project_id"] != expected_project_id:
        raise ValueError(
            f"publication project_id {root['project_id']!r} does not match {expected_project_id!r}."
        )
    publication = _publication_root(config)
    _require_keys(
        publication,
        ["display", "primary_candidate_id", "representative_runs", "figures", "tables", "source_data", "contract"],
        "publication",
    )
    table_documentation_mode = publication.get("table_documentation_mode", "canonical")
    if table_documentation_mode not in {"canonical", "generic"}:
        raise ValueError(
            "publication.table_documentation_mode must be 'canonical' or 'generic'."
        )
    matrix_sites = set(_require_list(experiment_matrix.get("sites"), "experiment_matrix.sites"))
    matrix_features = set(_require_list(experiment_matrix.get("feature_sets"), "experiment_matrix.feature_sets"))
    matrix_models = set(_require_list(experiment_matrix.get("models"), "experiment_matrix.models"))
    analysis_candidates = {
        item["candidate_label"] for item in configured_analysis_candidates(analysis_config)
    }
    analysis_sources = {item["id"] for item in configured_validation_sources(analysis_config)}
    allowed_display = {
        "sites": matrix_sites,
        "candidates": analysis_candidates,
        "feature_sets": matrix_features,
        "models": matrix_models,
        "validations": analysis_sources,
    }
    for section, allowed in allowed_display.items():
        display_items = configured_publication_display(config, section)
        if section == "candidates":
            forbidden = [
                item["id"] for item in display_items
                if "feature_set" in item or "model" in item or "model_id" in item
            ]
            if forbidden:
                raise ValueError(
                    "publication candidate display must not duplicate scientific feature/model identity: "
                    f"{forbidden}."
                )
        actual = {item["id"] for item in display_items}
        unknown = sorted(actual - allowed)
        if unknown:
            raise ValueError(f"publication.display.{section} references unknown IDs: {unknown}.")
    configured_publication_display(config, "feature_groups")

    primary = publication["primary_candidate_id"]
    if primary not in analysis_candidates:
        raise ValueError(f"publication.primary_candidate_id references unknown candidate {primary!r}.")

    figures = configured_figure_specs(config)
    tables = configured_table_specs(config)
    sources = configured_source_data_specs(config)
    figure_ids = {item["id"] for item in figures}
    table_ids = {item["id"] for item in tables}
    source_ids = {item["id"] for item in sources}

    def validate_artifacts(items: Sequence[Mapping[str, Any]], kind: str, builders: set[str] | frozenset[str]) -> None:
        seen_numbers: set[str] = set()
        seen_stems: set[str] = set()
        role_orders: set[tuple[str, int]] = set()
        for item in items:
            name = f"publication.{kind}.{item['id']}"
            required = ["stem", "builder"]
            if kind != "source_data":
                required.extend(["role", "number", "caption"])
            _require_keys(item, required, name)
            stem = item["stem"]
            if not isinstance(stem, str) or not stem or Path(stem).name != stem or ".." in Path(stem).parts:
                raise ValueError(f"{name}.stem must be a portable filename stem.")
            if stem in seen_stems:
                raise ValueError(f"{name}.stem duplicates {stem!r}.")
            seen_stems.add(stem)
            if item["builder"] not in builders:
                raise ValueError(f"{name}.builder references unknown builder {item['builder']!r}.")
            if kind != "source_data":
                if item["role"] not in {"main", "supplementary"}:
                    raise ValueError(f"{name}.role must be main or supplementary.")
                if not isinstance(item["caption"], str) or not item["caption"].strip():
                    raise ValueError(f"{name}.caption must not be empty.")
                number = item["number"]
                if number in seen_numbers:
                    raise ValueError(f"{name}.number duplicates {number!r}.")
                seen_numbers.add(number)
                role_order = (item["role"], item["order"])
                if role_order in role_orders:
                    raise ValueError(f"{name}.order duplicates order {item['order']} in role {item['role']}.")
                role_orders.add(role_order)

    validate_artifacts(figures, "figures", FIGURE_BUILDER_IDS)
    validate_artifacts(tables, "tables", TABLE_BUILDER_IDS)
    validate_artifacts(sources, "source_data", SOURCE_DATA_BUILDER_IDS)

    representative_ids = {item["id"] for item in configured_representative_runs(config)}
    for table in tables:
        parameters = _require_mapping(table.get("parameters", {}), f"table {table['id']}.parameters")
        if table["builder"] == "validation_publication_summary":
            source = parameters.get("validation_source")
            if source not in analysis_sources:
                raise ValueError(
                    f"Table {table['id']} references unknown validation source {source!r}."
                )
        if table["builder"] == "representative_validation_table":
            run_id = parameters.get("representative_run")
            if run_id not in representative_ids:
                raise ValueError(
                    f"Table {table['id']} references unknown representative run {run_id!r}."
                )

    panel_ids: set[str] = set()
    panel_stems: set[str] = set()
    selection_references = {
        "site": matrix_sites,
        "target_site": matrix_sites,
        "candidate": analysis_candidates,
        "feature_set": matrix_features,
        "model": matrix_models,
        "validation_strategy": analysis_sources,
    }

    def validate_selection(selection: Mapping[str, Any], name: str) -> None:
        for key, allowed in selection_references.items():
            if key not in selection:
                continue
            raw = selection[key]
            values = [part.strip() for part in str(raw).split(";") if part.strip()]
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise ValueError(f"{name}.{key} references unknown IDs: {unknown}.")

    for figure in figures:
        validate_selection(
            _require_mapping(figure.get("selection", {}), f"figure {figure['id']}.selection"),
            f"figure {figure['id']}.selection",
        )
        panels = _require_list(figure.get("panels", []), f"figure {figure['id']}.panels")
        seen_letters: set[str] = set()
        for panel in panels:
            panel = _require_mapping(panel, f"figure {figure['id']} panel")
            _require_keys(panel, ["id", "letter", "title", "stem"], f"figure {figure['id']} panel")
            if panel["id"] in panel_ids or panel["stem"] in panel_stems:
                raise ValueError(f"Duplicate panel ID/stem in figure {figure['id']}: {panel['id']!r}.")
            if panel["letter"] in seen_letters:
                raise ValueError(f"Figure {figure['id']} duplicates panel letter {panel['letter']!r}.")
            validate_selection(
                _require_mapping(panel.get("selection", {}), f"panel {panel['id']}.selection"),
                f"panel {panel['id']}.selection",
            )
            panel_ids.add(panel["id"]); panel_stems.add(panel["stem"]); seen_letters.add(panel["letter"])
        unknown_tables = sorted(set(figure.get("paired_table_ids", [])) - table_ids)
        unknown_sources = sorted(set(figure.get("paired_source_data_ids", [])) - source_ids)
        if unknown_tables or unknown_sources:
            raise ValueError(
                f"Figure {figure['id']} has invalid pairing targets: tables={unknown_tables}, sources={unknown_sources}."
            )
    for table in tables:
        unknown = sorted(set(table.get("paired_figure_ids", [])) - figure_ids)
        if unknown:
            raise ValueError(f"Table {table['id']} has invalid Figure pairings: {unknown}.")
    for source in sources:
        unknown = sorted(set(source.get("paired_figure_ids", [])) - figure_ids)
        if unknown:
            raise ValueError(f"Source Data {source['id']} has invalid Figure pairings: {unknown}.")

    for run in configured_representative_runs(config):
        refs = {
            "site": ({run["target_site"], run["train_site"]}, matrix_sites),
            "feature": ({run["feature_set"]}, matrix_features),
            "model": ({run["model"]}, matrix_models),
            "figure": ({run["figure_id"]}, figure_ids),
            "table": ({run["table_id"]}, table_ids),
        }
        for label, (actual, allowed) in refs.items():
            unknown = sorted(actual - allowed)
            if unknown:
                raise ValueError(f"Representative run {run['id']} references unknown {label}: {unknown}.")
    publication_display_labels(config)
    configured_publication_contract(config)


def configured_validation_sources(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return validated analysis sources in configuration order."""

    analysis = _require_mapping(config.get("analysis"), "analysis_config.analysis")
    values = _require_list(analysis.get("validation_sources"), "analysis.validation_sources")
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        name = f"analysis.validation_sources[{index}]"
        source = _require_mapping(value, name)
        _require_keys(source, ["id", "weight"], name)
        source_id = source["id"]
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{name}.id must be a non-empty string.")
        if source_id not in SUPPORTED_ANALYSIS_SOURCES:
            raise ValueError(f"{name}.id references unsupported validation source {source_id!r}.")
        if source_id in seen:
            raise ValueError(f"{name}.id duplicates validation source {source_id!r}.")
        weight = _require_numeric(source["weight"], f"{name}.weight")
        if weight <= 0:
            raise ValueError(f"{name}.weight must be > 0.")
        seen.add(source_id)
        sources.append({"id": source_id, "weight": float(weight)})
    if not sources:
        raise ValueError("analysis.validation_sources must not be empty.")
    return tuple(sources)


def configured_site_rank_comparisons(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return validated site-rank comparisons in configuration order."""

    analysis = _require_mapping(config.get("analysis"), "analysis_config.analysis")
    values = _require_list(
        analysis.get("site_rank_comparisons"),
        "analysis.site_rank_comparisons",
    )
    comparisons: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_ids = {source["id"] for source in configured_validation_sources(config)}
    for index, value in enumerate(values):
        name = f"analysis.site_rank_comparisons[{index}]"
        item = _require_mapping(value, name)
        _require_keys(item, ["id", "sites", "validation_sources"], name)
        comparison_id = item["id"]
        if not isinstance(comparison_id, str) or not comparison_id:
            raise ValueError(f"{name}.id must be a non-empty string.")
        if not comparison_id.replace("_", "").isalnum():
            raise ValueError(f"{name}.id must contain only letters, numbers, and underscores.")
        if comparison_id in seen:
            raise ValueError(f"{name}.id duplicates comparison {comparison_id!r}.")
        sites = _require_list(item["sites"], f"{name}.sites")
        if len(sites) != 2 or any(not isinstance(site, str) or not site for site in sites):
            raise ValueError(f"{name}.sites must contain exactly two non-empty site IDs.")
        if sites[0] == sites[1]:
            raise ValueError(f"{name}.sites must contain two different site IDs.")
        sources = _require_list(item["validation_sources"], f"{name}.validation_sources")
        if not sources or any(source not in source_ids for source in sources):
            raise ValueError(f"{name}.validation_sources must reference configured sources.")
        if len(set(sources)) != len(sources):
            raise ValueError(f"{name}.validation_sources contains duplicates.")
        seen.add(comparison_id)
        comparisons.append(
            {"id": comparison_id, "sites": tuple(sites), "validation_sources": tuple(sources)}
        )
    return tuple(comparisons)


def configured_analysis_candidates(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return explicit candidates ordered by their configured order field."""

    analysis = _require_mapping(config.get("analysis"), "analysis_config.analysis")
    selection = _require_mapping(
        analysis.get("candidate_selection"),
        "analysis.candidate_selection",
    )
    if selection.get("mode") != "explicit":
        raise ValueError("analysis.candidate_selection.mode must be 'explicit'.")
    values = _require_list(selection.get("candidates"), "analysis.candidate_selection.candidates")
    if not values:
        raise ValueError("analysis.candidate_selection.candidates must not be empty.")
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_orders: set[int] = set()
    for index, value in enumerate(values):
        name = f"analysis.candidate_selection.candidates[{index}]"
        item = _require_mapping(value, name)
        _require_keys(item, ["id", "feature_set", "model", "order"], name)
        candidate_id = item["id"]
        feature_set = item["feature_set"]
        model_id = item["model"]
        order = item["order"]
        if any(not isinstance(field, str) or not field for field in (candidate_id, feature_set, model_id)):
            raise ValueError(f"{name} id, feature_set, and model must be non-empty strings.")
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError(f"{name}.order must be a positive integer.")
        if candidate_id in seen_ids:
            raise ValueError(f"{name}.id duplicates candidate {candidate_id!r}.")
        if order in seen_orders:
            raise ValueError(f"{name}.order duplicates order {order}.")
        seen_ids.add(candidate_id)
        seen_orders.add(order)
        candidates.append(
            {
                "candidate_label": candidate_id,
                "feature_set": feature_set,
                "model_id": model_id,
                "order": order,
                "role": str(item.get("role", "candidate")),
            }
        )
    ordered = tuple(sorted(candidates, key=lambda item: item["order"]))
    expected_orders = tuple(range(1, len(ordered) + 1))
    actual_orders = tuple(item["order"] for item in ordered)
    if actual_orders != expected_orders:
        raise ValueError(
            "analysis candidate order must be contiguous starting at 1; "
            f"found {actual_orders}."
        )
    return ordered


def configured_focused_model_comparisons(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return configured focused model-comparison output specifications."""

    analysis = _require_mapping(config.get("analysis"), "analysis_config.analysis")
    values = _require_list(
        analysis.get("focused_model_comparisons", []),
        "analysis.focused_model_comparisons",
    )
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        name = f"analysis.focused_model_comparisons[{index}]"
        item = _require_mapping(value, name)
        _require_keys(item, ["output_stem", "feature_set", "models"], name)
        stem = item["output_stem"]
        feature_set = item["feature_set"]
        models = _require_list(item["models"], f"{name}.models")
        if not isinstance(stem, str) or not stem or "/" in stem or "\\" in stem:
            raise ValueError(f"{name}.output_stem must be a portable file stem.")
        if stem in seen:
            raise ValueError(f"{name}.output_stem duplicates {stem!r}.")
        if not isinstance(feature_set, str) or not feature_set:
            raise ValueError(f"{name}.feature_set must be a non-empty string.")
        if not models or any(not isinstance(model, str) or not model for model in models):
            raise ValueError(f"{name}.models must contain non-empty model IDs.")
        if len(set(models)) != len(models):
            raise ValueError(f"{name}.models contains duplicates.")
        seen.add(stem)
        results.append({"output_stem": stem, "feature_set": feature_set, "models": tuple(models)})
    return tuple(results)


def configured_final_mapping_policy(
    config: Mapping[str, Any],
    experiment_matrix: Mapping[str, Any],
) -> dict[str, Any]:
    """Return validated final-map sites and seed."""

    analysis = _require_mapping(config.get("analysis"), "analysis_config.analysis")
    mapping = _require_mapping(analysis.get("final_mapping"), "analysis.final_mapping")
    _require_keys(mapping, ["sites", "seed"], "analysis.final_mapping")
    seed = mapping["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("analysis.final_mapping.seed must be a non-negative integer.")
    configured_sites = tuple(_require_list(experiment_matrix.get("sites"), "experiment_matrix.sites"))
    site_policy = mapping["sites"]
    if site_policy == "all_configured":
        sites = configured_sites
    else:
        values = _require_list(site_policy, "analysis.final_mapping.sites")
        if not values or any(not isinstance(site, str) or not site for site in values):
            raise ValueError("analysis.final_mapping.sites must contain non-empty site IDs.")
        if len(set(values)) != len(values):
            raise ValueError("analysis.final_mapping.sites contains duplicates.")
        unknown = [site for site in values if site not in configured_sites]
        if unknown:
            raise ValueError(f"analysis.final_mapping.sites references unknown sites: {unknown}.")
        sites = tuple(values)
    return {"sites": sites, "seed": seed}


def validate_candidate_contract(
    rows: Sequence[Mapping[str, Any]],
    analysis_config: Mapping[str, Any],
) -> None:
    """Validate candidate CSV identity and order against analysis policy."""

    configured = configured_analysis_candidates(analysis_config)
    actual = [
        (
            str(row.get("candidate_label", "")),
            str(row.get("feature_set", "")),
            str(row.get("model_id", "")),
        )
        for row in rows
    ]
    expected = [
        (candidate["candidate_label"], candidate["feature_set"], candidate["model_id"])
        for candidate in configured
    ]
    if len({row[0] for row in actual}) != len(actual):
        raise ValueError("Candidate CSV contains duplicate candidate_label values.")
    if actual != expected:
        raise ValueError(
            "Candidate CSV identity/order does not match analysis.yaml: "
            f"expected={expected}, actual={actual}."
        )


def validate_analysis_config(
    config: Mapping[str, Any],
    experiment_matrix: Mapping[str, Any] | None = None,
    *,
    expected_project_id: str | None = None,
) -> None:
    """Validate the explicit Stage 11/12 analysis policy."""

    root = _require_mapping(config, "analysis_config")
    _require_keys(root, ["schema_version", "project_id", "analysis"], "analysis_config")
    if root["schema_version"] != 1:
        raise ValueError("analysis_config.schema_version must be 1.")
    project_id = root["project_id"]
    if not isinstance(project_id, str) or not project_id:
        raise ValueError("analysis_config.project_id must be a non-empty string.")
    if expected_project_id is not None and project_id != expected_project_id:
        raise ValueError(
            f"analysis_config.project_id={project_id!r} does not match {expected_project_id!r}."
        )
    analysis = _require_mapping(root["analysis"], "analysis_config.analysis")
    _require_keys(
        analysis,
        [
            "metric",
            "integration",
            "validation_sources",
            "require_complete_sources",
            "site_rank_comparisons",
            "candidate_selection",
            "final_mapping",
        ],
        "analysis_config.analysis",
    )
    if analysis["metric"] != "AP_mean":
        raise ValueError("analysis.metric must be 'AP_mean' for the current Stage 11 mechanics.")
    integration = _require_mapping(analysis["integration"], "analysis.integration")
    _require_keys(integration, ["method"], "analysis.integration")
    if integration["method"] != "equal_weight_mean":
        raise ValueError("analysis.integration.method must be 'equal_weight_mean'.")
    if not _require_bool(
        analysis["require_complete_sources"],
        "analysis.require_complete_sources",
    ):
        raise ValueError("analysis.require_complete_sources must be true.")
    sources = configured_validation_sources(root)
    if {source["id"] for source in sources} != set(SUPPORTED_ANALYSIS_SOURCES):
        raise ValueError(
            "analysis.validation_sources must contain within_site, cross_site, and "
            "spatial_block_cv for the current Stage 11 output contract."
        )
    if len({source["weight"] for source in sources}) != 1:
        raise ValueError("equal_weight_mean requires equal validation-source weights.")
    comparisons = configured_site_rank_comparisons(root)
    if len(comparisons) != 1:
        raise ValueError(
            "analysis.site_rank_comparisons must contain exactly one comparison for "
            "the current Stage 11 output contract."
        )
    candidates = configured_analysis_candidates(root)
    focused = configured_focused_model_comparisons(root)

    if experiment_matrix is not None:
        sites = set(_require_list(experiment_matrix.get("sites"), "experiment_matrix.sites"))
        features = set(configured_feature_sets(experiment_matrix))
        models = set(_require_list(experiment_matrix.get("models"), "experiment_matrix.models"))
        for comparison in comparisons:
            unknown = [site for site in comparison["sites"] if site not in sites]
            if unknown:
                raise ValueError(f"Site-rank comparison references unknown sites: {unknown}.")
        for candidate in candidates:
            if candidate["feature_set"] not in features:
                raise ValueError(
                    f"Candidate {candidate['candidate_label']!r} references unknown feature set "
                    f"{candidate['feature_set']!r}."
                )
            if candidate["model_id"] not in models:
                raise ValueError(
                    f"Candidate {candidate['candidate_label']!r} references unknown model "
                    f"{candidate['model_id']!r}."
                )
        for item in focused:
            if item["feature_set"] not in features:
                raise ValueError(
                    f"Focused comparison {item['output_stem']!r} references unknown feature set."
                )
            unknown_models = [model for model in item["models"] if model not in models]
            if unknown_models:
                raise ValueError(
                    f"Focused comparison {item['output_stem']!r} references unknown models: "
                    f"{unknown_models}."
                )
        configured_final_mapping_policy(root, experiment_matrix)


def get_site_config(
    sites_config: Mapping[str, Any] | str,
    site_id: str | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the config block for ``site_id`` from ``config_sites.yaml``."""

    if isinstance(sites_config, str) and isinstance(site_id, Mapping):
        sites_config, site_id = site_id, sites_config
    if not isinstance(site_id, str):
        raise ValueError("site_id must be a string.")

    root = _require_mapping(sites_config, "sites_config")
    sites = _require_mapping(root.get("sites"), "sites_config.sites")

    if site_id not in sites:
        available = ", ".join(sorted(str(key) for key in sites))
        raise KeyError(f"Unknown site_id {site_id!r}. Available sites: {available}")

    site_config = sites[site_id]
    _require_mapping(site_config, f"sites.{site_id}")
    return dict(site_config)


def resolve_project_paths(
    project_config: Mapping[str, Any],
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, Path]:
    """Resolve and validate the ``paths`` section of ``config_project.yaml``."""

    root = _require_mapping(project_config, "project_config")
    paths = _require_mapping(root.get("paths"), "project_config.paths")

    required = [
        "framework_root",
        "data_root",
        "project_root",
        "experiment_root",
        "pipeline_root",
        "output_root",
        "docs_root",
    ]
    _require_keys(paths, required, "project_config.paths")

    runtime_binding = _effective_runtime_binding(runtime_binding)
    if runtime_binding is None:
        return {
            key: resolve_repo_path(paths[key], framework_root)
            for key in required
        }
    if framework_root is not None:
        raise ValueError("framework_root and runtime_binding are mutually exclusive authorities.")
    _validate_runtime_binding(runtime_binding)
    if str(paths["data_root"]) != runtime_binding.data_locator:
        raise ValueError("Materialized data locator differs from the runtime data binding.")

    config_root = runtime_binding.config_root
    pipeline_locator = _portable_locator(paths["pipeline_root"], "project_config.paths.pipeline_root")
    project_locator = _portable_locator(paths["project_root"], "project_config.paths.project_root")
    experiment_locator = _portable_locator(
        paths["experiment_root"],
        "project_config.paths.experiment_root",
    )
    return {
        "framework_root": runtime_binding.framework_code_root,
        "data_root": runtime_binding.data_root,
        "project_root": config_root.joinpath(*project_locator.parts).resolve(strict=False),
        "experiment_root": config_root.joinpath(*experiment_locator.parts).resolve(strict=False),
        "pipeline_root": runtime_binding.framework_code_root.joinpath(
            *pipeline_locator.parts
        ).resolve(strict=False),
        "output_root": runtime_binding.output_root,
        "docs_root": runtime_binding.docs_root,
    }


def resolve_site_input_paths(
    site_config: Mapping[str, Any],
    framework_root: str | Path | None = None,
    *,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, Any]:
    """Resolve raster and vector input paths for one site config.

    The returned dictionary has ``bathymetry`` and ``backscatter`` mappings keyed
    by frequency, plus ``label_path`` and ``crop_shp`` entries. Every required
    file must exist.
    """

    site = _require_mapping(site_config, "site_config")
    _require_keys(site, ["paths", "input_files", "frequencies"], "site_config")

    paths = _require_mapping(site["paths"], "site_config.paths")
    input_files = _require_mapping(site["input_files"], "site_config.input_files")
    frequencies = _require_list(site["frequencies"], "site_config.frequencies")

    _require_keys(
        paths,
        ["bathy_dir", "sonar_dir", "label_path", "crop_shp"],
        "site_config.paths",
    )
    _require_keys(
        input_files,
        ["bathymetry", "backscatter"],
        "site_config.input_files",
    )

    bathy_files = _require_mapping(
        input_files["bathymetry"],
        "site_config.input_files.bathymetry",
    )
    backscatter_files = _require_mapping(
        input_files["backscatter"],
        "site_config.input_files.backscatter",
    )

    runtime_binding = _effective_runtime_binding(runtime_binding)
    if runtime_binding is None:
        resolver = lambda value: resolve_repo_path(value, framework_root)
    else:
        if framework_root is not None:
            raise ValueError(
                "framework_root and runtime_binding are mutually exclusive authorities."
            )
        resolver = lambda value: resolve_runtime_data_path(value, runtime_binding)
    bathy_dir = resolver(paths["bathy_dir"])
    sonar_dir = resolver(paths["sonar_dir"])
    label_path = resolver(paths["label_path"])
    crop_shp = resolver(paths["crop_shp"])

    resolved_bathy: dict[int, Path] = {}
    resolved_backscatter: dict[int, Path] = {}

    for frequency in frequencies:
        bathy_name = _lookup_frequency_key(bathy_files, frequency)
        backscatter_name = _lookup_frequency_key(backscatter_files, frequency)

        if bathy_name is None:
            raise ValueError(f"Missing bathymetry filename for frequency {frequency}.")
        if backscatter_name is None:
            raise ValueError(f"Missing backscatter filename for frequency {frequency}.")

        freq_int = int(frequency)
        resolved_bathy[freq_int] = bathy_dir / str(bathy_name)
        resolved_backscatter[freq_int] = sonar_dir / str(backscatter_name)

    resolved = {
        "bathymetry": resolved_bathy,
        "backscatter": resolved_backscatter,
        "label_path": label_path,
        "crop_shp": crop_shp,
    }

    missing = [
        str(path)
        for value in resolved.values()
        for path in (value.values() if isinstance(value, dict) else [value])
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required site input file(s): " + "; ".join(missing))

    return resolved


def ensure_output_dirs(
    project_config: Mapping[str, Any],
    stage_name: str | None = None,
    *,
    framework_root: str | Path | None = None,
    runtime_binding: RuntimePathBinding | None = None,
) -> dict[str, Path]:
    """Create and return standard output directories for a project.

    The returned mapping always exposes canonical paths. When ``stage_name`` is
    supplied, only that stage's owned directory is created; this prevents one
    stage from pre-creating another stage's output contract.
    """

    paths = resolve_project_paths(
        project_config,
        framework_root=framework_root,
        runtime_binding=runtime_binding,
    )
    output_root = paths["output_root"]

    dirs = {
        "output_root": output_root,
        "prepared": output_root / "prepared",
        "common_valid_rasters": output_root / "common_valid_rasters",
        "base_features": output_root / "base_features",
        "feature_sets": output_root / "feature_sets",
        "samples": output_root / "samples",
        "within_site": output_root / "within_site",
        "cross_site": output_root / "cross_site",
        "spatial_block_cv": output_root / "spatial_block_cv",
        "analysis_compare": output_root / "analysis_compare",
        "final_maps": output_root / "final_maps",
        "report_outputs": output_root / "report_outputs",
    }

    output_root.mkdir(parents=True, exist_ok=True)
    stage_owned = {
        "01": ("prepared", "common_valid_rasters"),
        "02": ("base_features",),
        "03": ("feature_sets",),
        "04": ("samples",),
        "05": ("within_site",),
        "06": ("within_site",),
        "07": ("cross_site",),
        "08": ("cross_site",),
        "09": ("spatial_block_cv",),
        "10": ("spatial_block_cv",),
        "11": ("analysis_compare",),
        "12": ("final_maps",),
        "13": ("report_outputs",),
    }
    if stage_name is None:
        names = tuple(key for key in dirs if key != "output_root")
    else:
        normalized = str(stage_name).split("_", 1)[0].zfill(2)
        if normalized not in stage_owned:
            raise ValueError(f"Unknown pipeline stage for output ownership: {stage_name!r}")
        names = stage_owned[normalized]
    for name in names:
        dirs[name].mkdir(parents=True, exist_ok=True)

    return dirs


def configured_model_persistence(
    project_config: Mapping[str, Any],
) -> dict[str, bool]:
    """Resolve validation and final-model persistence from project outputs.

    Historical configurations that predate the split policy inherit
    ``outputs.save_models`` (or ``True`` when that legacy key is also absent).
    """

    root = _require_mapping(project_config, "project_config")
    outputs = _require_mapping(root.get("outputs"), "project_config.outputs")
    legacy_default = outputs.get("save_models", True)
    legacy_default = _require_bool(legacy_default, "outputs.save_models")

    resolved: dict[str, bool] = {}
    for key in ("save_validation_models", "save_final_models"):
        value = outputs.get(key, legacy_default)
        resolved[key] = _require_bool(value, f"outputs.{key}")
    return resolved


def validate_project_config(config: Mapping[str, Any]) -> None:
    """Validate the current ``config_project.yaml`` structure."""

    root = _require_mapping(config, "project_config")
    _require_keys(
        root,
        [
            "project",
            "paths",
            "spatial",
            "analysis",
            "base_features",
            "sampling",
            "validation",
            "evaluation_metrics",
            "ensemble",
            "outputs",
        ],
        "project_config",
    )

    project = _require_mapping(root["project"], "project_config.project")
    _require_keys(project, ["id", "type", "status", "title"], "project_config.project")

    paths = _require_mapping(root["paths"], "project_config.paths")
    _require_keys(
        paths,
        [
            "framework_root",
            "data_root",
            "project_root",
            "experiment_root",
            "pipeline_root",
            "output_root",
            "docs_root",
        ],
        "project_config.paths",
    )

    spatial = _require_mapping(root["spatial"], "project_config.spatial")
    _require_keys(spatial, ["crs_out", "target_resolution_m"], "project_config.spatial")
    if not isinstance(spatial["crs_out"], str):
        raise ValueError("project_config.spatial.crs_out must be a string.")
    if _require_numeric(spatial["target_resolution_m"], "spatial.target_resolution_m") <= 0:
        raise ValueError("project_config.spatial.target_resolution_m must be > 0.")

    analysis = _require_mapping(root["analysis"], "project_config.analysis")
    _require_keys(
        analysis,
        ["task_type", "positive_class_name", "negative_class_name"],
        "project_config.analysis",
    )
    if analysis["task_type"] != "binary_classification":
        raise ValueError("Only analysis.task_type='binary_classification' is supported.")

    base_features = _require_mapping(root["base_features"], "project_config.base_features")
    _validate_base_features_config(base_features)

    sampling = _require_mapping(root["sampling"], "project_config.sampling")
    _require_keys(
        sampling,
        ["random_seed", "n_neg_per_pos", "positive_sampling", "negative_sampling"],
        "project_config.sampling",
    )

    validation = _require_mapping(root["validation"], "project_config.validation")
    _require_keys(validation, ["within_site", "cross_site"], "project_config.validation")
    _require_bool(validation["within_site"], "validation.within_site")
    _require_bool(validation["cross_site"], "validation.cross_site")

    metrics = _require_mapping(root["evaluation_metrics"], "project_config.evaluation_metrics")
    _require_keys(metrics, ["primary_metric", "report_metrics"], "evaluation_metrics")
    _require_list(metrics["report_metrics"], "evaluation_metrics.report_metrics")

    ensemble = _require_mapping(root["ensemble"], "project_config.ensemble")
    _require_keys(ensemble, ["enabled", "seeds"], "project_config.ensemble")
    _require_bool(ensemble["enabled"], "ensemble.enabled")
    _require_list(ensemble["seeds"], "ensemble.seeds")

    configured_model_persistence(root)


def validate_sites_config(config: Mapping[str, Any]) -> None:
    """Validate the current ``config_sites.yaml`` structure."""

    root = _require_mapping(config, "sites_config")
    sites = _require_mapping(root.get("sites"), "sites_config.sites")
    if not sites:
        raise ValueError("sites_config.sites must contain at least one site.")

    for site_id, site_config in sites.items():
        site = _require_mapping(site_config, f"sites.{site_id}")
        _require_keys(
            site,
            ["site_name", "short_name", "frequencies", "paths", "input_files", "label"],
            f"sites.{site_id}",
        )

        frequencies = _require_list(site["frequencies"], f"sites.{site_id}.frequencies")
        if not frequencies:
            raise ValueError(f"sites.{site_id}.frequencies must not be empty.")
        configured_frequencies(site)

        paths = _require_mapping(site["paths"], f"sites.{site_id}.paths")
        _require_keys(
            paths,
            ["site_root", "bathy_dir", "sonar_dir", "label_dir", "crop_dir", "label_path", "crop_shp"],
            f"sites.{site_id}.paths",
        )

        input_files = _require_mapping(site["input_files"], f"sites.{site_id}.input_files")
        _require_keys(
            input_files,
            ["bathymetry", "backscatter"],
            f"sites.{site_id}.input_files",
        )

        bathymetry = _require_mapping(
            input_files["bathymetry"],
            f"sites.{site_id}.input_files.bathymetry",
        )
        backscatter = _require_mapping(
            input_files["backscatter"],
            f"sites.{site_id}.input_files.backscatter",
        )
        for frequency in frequencies:
            if _lookup_frequency_key(bathymetry, frequency) is None:
                raise ValueError(f"sites.{site_id} missing bathymetry file for {frequency}.")
            if _lookup_frequency_key(backscatter, frequency) is None:
                raise ValueError(f"sites.{site_id} missing backscatter file for {frequency}.")

        label = _require_mapping(site["label"], f"sites.{site_id}.label")
        _require_keys(
            label,
            ["format", "positive_field", "positive_value", "negative_value"],
            f"sites.{site_id}.label",
        )
        if "buffer_m" in label:
            buffer_m = _require_numeric(label["buffer_m"], f"sites.{site_id}.label.buffer_m")
            if float(buffer_m) < 0:
                raise ValueError(f"sites.{site_id}.label.buffer_m must be >= 0.")
        if "all_touched" in label:
            _require_bool(label["all_touched"], f"sites.{site_id}.label.all_touched")
        clip_to_valid_feature_area = label.get("clip_to_valid_feature_area", False)
        if "clip_to_valid_feature_area" in label:
            clip_to_valid_feature_area = _require_bool(
                label["clip_to_valid_feature_area"],
                f"sites.{site_id}.label.clip_to_valid_feature_area",
            )
        clip_mode = label.get("clip_mode", "none")
        if "clip_mode" in label and not isinstance(clip_mode, str):
            raise ValueError(f"sites.{site_id}.label.clip_mode must be a string.")
        if clip_to_valid_feature_area and clip_mode != "finite_mask":
            raise ValueError(
                f"sites.{site_id}.label.clip_mode must be 'finite_mask' when "
                "clip_to_valid_feature_area is true."
            )


def configured_cross_site_pairs(
    experiment_matrix: Mapping[str, Any],
    *,
    train_site: str | None = None,
    test_site: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return validated cross-site pairs in configuration order.

    Pair direction and inventory come only from
    ``validation.cross_site.pairs``. Optional train/test filters select from
    that inventory; they never create reverse or otherwise unconfigured pairs.
    """

    root = _require_mapping(experiment_matrix, "experiment_matrix_config")
    site_values = _require_list(root.get("sites"), "sites")
    if any(not isinstance(site_id, str) or not site_id for site_id in site_values):
        raise ValueError("sites must contain only non-empty strings.")
    site_ids = tuple(site_values)
    known_sites = set(site_ids)

    validation = _require_mapping(root.get("validation"), "validation")
    cross_site = _require_mapping(validation.get("cross_site"), "validation.cross_site")
    enabled = _require_bool(cross_site.get("enable"), "validation.cross_site.enable")
    if not enabled:
        return ()

    pair_values = _require_list(cross_site.get("pairs"), "validation.cross_site.pairs")
    if not pair_values:
        raise ValueError(
            "validation.cross_site.pairs must not be empty when cross-site validation is enabled."
        )

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(pair_values):
        name = f"validation.cross_site.pairs[{index}]"
        pair = _require_mapping(value, name)
        _require_keys(pair, ["train", "test"], name)
        train = pair["train"]
        test = pair["test"]
        if not isinstance(train, str) or not train:
            raise ValueError(f"{name}.train must be a non-empty string.")
        if not isinstance(test, str) or not test:
            raise ValueError(f"{name}.test must be a non-empty string.")
        if train not in known_sites:
            raise ValueError(f"{name}.train references unknown site {train!r}.")
        if test not in known_sites:
            raise ValueError(f"{name}.test references unknown site {test!r}.")
        if train == test:
            raise ValueError(f"{name} is a self transfer ({train!r} -> {test!r}).")
        pair_key = (train, test)
        if pair_key in seen:
            raise ValueError(f"{name} duplicates configured pair {train!r} -> {test!r}.")
        seen.add(pair_key)
        pairs.append(pair_key)

    for value, option_name in ((train_site, "train_site"), (test_site, "test_site")):
        if value is not None and value not in known_sites:
            raise ValueError(
                f"{option_name}={value!r} is not listed in experiment_matrix.sites. "
                f"Available values: {', '.join(site_ids)}"
            )

    selected = tuple(
        pair
        for pair in pairs
        if (train_site is None or pair[0] == train_site)
        and (test_site is None or pair[1] == test_site)
    )
    if (train_site is not None or test_site is not None) and not selected:
        raise ValueError(
            "No configured cross-site pair matches the requested filters: "
            f"train_site={train_site!r}, test_site={test_site!r}."
        )
    return selected


def cross_site_expected_run_count(experiment_matrix: Mapping[str, Any]) -> int:
    """Return the configured pair x feature x model x seed run count."""

    pairs = configured_cross_site_pairs(experiment_matrix)
    feature_sets = _require_list(experiment_matrix.get("feature_sets"), "feature_sets")
    models = _require_list(experiment_matrix.get("models"), "models")
    seeds = _require_list(experiment_matrix.get("seeds"), "seeds")
    return len(pairs) * len(feature_sets) * len(models) * len(seeds)


def configured_feature_sets(
    experiment_matrix: Mapping[str, Any],
    *,
    feature_set: str | None = None,
    available_feature_set_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the authoritative feature-set inventory in matrix order.

    An optional filter selects one configured feature set without creating a
    new active ID. When available definitions are supplied, every configured
    ID must resolve to one of them.
    """

    root = _require_mapping(experiment_matrix, "experiment_matrix_config")
    values = _require_list(root.get("feature_sets"), "feature_sets")
    if not values:
        raise ValueError("feature_sets must not be empty.")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("feature_sets must contain only non-empty strings.")

    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"feature_sets contains duplicate values: {duplicates}.")

    configured = tuple(values)
    if available_feature_set_ids is not None:
        available = set(available_feature_set_ids)
        unknown = [value for value in configured if value not in available]
        if unknown:
            raise ValueError(
                f"feature_sets references undefined feature-set IDs: {unknown}."
            )

    if feature_set is None:
        return configured
    if feature_set not in configured:
        raise ValueError(
            f"feature_set={feature_set!r} is not listed in experiment_matrix.feature_sets. "
            f"Available values: {', '.join(configured)}"
        )
    return (feature_set,)


def spatial_expected_fold_run_count(experiment_matrix: Mapping[str, Any]) -> int:
    """Return site x feature x model x seed x fold spatial run count."""

    feature_sets = configured_feature_sets(experiment_matrix)
    sites = _require_list(experiment_matrix.get("sites"), "sites")
    seeds = _require_list(experiment_matrix.get("seeds"), "seeds")
    validation = _require_mapping(experiment_matrix.get("validation"), "validation")
    spatial = _require_mapping(
        validation.get("spatial_block_cv"),
        "validation.spatial_block_cv",
    )
    if not _require_bool(spatial.get("enable"), "validation.spatial_block_cv.enable"):
        return 0
    stage1 = _require_mapping(spatial.get("stage1"), "validation.spatial_block_cv.stage1")
    models = _require_list(stage1.get("models"), "validation.spatial_block_cv.stage1.models")
    n_splits = spatial.get("n_splits")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("validation.spatial_block_cv.n_splits must be an integer >= 2.")
    return len(sites) * len(feature_sets) * len(models) * len(seeds) * n_splits


def expand_spatial_fold_run_matrix(
    experiment_matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand configured spatial fold runs without reading or writing outputs."""

    validate_experiment_matrix(experiment_matrix)
    spatial = experiment_matrix["validation"]["spatial_block_cv"]
    if not spatial.get("enable"):
        return []
    models = spatial["stage1"]["models"]
    return [
        {
            "validation_type": "spatial_block_cv",
            "site_id": site_id,
            "feature_set": feature_set_id,
            "model_id": model_id,
            "seed": seed,
            "fold_id": fold_id,
        }
        for site_id, feature_set_id, model_id, seed, fold_id in product(
            experiment_matrix["sites"],
            configured_feature_sets(experiment_matrix),
            models,
            experiment_matrix["seeds"],
            range(int(spatial["n_splits"])),
        )
    ]


def validate_experiment_matrix(config: Mapping[str, Any]) -> None:
    """Validate the current ``experiment_matrix.yaml`` structure."""

    root = _require_mapping(config, "experiment_matrix_config")
    _require_keys(
        root,
        ["experiment_matrix", "sites", "feature_sets", "models", "seeds", "validation", "run_plan"],
        "experiment_matrix_config",
    )

    matrix = _require_mapping(root["experiment_matrix"], "experiment_matrix")
    _require_keys(matrix, ["project_id"], "experiment_matrix")

    for list_name in ["sites", "models", "seeds"]:
        values = _require_list(root[list_name], list_name)
        if not values:
            raise ValueError(f"{list_name} must not be empty.")
    configured_feature_sets(root)

    validation = _require_mapping(root["validation"], "validation")
    _require_keys(validation, ["within_site", "cross_site", "spatial_block_cv"], "validation")

    within_site = _require_mapping(validation["within_site"], "validation.within_site")
    _require_keys(within_site, ["enable"], "validation.within_site")
    _require_bool(within_site["enable"], "validation.within_site.enable")
    if within_site["enable"]:
        _require_keys(
            within_site,
            ["split_method", "train_fraction", "test_fraction", "stratify_by_label"],
            "validation.within_site",
        )
        train_fraction = _require_numeric(
            within_site["train_fraction"],
            "validation.within_site.train_fraction",
        )
        test_fraction = _require_numeric(
            within_site["test_fraction"],
            "validation.within_site.test_fraction",
        )
        if abs(float(train_fraction) + float(test_fraction) - 1.0) > 1e-6:
            raise ValueError("within_site train_fraction + test_fraction must equal 1.0.")

    cross_site = _require_mapping(validation["cross_site"], "validation.cross_site")
    _require_keys(cross_site, ["enable"], "validation.cross_site")
    _require_bool(cross_site["enable"], "validation.cross_site.enable")
    if cross_site["enable"]:
        configured_cross_site_pairs(root)

    spatial_block_cv = _require_mapping(
        validation["spatial_block_cv"],
        "validation.spatial_block_cv",
    )
    _require_keys(spatial_block_cv, ["enable"], "validation.spatial_block_cv")
    _require_bool(spatial_block_cv["enable"], "validation.spatial_block_cv.enable")
    if spatial_block_cv["enable"]:
        _require_keys(
            spatial_block_cv,
            [
                "split_method",
                "n_splits",
                "block_size_m",
                "origin",
                "shuffle_blocks",
                "min_positive_per_fold",
                "min_negative_per_fold",
                "stage1",
            ],
            "validation.spatial_block_cv",
        )
        if spatial_block_cv["split_method"] != "stratified_group_kfold":
            raise ValueError(
                "validation.spatial_block_cv.split_method must be "
                "'stratified_group_kfold'."
            )
        n_splits = spatial_block_cv["n_splits"]
        if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("validation.spatial_block_cv.n_splits must be an integer >= 2.")
        block_size_m = _require_numeric(
            spatial_block_cv["block_size_m"],
            "validation.spatial_block_cv.block_size_m",
        )
        if float(block_size_m) <= 0:
            raise ValueError("validation.spatial_block_cv.block_size_m must be > 0.")
        if not isinstance(spatial_block_cv["origin"], str):
            raise ValueError("validation.spatial_block_cv.origin must be a string.")
        _require_bool(
            spatial_block_cv["shuffle_blocks"],
            "validation.spatial_block_cv.shuffle_blocks",
        )
        min_positive = spatial_block_cv["min_positive_per_fold"]
        if isinstance(min_positive, bool) or not isinstance(min_positive, int) or min_positive < 1:
            raise ValueError(
                "validation.spatial_block_cv.min_positive_per_fold must be an integer >= 1."
            )
        min_negative = spatial_block_cv["min_negative_per_fold"]
        if isinstance(min_negative, bool) or not isinstance(min_negative, int) or min_negative < 1:
            raise ValueError(
                "validation.spatial_block_cv.min_negative_per_fold must be an integer >= 1."
            )

        stage1 = _require_mapping(
            spatial_block_cv["stage1"],
            "validation.spatial_block_cv.stage1",
        )
        for list_name in ["models"]:
            values = _require_list(
                stage1.get(list_name),
                f"validation.spatial_block_cv.stage1.{list_name}",
            )
            if not values:
                raise ValueError(
                    f"validation.spatial_block_cv.stage1.{list_name} must not be empty."
                )
            if any(not isinstance(value, str) for value in values):
                raise ValueError(
                    f"validation.spatial_block_cv.stage1.{list_name} must contain only strings."
                )

    run_plan = _require_mapping(root["run_plan"], "run_plan")
    _require_keys(
        run_plan,
        ["run_all_feature_sets", "run_all_models", "run_all_seeds"],
        "run_plan",
    )


def expand_run_matrix(experiment_matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the experiment matrix into within-site and cross-site run records."""

    validate_experiment_matrix(experiment_matrix)

    sites = list(experiment_matrix["sites"])
    feature_sets = list(configured_feature_sets(experiment_matrix))
    models = list(experiment_matrix["models"])
    seeds = list(experiment_matrix["seeds"])
    validation = experiment_matrix["validation"]

    runs: list[dict[str, Any]] = []

    within_site = validation["within_site"]
    if within_site.get("enable"):
        for site_id, feature_set_id, model_id, seed in product(sites, feature_sets, models, seeds):
            runs.append(
                {
                    "validation_type": "within_site",
                    "train_site": site_id,
                    "test_site": site_id,
                    "feature_set": feature_set_id,
                    "model": model_id,
                    "seed": seed,
                    "split_method": within_site.get("split_method"),
                }
            )

    cross_site = validation["cross_site"]
    if cross_site.get("enable"):
        for pair, feature_set_id, model_id, seed in product(
            configured_cross_site_pairs(experiment_matrix),
            feature_sets,
            models,
            seeds,
        ):
            runs.append(
                {
                    "validation_type": "cross_site",
                    "train_site": pair[0],
                    "test_site": pair[1],
                    "feature_set": feature_set_id,
                    "model": model_id,
                    "seed": seed,
                    "split_method": None,
                }
            )

    expected_total = experiment_matrix.get("run_plan", {}).get("total_expected_runs")
    if expected_total is not None and len(runs) != int(expected_total):
        raise ValueError(
            f"Expanded run count ({len(runs)}) does not match "
            f"run_plan.total_expected_runs ({expected_total})."
        )

    return runs


def get_git_metadata(repo_root: str | Path | None = None) -> dict[str, str | bool | None]:
    """Return git branch, commit hash, tag, and dirty-state metadata."""

    root = _project_root(repo_root)

    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return result.stdout.strip() or None

    status = run_git(["status", "--porcelain"])
    return {
        "git_hash": run_git(["rev-parse", "HEAD"]),
        "git_short_hash": run_git(["rev-parse", "--short", "HEAD"]),
        "git_branch": run_git(["branch", "--show-current"]),
        "git_tag": run_git(["describe", "--tags", "--exact-match"]),
        "git_dirty": bool(status),
    }


def _validate_feature_set_config(config: Mapping[str, Any], expected_id: str | None = None) -> None:
    """Validate a feature-set YAML file."""

    root = _require_mapping(config, "feature_set_config")
    _require_keys(root, ["feature_set", "features"], "feature_set_config")

    feature_set = _require_mapping(root["feature_set"], "feature_set")
    _require_keys(
        feature_set,
        ["id", "type", "frequencies", "reference_frequency", "feature_count"],
        "feature_set",
    )
    if expected_id is not None and feature_set["id"] != expected_id:
        raise ValueError(
            f"Feature-set id mismatch: expected {expected_id!r}, "
            f"found {feature_set['id']!r}."
        )

    features = _require_mapping(root["features"], "features")
    _require_keys(
        features,
        ["acoustic", "frequency_difference", "terrain", "texture"],
        "features",
    )
    flattened = []
    for group_name in ["acoustic", "frequency_difference", "terrain", "texture"]:
        flattened.extend(_require_list(features[group_name], f"features.{group_name}"))

    if len(flattened) != int(feature_set["feature_count"]):
        raise ValueError(
            f"feature_set.feature_count={feature_set['feature_count']} does not match "
            f"listed feature count={len(flattened)}."
        )


def _validate_model_config(config: Mapping[str, Any], expected_id: str | None = None) -> None:
    """Validate a model YAML file."""

    root = _require_mapping(config, "model_config")
    _require_keys(root, ["model", "params"], "model_config")

    model = _require_mapping(root["model"], "model")
    _require_keys(model, ["id", "display_name", "type", "task"], "model")
    if expected_id is not None and model["id"] != expected_id:
        raise ValueError(
            f"Model id mismatch: expected {expected_id!r}, found {model['id']!r}."
        )
    if model["task"] != "binary_classification":
        raise ValueError("Only model.task='binary_classification' is supported.")

    _require_mapping(root["params"], "params")


def _validate_base_features_config(config: Mapping[str, Any]) -> None:
    """Validate base feature generation parameters."""

    _require_keys(
        config,
        ["ndi_pairs", "acoustic_normalization", "depth_lite", "relative_depth", "texture"],
        "project_config.base_features",
    )

    acoustic = _require_mapping(
        config["acoustic_normalization"],
        "base_features.acoustic_normalization",
    )
    _require_keys(
        acoustic,
        ["method", "depth_bins_q", "min_per_bin", "eps"],
        "base_features.acoustic_normalization",
    )
    if acoustic["method"] != "depth_bin_robust":
        raise ValueError("base_features.acoustic_normalization.method must be 'depth_bin_robust'.")
    if _require_numeric(acoustic["depth_bins_q"], "base_features.acoustic_normalization.depth_bins_q") <= 0:
        raise ValueError("base_features.acoustic_normalization.depth_bins_q must be > 0.")
    if _require_numeric(acoustic["min_per_bin"], "base_features.acoustic_normalization.min_per_bin") <= 0:
        raise ValueError("base_features.acoustic_normalization.min_per_bin must be > 0.")
    if _require_numeric(acoustic["eps"], "base_features.acoustic_normalization.eps") <= 0:
        raise ValueError("base_features.acoustic_normalization.eps must be > 0.")

    depth_lite = _require_mapping(config["depth_lite"], "base_features.depth_lite")
    _require_keys(depth_lite, ["method", "clip", "eps"], "base_features.depth_lite")
    if depth_lite["method"] != "robust_median_p5_p95":
        raise ValueError("base_features.depth_lite.method must be 'robust_median_p5_p95'.")
    if _require_numeric(depth_lite["clip"], "base_features.depth_lite.clip") <= 0:
        raise ValueError("base_features.depth_lite.clip must be > 0.")
    if _require_numeric(depth_lite["eps"], "base_features.depth_lite.eps") <= 0:
        raise ValueError("base_features.depth_lite.eps must be > 0.")

    relative_depth = _require_mapping(
        config["relative_depth"],
        "base_features.relative_depth",
    )
    _require_keys(
        relative_depth,
        ["method", "coarse_res_m", "radius_m", "iqr_eps", "clip"],
        "base_features.relative_depth",
    )
    if relative_depth["method"] != "local_median_iqr":
        raise ValueError("base_features.relative_depth.method must be 'local_median_iqr'.")
    if _require_numeric(relative_depth["coarse_res_m"], "base_features.relative_depth.coarse_res_m") <= 0:
        raise ValueError("base_features.relative_depth.coarse_res_m must be > 0.")
    if _require_numeric(relative_depth["radius_m"], "base_features.relative_depth.radius_m") <= 0:
        raise ValueError("base_features.relative_depth.radius_m must be > 0.")
    if _require_numeric(relative_depth["iqr_eps"], "base_features.relative_depth.iqr_eps") <= 0:
        raise ValueError("base_features.relative_depth.iqr_eps must be > 0.")
    if _require_numeric(relative_depth["clip"], "base_features.relative_depth.clip") <= 0:
        raise ValueError("base_features.relative_depth.clip must be > 0.")

    texture = _require_mapping(config["texture"], "base_features.texture")
    _require_keys(
        texture,
        ["method", "gauss_sigma_px", "log_sigma_px", "use_log1p", "clip"],
        "base_features.texture",
    )
    if texture["method"] != "gaussian_low_high_log":
        raise ValueError("base_features.texture.method must be 'gaussian_low_high_log'.")
    if _require_numeric(texture["gauss_sigma_px"], "base_features.texture.gauss_sigma_px") <= 0:
        raise ValueError("base_features.texture.gauss_sigma_px must be > 0.")
    if _require_numeric(texture["log_sigma_px"], "base_features.texture.log_sigma_px") <= 0:
        raise ValueError("base_features.texture.log_sigma_px must be > 0.")
    _require_bool(texture["use_log1p"], "base_features.texture.use_log1p")
    if _require_numeric(texture["clip"], "base_features.texture.clip") <= 0:
        raise ValueError("base_features.texture.clip must be > 0.")

    configured_ndi_pairs(config)


def configured_frequencies(site_config: Mapping[str, Any]) -> tuple[int, ...]:
    """Return one site's unique positive integer frequencies in config order."""

    site = _require_mapping(site_config, "site_config")
    values = _require_list(site.get("frequencies"), "site_config.frequencies")
    if not values:
        raise ValueError("site_config.frequencies must not be empty.")

    frequencies: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"site_config.frequencies[{index}] must be a positive integer, got {value!r}."
            )
        frequencies.append(value)
    duplicates = sorted({value for value in frequencies if frequencies.count(value) > 1})
    if duplicates:
        raise ValueError(f"site_config.frequencies contains duplicate values: {duplicates}.")
    return tuple(frequencies)


def configured_ndi_pairs(base_features_config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return explicitly configured NDI definitions in their declared order."""

    config = _require_mapping(base_features_config, "base_features")
    pairs = _require_list(config.get("ndi_pairs"), "base_features.ndi_pairs")
    normalized: list[dict[str, Any]] = []
    names: list[str] = []
    frequency_pairs: list[tuple[int, int]] = []
    for index, value in enumerate(pairs):
        pair = _require_mapping(value, f"base_features.ndi_pairs[{index}]")
        _require_keys(
            pair,
            ["feature_name", "high_frequency", "low_frequency"],
            f"base_features.ndi_pairs[{index}]",
        )
        feature_name = pair["feature_name"]
        high = pair["high_frequency"]
        low = pair["low_frequency"]
        if not isinstance(feature_name, str) or not feature_name.strip():
            raise ValueError(
                f"base_features.ndi_pairs[{index}].feature_name must be a non-empty string."
            )
        for key, frequency in (("high_frequency", high), ("low_frequency", low)):
            if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency <= 0:
                raise ValueError(
                    f"base_features.ndi_pairs[{index}].{key} must be a positive integer."
                )
        if high == low:
            raise ValueError(
                f"base_features.ndi_pairs[{index}] cannot use the same frequency twice: {high}."
            )
        names.append(feature_name)
        frequency_pairs.append((high, low))
        normalized.append(
            {
                "feature_name": feature_name,
                "high_frequency": high,
                "low_frequency": low,
            }
        )

    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    duplicate_pairs = sorted({pair for pair in frequency_pairs if frequency_pairs.count(pair) > 1})
    if duplicate_names:
        raise ValueError(f"base_features.ndi_pairs has duplicate feature names: {duplicate_names}.")
    if duplicate_pairs:
        raise ValueError(f"base_features.ndi_pairs has duplicate frequency pairs: {duplicate_pairs}.")
    return tuple(normalized)


def generated_base_feature_names(
    frequencies: Sequence[int],
    ndi_pairs: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return Stage 02's deterministic feature-name inventory."""

    names: list[str] = []
    names.extend(f"a{frequency}_norm" for frequency in frequencies)
    names.extend(str(pair["feature_name"]) for pair in ndi_pairs)
    names.extend(f"dpth_lite_{frequency}" for frequency in frequencies)
    names.extend(f"dpth_rel_{frequency}" for frequency in frequencies)
    for frequency in frequencies:
        names.extend(
            (f"a{frequency}_low", f"a{frequency}_high", f"a{frequency}_log1p")
        )
    return tuple(names)


def _lookup_frequency_key(
    mapping: Mapping[Any, Any],
    frequency: Any,
) -> Any | None:
    """Return a frequency-keyed value while allowing int or string YAML keys."""

    if frequency in mapping:
        return mapping[frequency]
    frequency_str = str(frequency)
    if frequency_str in mapping:
        return mapping[frequency_str]
    try:
        frequency_int = int(frequency)
    except (TypeError, ValueError):
        frequency_int = None
    if frequency_int is not None and frequency_int in mapping:
        return mapping[frequency_int]
    return None

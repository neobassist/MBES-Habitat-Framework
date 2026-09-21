#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model construction and fitted-model helpers for classification stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Validate that ``value`` is a mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _require_joblib() -> Any:
    """Import joblib with a clear dependency error."""

    try:
        import joblib  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "joblib is required for model serialization. Install joblib in the "
            "mbes_seaweed environment before running this stage."
        ) from exc
    return joblib


def model_parts(model_id: str, model_config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return validated model metadata and params."""

    root = _require_mapping(model_config, "model_config")
    model = dict(_require_mapping(root.get("model"), "model"))
    params = dict(_require_mapping(root.get("params"), "params"))

    if model.get("id") != model_id:
        raise ValueError(f"Model id mismatch: expected {model_id!r}, found {model.get('id')!r}.")
    if model.get("task") != "binary_classification":
        raise ValueError(f"model.task must be 'binary_classification' for model_id={model_id}.")
    if "type" not in model:
        raise ValueError(f"model.type is required for model_id={model_id}.")

    return model, params


def make_model(model_config: Mapping[str, Any], seed: int) -> tuple[Any, dict[str, Any]]:
    """Instantiate a supported binary classifier and return effective params."""

    model_id = str(_require_mapping(model_config.get("model"), "model").get("id"))
    model, params = model_parts(model_id, model_config)
    model_type = str(model["type"])
    effective_params = dict(params)

    if model_type == "sklearn_random_forest":
        try:
            from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"scikit-learn is required for model_id={model_id} "
                f"(model.type={model_type})."
            ) from exc
        effective_params["random_state"] = int(seed)
        return RandomForestClassifier(**effective_params), effective_params

    if model_type == "xgboost_classifier":
        try:
            from xgboost import XGBClassifier  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"xgboost is required for model_id={model_id} "
                f"(model.type={model_type})."
            ) from exc
        effective_params["random_state"] = int(seed)
        return XGBClassifier(**effective_params), effective_params

    if model_type == "lightgbm_classifier":
        try:
            from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"lightgbm is required for model_id={model_id} "
                f"(model.type={model_type})."
            ) from exc
        effective_params["random_state"] = int(seed)
        return LGBMClassifier(**effective_params), effective_params

    if model_type == "catboost_classifier":
        try:
            from catboost import CatBoostClassifier  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"catboost is required for model_id={model_id} "
                f"(model.type={model_type})."
            ) from exc
        effective_params.setdefault("verbose", False)
        effective_params.setdefault("allow_writing_files", False)
        effective_params["random_seed"] = int(seed)
        return CatBoostClassifier(**effective_params), effective_params

    raise ValueError(f"Unsupported model.type={model_type!r} for model_id={model_id}.")


def build_model(model_config: Mapping[str, Any], seed: int) -> tuple[Any, dict[str, Any]]:
    """Alias for ``make_model`` for training-stage readability."""

    return make_model(model_config=model_config, seed=seed)


def predict_positive_scores(model: Any, x_test: Any, *, model_id: str) -> np.ndarray:
    """Return positive-class probabilities from a fitted classifier."""

    if not hasattr(model, "predict_proba"):
        raise ValueError(f"Model does not support predict_proba: model_id={model_id}")

    probabilities = model.predict_proba(x_test)
    if probabilities.ndim != 2:
        raise ValueError(f"predict_proba returned non-2D output for model_id={model_id}.")

    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError(f"Fitted model has no classes_ attribute: model_id={model_id}")
    positive_matches = np.where(np.asarray(classes) == 1)[0]
    if positive_matches.size != 1:
        raise ValueError(f"Unable to locate positive class 1 in model.classes_: {classes}")

    return np.asarray(probabilities[:, int(positive_matches[0])], dtype="float64")


def extract_feature_importance(
    model: Any,
    *,
    model_id: str,
    model_type: str,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, str]:
    """Return feature importances and their source label."""

    if model_type == "catboost_classifier":
        if not hasattr(model, "get_feature_importance"):
            raise ValueError(f"CatBoost model has no get_feature_importance method: {model_id}")
        importance = np.asarray(model.get_feature_importance(), dtype="float64")
        importance_type = "get_feature_importance"
    else:
        if not hasattr(model, "feature_importances_"):
            raise ValueError(f"Model has no feature_importances_ attribute: {model_id}")
        importance = np.asarray(model.feature_importances_, dtype="float64")
        importance_type = "feature_importances_"

    if importance.size != len(feature_columns):
        raise ValueError(
            f"Feature importance length mismatch for model_id={model_id}: "
            f"importance={importance.size}, features={len(feature_columns)}"
        )
    if not np.all(np.isfinite(importance)):
        raise ValueError(f"Feature importance contains NaN or infinite values: model_id={model_id}")

    return importance, importance_type


def save_model(model: Any, path: str | Path) -> None:
    """Save a fitted model with joblib."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _require_joblib().dump(model, output_path)


def load_model(path: str | Path) -> Any:
    """Load a fitted model with joblib."""

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"model.joblib not found: {model_path}")
    if not model_path.is_file():
        raise ValueError(f"model path is not a file: {model_path}")
    return _require_joblib().load(model_path)

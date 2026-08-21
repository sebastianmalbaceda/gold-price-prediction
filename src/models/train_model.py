"""Entrenamiento y persistencia de modelos (fases 11-16)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.config import get_config, path_from_root


def _validate_feature_list(feature_list: list[str]) -> None:
    if not feature_list or any(not isinstance(c, str) or not c for c in feature_list):
        raise ValueError("feature_list debe contener al menos una columna valida")
    if len(set(feature_list)) != len(feature_list):
        raise ValueError("feature_list contiene columnas duplicadas")


def _atomic_joblib(value, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        joblib.dump(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(content: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_model_artifacts(
    model,
    preprocessor,
    feature_list: list[str],
    metrics: dict | None = None,
    cfg: dict | None = None,
) -> dict:
    """Guarda modelo, preprocesador, features y metricas."""
    if model is None or preprocessor is None:
        raise ValueError("model y preprocessor son obligatorios")
    _validate_feature_list(feature_list)
    cfg = cfg or get_config()
    m = cfg["model"]
    paths = {
        "model": path_from_root(m["final_model_path"]),
        "preprocessor": path_from_root(m["preprocessor_path"]),
        "features": path_from_root(m["feature_list_path"]),
        "metrics": path_from_root(m["metrics_path"]),
    }
    for p in set(paths.values()):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    _atomic_joblib(model, paths["model"])
    _atomic_joblib(preprocessor, paths["preprocessor"])
    _atomic_text(json.dumps(feature_list, indent=2, ensure_ascii=False), paths["features"])
    if metrics is not None:
        _atomic_text(
            json.dumps(metrics, indent=2, default=str, ensure_ascii=False), paths["metrics"]
        )
    print(f"[save] modelo -> {paths['model']}")
    print(f"[save] preprocesador -> {paths['preprocessor']}")
    print(f"[save] features ({len(feature_list)}) -> {paths['features']}")
    return {k: str(v) for k, v in paths.items()}


def load_model_artifacts(cfg: dict | None = None) -> tuple:
    """Carga modelo + preprocesador + features guardados.

    Los ficheros joblib deben proceder de una fuente confiable: cargarlos
    implica deserializacion de Python y no es seguro hacerlo con artefactos
    recibidos de terceros.
    """
    cfg = cfg or get_config()
    m = cfg["model"]
    model_path = path_from_root(m["final_model_path"])
    preprocessor_path = path_from_root(m["preprocessor_path"])
    features_path = path_from_root(m["feature_list_path"])
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    features = json.loads(features_path.read_text(encoding="utf-8"))
    _validate_feature_list(features)
    return model, preprocessor, features

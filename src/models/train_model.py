"""Entrenamiento y persistencia de modelos (fases 11-16)."""

from __future__ import annotations

import json

import joblib

from src.config import get_config, path_from_root


def save_model_artifacts(
    model,
    preprocessor,
    feature_list: list[str],
    metrics: dict | None = None,
    cfg: dict | None = None,
) -> dict:
    """Guarda modelo, preprocesador, lista de features y métricas."""
    cfg = cfg or get_config()
    m = cfg["model"]
    paths = {
        "model": path_from_root(m["final_model_path"]),
        "preprocessor": path_from_root(m["preprocessor_path"]),
        "features": path_from_root(m["feature_list_path"]),
        "metrics": path_from_root(m["metrics_path"]),
    }
    for p in set(paths.values()):
        p.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, paths["model"])
    joblib.dump(preprocessor, paths["preprocessor"])
    with open(paths["features"], "w") as f:
        json.dump(feature_list, f, indent=2)
    if metrics is not None:
        with open(paths["metrics"], "w") as f:
            json.dump(metrics, f, indent=2, default=str)
    print(f"[save] modelo -> {paths['model']}")
    print(f"[save] preprocesador -> {paths['preprocessor']}")
    print(f"[save] features ({len(feature_list)}) -> {paths['features']}")
    return {k: str(v) for k, v in paths.items()}


def load_model_artifacts(cfg: dict | None = None) -> tuple:
    """Carga modelo + preprocesador + features guardados."""
    cfg = cfg or get_config()
    m = cfg["model"]
    model = joblib.load(path_from_root(m["final_model_path"]))
    preprocessor = joblib.load(path_from_root(m["preprocessor_path"]))
    with open(path_from_root(m["feature_list_path"])) as f:
        features = json.load(f)
    return model, preprocessor, features

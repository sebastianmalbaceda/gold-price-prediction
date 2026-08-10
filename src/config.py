"""Configuración central del proyecto: carga YAML y rutas raíz."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Raíz del proyecto: src/config.py -> parents[2] es la raíz cuando se ejecuta
# como paquete (python -m src.config). Si se ejecuta como script desde
# cualquier cwd, se resuelve desde el propio fichero.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_paths(cfg: dict) -> dict:
    """Resuelve rutas relativas del YAML a absolutas bajo PROJECT_ROOT.

    Las rutas de data/, models/, reports/ y configs/ del YAML son relativas
    a la raíz del proyecto. Al resolverlas aquí, todos los módulos
    (notebooks, scripts, API) funcionan desde cualquier cwd.
    """
    import copy

    cfg = copy.deepcopy(cfg)
    root = str(PROJECT_ROOT)

    # data.*
    for key in ("raw_path", "interim_path", "processed_dir",
                "train_path", "val_path", "test_path"):
        if key in cfg.get("data", {}):
            cfg["data"][key] = str(PROJECT_ROOT / cfg["data"][key])

    # model.*
    for key in ("models_dir", "final_model_path", "preprocessor_path",
                "feature_list_path", "metrics_path"):
        if key in cfg.get("model", {}):
            cfg["model"][key] = str(PROJECT_ROOT / cfg["model"][key])

    return cfg


def load_yaml(name: str = "config.yaml") -> dict:
    """Carga un YAML de configs/ con rutas resueltas a absolutas."""
    path = PROJECT_ROOT / "configs" / name
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_paths(cfg) if name == "config.yaml" else cfg


def get_config() -> dict:
    return load_yaml("config.yaml")


def get_params() -> dict:
    return load_yaml("params.yaml")


def path_from_root(*parts: str) -> Path:
    """Devuelve una ruta absoluta relativa a la raíz del proyecto."""
    return PROJECT_ROOT.joinpath(*parts)

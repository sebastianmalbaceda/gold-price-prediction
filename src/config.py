"""Configuracion central del proyecto: carga YAML y rutas raiz."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/config.py -> src/ -> raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = PROJECT_ROOT / "configs"
_PATH_KEYS = (
    "raw_path",
    "interim_path",
    "processed_dir",
    "train_path",
    "val_path",
    "test_path",
)
_MODEL_PATH_KEYS = (
    "models_dir",
    "final_model_path",
    "preprocessor_path",
    "feature_list_path",
    "metrics_path",
)


def _as_project_path(value: Any) -> str:
    """Convierte una ruta de configuracion en una ruta absoluta normalizada.

    Se aceptan rutas absolutas para facilitar despliegues, pero las rutas del
    YAML distribuido son relativas a la raiz del repositorio. No se modifica el
    objeto original: ``_resolve_paths`` trabaja sobre una copia profunda.
    """
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Ruta de configuracion invalida: {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _resolve_paths(cfg: dict) -> dict:
    """Resuelve rutas relativas del YAML a absolutas bajo ``PROJECT_ROOT``."""
    import copy

    if not isinstance(cfg, dict):
        raise ValueError("La configuracion YAML debe ser un mapping")
    cfg = copy.deepcopy(cfg)
    for section in ("data", "model"):
        if section in cfg and not isinstance(cfg[section], dict):
            raise ValueError(f"La seccion {section} debe ser un mapping")

    for key in _PATH_KEYS:
        if key in cfg.get("data", {}):
            cfg["data"][key] = _as_project_path(cfg["data"][key])

    for key in _MODEL_PATH_KEYS:
        if key in cfg.get("model", {}):
            cfg["model"][key] = _as_project_path(cfg["model"][key])

    return cfg


def _config_path(name: str) -> Path:
    """Devuelve una ruta de configuracion confinada a ``configs/``."""
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("El nombre de configuracion debe ser un fichero simple")
    path = (_CONFIG_DIR / name).resolve()
    if path.parent != _CONFIG_DIR.resolve():
        raise ValueError("La configuracion debe estar dentro de configs/")
    return path


def load_yaml(name: str = "config.yaml") -> dict:
    """Carga un YAML de ``configs/`` con rutas y estructura validadas."""
    path = _config_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"No existe la configuracion: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"La configuracion {name} no contiene un mapping YAML")
    return _resolve_paths(cfg) if name == "config.yaml" else cfg


def get_config() -> dict:
    return load_yaml("config.yaml")


def get_params() -> dict:
    return load_yaml("params.yaml")


def path_from_root(*parts: str) -> Path:
    """Devuelve una ruta absoluta relativa a la raiz del proyecto."""
    return PROJECT_ROOT.joinpath(*parts)

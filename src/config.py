"""Configuración central del proyecto: carga YAML y rutas raíz."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Raíz del proyecto: src/config.py -> parents[2] es la raíz cuando se ejecuta
# como paquete (python -m src.config). Si se ejecuta como script desde
# cualquier cwd, se resuelve desde el propio fichero.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str = "config.yaml") -> dict:
    """Carga un YAML de configs/."""
    path = PROJECT_ROOT / "configs" / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config() -> dict:
    return load_yaml("config.yaml")


def get_params() -> dict:
    return load_yaml("params.yaml")


def path_from_root(*parts: str) -> Path:
    """Devuelve una ruta absoluta relativa a la raíz del proyecto."""
    return PROJECT_ROOT.joinpath(*parts)

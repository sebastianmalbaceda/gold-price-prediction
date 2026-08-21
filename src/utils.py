"""Utilidades comunes: semillas, guardado de figuras y JSON."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from src.config import path_from_root


def set_seed(seed: int = 42) -> None:
    """Fija las fuentes de aleatoriedad usadas por el proyecto."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed debe ser un entero")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_fig(fig, name: str, subdir: str = "figures") -> Path:
    """Guarda una figura matplotlib en reports/<subdir>/."""
    if not name or Path(name).name != name:
        raise ValueError("name debe ser un nombre de fichero sin subdirectorios")
    out_dir = path_from_root("reports", subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[fig] {path}")
    return path


def set_publication_style() -> None:
    """Estilo de gráficas nivel publicación científica.

    Tipografía serif, grid sutil, paleta accesible y tamaño adecuado
    para informes impresos y defensa de TFG/TFM.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 150,
        }
    )


def save_json(obj, name: str, subdir: str = "") -> Path:
    """Guarda un JSON en reports/ (o subcarpeta)."""
    if not name or Path(name).name != name:
        raise ValueError("name debe ser un nombre de fichero sin subdirectorios")
    out_dir = path_from_root("reports", subdir) if subdir else path_from_root("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[json] {path}")
    return path

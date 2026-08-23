"""Utilidades comunes: semillas, guardado de figuras y JSON."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.config import path_from_root


def atomic_write(path: Path, writer: Callable[[Path], None]) -> Path:
    """Escribe ``path`` de forma atomica delegando el volcado en ``writer``.

    Es la unica implementacion del patron escribir-temporal-y-renombrar del
    proyecto; antes estaba duplicada en ``train_model``, ``classifier``,
    ``deep_learning``, ``save_json`` y ``scripts/monitor_drift.py``.

    El nombre temporal incluye el PID para que dos procesos que guarden el
    mismo artefacto en paralelo no se pisen el fichero intermedio.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_write_text(content: str, path: Path, encoding: str = "utf-8") -> Path:
    """Escribe texto de forma atomica."""
    return atomic_write(path, lambda p: p.write_text(content, encoding=encoding))


def atomic_write_joblib(value, path: Path) -> Path:
    """Serializa un objeto con joblib de forma atomica."""
    import joblib

    return atomic_write(path, lambda p: joblib.dump(value, p))


def set_seed(seed: int = 42) -> None:
    """Fija las fuentes de aleatoriedad usadas por el proyecto."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed debe ser un entero")
    random.seed(seed)
    # ``np.random.seed`` fija el generador legado global. Se mantiene porque
    # scikit-learn y varias dependencias siguen leyendolo; el codigo propio
    # debe usar ``np.random.default_rng(seed)``.
    np.random.seed(seed)  # noqa: NPY002
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
    atomic_write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), path)
    print(f"[json] {path}")
    return path

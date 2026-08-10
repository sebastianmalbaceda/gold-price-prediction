"""Utilidades comunes: semillas, guardado de figuras y JSON."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, path_from_root


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)


def save_fig(fig, name: str, subdir: str = "figures") -> Path:
    """Guarda una figura matplotlib en reports/<subdir>/."""
    out_dir = path_from_root("reports", subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[fig] {path}")
    return path


def save_json(obj, name: str, subdir: str = "") -> Path:
    """Guarda un JSON en reports/ (o subcarpeta)."""
    out_dir = path_from_root("reports", subdir) if subdir else path_from_root("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[json] {path}")
    return path

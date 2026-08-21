"""Division temporal estricta y TimeSeriesSplit (fase 7)."""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import get_config, get_params


def temporal_split(df: pd.DataFrame, cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Divide por fechas según config, sin aleatoriedad ni solapamientos."""
    cfg = cfg or get_config()
    if "date" not in df.columns:
        raise ValueError("El dataframe debe contener la columna 'date'")
    if df.empty:
        raise ValueError("No se puede dividir un dataframe vacio")

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("date debe contener fechas validas y unicas")
    if not dates.is_monotonic_increasing:
        raise ValueError("El dataframe debe estar ordenado cronologicamente")
    work = df.copy()
    work["date"] = dates

    split_cfg = cfg.get("split")
    if not isinstance(split_cfg, dict):
        raise ValueError("La configuracion debe definir split.train/val/test")

    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name in ("train", "val", "test"):
        bounds = split_cfg.get(name)
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Rango de split invalido para {name}")
        start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        if start > end:
            raise ValueError(f"Rango de split invertido para {name}")
        ranges[name] = (start, end)

    previous_end = None
    for name in ("train", "val", "test"):
        start, end = ranges[name]
        if previous_end is not None and start <= previous_end:
            raise ValueError("Los rangos temporalmente consecutivos se solapan")
        previous_end = end

    parts = {}
    for name, (start, end) in ranges.items():
        mask = (dates >= start) & (dates <= end)
        part = work.loc[mask].copy().reset_index(drop=True)
        if part.empty:
            raise ValueError(f"El split {name} no contiene observaciones")
        parts[name] = part
    return parts


def get_temporal_splitter(cfg: dict | None = None) -> TimeSeriesSplit:
    """Devuelve ``TimeSeriesSplit`` con los parámetros configurados."""
    cfg = cfg or get_config()
    cv = cfg.get("cv") or get_params().get("cv", {})
    try:
        n_splits = int(cv["n_splits"])
        gap = int(cv.get("gap", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("cv requiere n_splits entero y gap opcional") from exc
    if n_splits < 2 or gap < 0:
        raise ValueError("cv.n_splits debe ser >=2 y cv.gap no negativo")
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def drop_warmup(df: pd.DataFrame, warmup: int = 260) -> pd.DataFrame:
    """Elimina filas iniciales hasta disponer de features no nulas.

    ``warmup`` es un mínimo explícito. Además se calcula la posición del
    primer valor válido de cada columna; las columnas completamente vacías se
    rechazan porque no pueden entrar en ningún modelo.
    """
    if not isinstance(warmup, numbers.Integral) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup debe ser un entero no negativo")
    if df.empty:
        return df.copy().reset_index(drop=True)

    valid = df.notna().to_numpy(dtype=bool)
    if (~valid.any(axis=0)).any():
        all_null = df.columns[~valid.any(axis=0)].tolist()
        raise ValueError(f"Columnas completamente nulas: {all_null[:10]}")

    first_valid_positions = np.argmax(valid, axis=0)
    warmup_by_feature = int(first_valid_positions.max())
    warmup_effective = max(int(warmup), warmup_by_feature)
    return df.iloc[warmup_effective:].reset_index(drop=True)

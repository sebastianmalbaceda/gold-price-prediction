"""División temporal estricta y TimeSeriesSplit (fase 7).

Para forecasting: orden temporal estricto, nunca datos aleatorios.
Train 2000-2019 | Val 2020-2022 | Test 2023-2025 (configurado en config.yaml).
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import get_config, get_params


def temporal_split(df: pd.DataFrame, cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Divide por fechas según config: train/val/test (orden temporal estricto)."""
    cfg = cfg or get_config()
    s = cfg["split"]
    train = df[(df["date"] >= s["train"][0]) & (df["date"] <= s["train"][1])]
    val = df[(df["date"] >= s["val"][0]) & (df["date"] <= s["val"][1])]
    test = df[(df["date"] >= s["test"][0]) & (df["date"] <= s["test"][1])]
    return {"train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def get_temporal_splitter(cfg: dict | None = None) -> TimeSeriesSplit:
    """TimeSeriesSplit para CV sobre train+val (n_splits y gap de config)."""
    cfg = cfg or get_config()
    cv = cfg.get("cv") or get_params().get("cv", {})
    return TimeSeriesSplit(n_splits=cv["n_splits"], gap=cv.get("gap", 0))


def drop_warmup(df: pd.DataFrame, warmup: int = 260) -> pd.DataFrame:
    """Elimina las primeras filas sin lags/ventanas completos.

    El warm-up se calcula por feature: se toma el primer índice sin NaN de
    cada columna (cubre lags 126d, rolling 126d y exógenas con huecos como
    us_gdp) y se eliminan todas las filas anteriores al máximo. Si se pasa
    `warmup` explícito, se usa max(warmup, warmup_por_feature).
    """
    if df.empty:
        return df
    first_valid = df.notna().idxmax()  # primer índice no-NaN por columna
    # Para columnas sin NaN, idxmax() es el primer índice -> no recorta
    min_idx = max(int(first_valid.min()), int(first_valid.max())
                  if first_valid.notna().all() else 0)
    warmup_effective = max(int(warmup), min_idx) if warmup else min_idx
    return df.iloc[warmup_effective:].reset_index(drop=True)

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

    La función es segura para dataframes sin NaN (no elimina nada por
    defecto cuando warmup=0) y para dataframes vacíos.
    """
    if df.empty:
        return df

    # Primer índice no-NaN por columna; para columnas sin NaN es 0
    first_valid = df.notna().idxmax()
    # Si TODAS las columnas tienen algún NaN, el máximo de primeros índices
    # marca el calentamiento necesario; si alguna columna no tiene NaN,
    # idxmax devuelve el primer índice (0) y no debe recortar.
    if first_valid.notna().all():
        # Hay columnas sin NaN: el warm-up por-feature solo aplica a las que
        # tienen NaN inicial; tomar el máximo entre ellas.
        cols_with_nan = df.columns[df.isna().any()]
        if len(cols_with_nan) > 0:
            warmup_by_feature = int(first_valid[cols_with_nan].max())
        else:
            warmup_by_feature = 0
    else:
        warmup_by_feature = int(first_valid.max())

    warmup_effective = max(int(warmup), warmup_by_feature)
    return df.iloc[warmup_effective:].reset_index(drop=True)

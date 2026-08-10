"""Feature engineering para series temporales (fase 9).

Todo se construye con ventanas deslizantes sobre datos PASADOS:
- Lags del target (gold_spot)
- Retornos logarítmicos
- Estadísticos rolling (media, desv., retorno acumulado) del target
- Lags de exógenas y sus retornos
- Variables de calendario (año, mes, día de semana)
- Indicadores de huecos (ausencia original de exógena)

Advertencia de leakage: las primeras `max(window)` filas del dataset
resultante contienen NaNs y deben eliminarse (o el split debe respetarlas).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_config


def _target_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"gold_spot_lag{lag}"] = df["gold_spot"].shift(lag)
    return df


def _target_returns(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    log_price = np.log(df["gold_spot"])
    for lag in lags:
        df[f"gold_ret_lag{lag}"] = log_price - log_price.shift(lag)
    return df


def _target_rolling(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    log_price = np.log(df["gold_spot"])
    for w in windows:
        df[f"gold_ret_roll{w}"] = log_price - log_price.shift(w)           # retorno w días
        df[f"gold_rollmean_{w}"] = df["gold_spot"].rolling(w, min_periods=w // 2).mean()
        df[f"gold_rollstd_{w}"] = df["gold_spot"].rolling(w, min_periods=w // 2).std()
    return df


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.DatetimeIndex(df["date"])
    df["year"] = dt.year.astype("int32")
    df["month"] = dt.month.astype("int32")
    df["dayofweek"] = dt.dayofweek.astype("int32")
    # Trimestre y día del año para captar estacionalidad
    df["quarter"] = dt.quarter.astype("int32")
    df["dayofyear"] = dt.dayofyear.astype("int32")
    return df


def _exogenous_features(df: pd.DataFrame, exog: list[str], lags: list[int]) -> pd.DataFrame:
    for col in exog:
        if col in ("date", "gold_spot"):
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
            # Retorno log del nivel de la exógena (siempre > 0 para índices/precios)
            if df[col].min() > 0:
                df[f"{col}_ret_lag{lag}"] = np.log(df[col]) - np.log(df[col]).shift(lag)
        # Indicador de dato original ausente (antes del ffill) si está disponible
    return df


def _gap_indicators(df: pd.DataFrame, exog: list[str], raw: pd.DataFrame | None) -> pd.DataFrame:
    """Marca con 1/0 si el valor original de la exógena faltaba ese día."""
    if raw is None:
        return df
    # Solo para columnas presentes en el dataframe limpio (las excluidas no tienen indicador)
    present = [c for c in exog
               if c in df.columns and c not in ("date", "gold_spot")
               and c in raw.columns]
    if not present:
        return df
    raw_idx = pd.DatetimeIndex(raw["date"])
    raw_series = raw.set_index("date")[present]
    aligned = raw_series.reindex(pd.DatetimeIndex(df["date"]))
    missing_df = aligned.isna().astype("int8")
    missing_df.columns = [f"{c}_missing" for c in missing_df.columns]
    # Alinear por posición (no por índice): reset del índice para que el
    # concat por columnas no haga un join externo y duplique filas.
    missing_df = missing_df.reset_index(drop=True)
    # Concatenar de una vez (evita fragmentación del DataFrame)
    return pd.concat([df.reset_index(drop=True), missing_df], axis=1)


def build_features(df: pd.DataFrame, cfg: dict | None = None,
                   raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Genera el feature set completo a partir del dataframe limpio."""
    cfg = cfg or get_config()
    f = cfg["features"]

    out = df.copy()
    out = _target_lags(out, f["target_lags"])
    out = _target_returns(out, f["target_lags"])
    out = _target_rolling(out, f["rolling_windows"])
    out = _calendar_features(out)

    exog = [c for c in df.columns if c not in ("date", "gold_spot")]
    out = _exogenous_features(out, exog, f["exogenous_lags"])
    out = _gap_indicators(out, exog, raw)

    # Defensa en profundidad: eliminar columnas completamente vacías
    # (p.ej. indicadores _missing de features excluidas en versiones previas)
    all_null = out.columns[out.isna().all()]
    if len(all_null) > 0:
        out = out.drop(columns=all_null)

    return out


def make_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Crea targets a futuro: gold_spot desplazado -h días (target=h).

    IMPORTANTE: se crean ANTES del split y se eliminan las filas cuyo
    target cae fuera del conjunto (fin de serie) para evitar targets
    con información futura dentro del propio conjunto de entrenamiento.
    """
    out = df.copy()
    for h in horizons:
        out[f"target_{h}"] = out["gold_spot"].shift(-h)
    # Filas con target futuro no disponible se descartan
    out = out.dropna(subset=[f"target_{h}" for h in horizons])
    return out.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame, horizons: list[int]) -> list[str]:
    """Columnas usadas como features (todo excepto date, gold_spot, targets)."""
    drop = {"date", "gold_spot"} | {f"target_{h}" for h in horizons}
    return [c for c in df.columns if c not in drop]


if __name__ == "__main__":
    import sys

    from src.data.load_data import build_interim, load_raw

    cfg = get_config()
    raw = load_raw(cfg)
    clean = build_interim(cfg)
    feats = build_features(clean, cfg, raw)
    feats = make_targets(feats, cfg["target"]["horizons"])
    out_path = cfg["data"]["processed_dir"]
    feats.to_parquet(f"{out_path}/features.parquet", index=False)
    print(f"[features] guardado en {out_path}/features.parquet shape={feats.shape}")
    print("[features] columnas:", len(get_feature_columns(feats, cfg["target"]["horizons"])))

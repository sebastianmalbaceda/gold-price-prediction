"""Carga de datos (fase 2-3): lectura, limpieza base y guardado en interim/."""
from __future__ import annotations

import pandas as pd

from src.config import get_config


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    """Carga el CSV crudo sin modificar (immutabilidad de data/raw)."""
    cfg = cfg or get_config()
    path = cfg["data"]["raw_path"]
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_daily_series(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Limpieza base de la serie diaria (fase 8, parte 1):

    1. Recorta a la ventana de análisis configurada.
    2. Elimina filas de fin de semana (no hay cotización de oro).
    3. Descarta features con cobertura insuficiente (config).
    4. Forward-fill de exógenas (solo usa pasado) y gold_spot (festivos).
    5. Elimina filas donde el target sigue ausente.
    """
    cfg = cfg or get_config()
    d = cfg["data"]
    start, end = pd.Timestamp(d["start_date"]), pd.Timestamp(d["end_date"])

    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    df = df[df["date"].dt.dayofweek < 5].copy()          # sin fines de semana
    df = df.reset_index(drop=True)

    # Features excluidas por cobertura (ver config)
    excluded = d.get("excluded_features", [])
    keep = [c for c in df.columns if c not in excluded]

    df = df[keep].copy()
    # gold_spot se rellena hacia delante como mucho 3 días hábiles
    df["gold_spot"] = df["gold_spot"].ffill(limit=3)

    # Forward-fill del resto de columnas (solo pasado, sin mirar futuro)
    df = df.ffill()

    # Descartar features con demasiados nulos residuales
    max_missing = d.get("max_missing_after_ffill", 0.10)
    missing = df.isna().mean()
    bad = missing[missing > max_missing].index.tolist()
    if bad:
        df = df.drop(columns=bad)

    # Quedan filas sin target -> se eliminan (no se pueden usar)
    before = len(df)
    df = df.dropna(subset=["gold_spot"])
    print(f"  [clean] filas con target: {len(df)} (descartadas {before - len(df)})")
    return df.reset_index(drop=True)


def build_interim(cfg: dict | None = None) -> pd.DataFrame:
    """Pipeline completo crudo -> interim (parquet). Idempotente."""
    cfg = cfg or get_config()
    df = load_raw(cfg)
    clean = clean_daily_series(df, cfg)
    out = cfg["data"]["interim_path"]
    clean.to_parquet(out, index=False)
    print(f"[interim] guardado en {out}  shape={clean.shape}")
    return clean


if __name__ == "__main__":
    build_interim()

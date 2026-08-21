"""Carga de datos (fases 2-3): lectura, limpieza base y guardado en interim/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import get_config


def _validate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza ``date`` y garantiza una observacion por fecha."""
    if "date" not in df.columns:
        raise ValueError("El dataset debe contener la columna 'date'")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("La columna 'date' contiene fechas invalidas")
    if out["date"].duplicated().any():
        duplicated = int(out["date"].duplicated().sum())
        raise ValueError(f"El dataset contiene {duplicated} fechas duplicadas")
    return out.sort_values("date").reset_index(drop=True)


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    """Carga el CSV crudo sin modificar (inmutabilidad de ``data/raw``)."""
    cfg = cfg or get_config()
    path = Path(cfg["data"]["raw_path"])
    if not path.is_file():
        raise FileNotFoundError(f"No existe el dataset crudo: {path}")
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"No se pudo leer el dataset crudo {path}: {exc}") from exc
    if "gold_spot" not in df.columns:
        raise ValueError("El dataset debe contener la columna 'gold_spot'")
    return _validate_dates(df)


def clean_daily_series(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Limpia la serie diaria de forma causal.

    1. Recorta a la ventana configurada y elimina fines de semana.
    2. Conserva ``date`` y ``gold_spot`` aunque no aparezcan en exclusiones.
    3. Convierte los valores a numerico y aplica forward-fill exclusivamente
       hacia delante. El precio del oro se rellena como maximo tres dias
       habiles; las exogenas pueden conservar su ultimo valor conocido.
    4. Descarta exogenas con cobertura residual insuficiente y filas sin target.

    No se usa ningun valor posterior para rellenar una observacion.
    """
    cfg = cfg or get_config()
    d = cfg.get("data", {})
    if "gold_spot" not in df.columns:
        raise ValueError("El dataframe debe contener la columna 'gold_spot'")

    work = _validate_dates(df)
    start, end = pd.Timestamp(d["start_date"]), pd.Timestamp(d["end_date"])
    if start > end:
        raise ValueError("start_date no puede ser posterior a end_date")

    work = work[(work["date"] >= start) & (work["date"] <= end)]
    work = work[work["date"].dt.dayofweek < 5].reset_index(drop=True)
    if work.empty:
        return work

    # La configuracion historica guardaba esta lista bajo ``features``;
    # tambien se acepta bajo ``data`` para compatibilidad con configuraciones
    # antiguas, pero ambas fuentes se combinan.
    excluded = set(d.get("excluded_features", []))
    excluded.update(cfg.get("features", {}).get("excluded_features", []))
    protected = {"date", "gold_spot"}
    keep = [c for c in work.columns if c not in excluded or c in protected]
    work = work[keep].copy()

    value_cols = [c for c in work.columns if c not in protected]
    for col in ["gold_spot", *value_cols]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # El target tiene un limite explicito: no convertir una ausencia larga en
    # un precio ficticio. Las exogenas se mantienen causales mediante ffill.
    work["gold_spot"] = work["gold_spot"].ffill(limit=3)
    if value_cols:
        work[value_cols] = work[value_cols].ffill()

    max_missing = float(d.get("max_missing_after_ffill", 0.10))
    if not 0 <= max_missing <= 1:
        raise ValueError("max_missing_after_ffill debe estar entre 0 y 1")
    missing = work[value_cols].isna().mean() if value_cols else pd.Series(dtype=float)
    bad = missing[missing > max_missing].index.tolist()
    if bad:
        work = work.drop(columns=bad)

    before = len(work)
    work = work.dropna(subset=["gold_spot"])
    print(f"  [clean] filas con target: {len(work)} (descartadas {before - len(work)})")
    return work.reset_index(drop=True)


def build_interim(cfg: dict | None = None) -> pd.DataFrame:
    """Pipeline completo crudo -> interim (parquet), idempotente."""
    cfg = cfg or get_config()
    df = load_raw(cfg)
    clean = clean_daily_series(df, cfg)
    out = Path(cfg["data"]["interim_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out, index=False)
    print(f"[interim] guardado en {out}  shape={clean.shape}")
    return clean


if __name__ == "__main__":
    build_interim()

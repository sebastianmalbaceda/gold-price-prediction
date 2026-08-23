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


def _max_internal_gap(series: pd.Series) -> int:
    """Longitud del mayor hueco de NaN *interior* de una serie.

    Los NaN iniciales y finales se ignoran: no son rellenables por ffill y no
    generan valores ficticios propagados.
    """
    observed = series.notna().to_numpy()
    if not observed.any():
        return int(len(series))
    first, last = observed.argmax(), len(observed) - observed[::-1].argmax() - 1
    inner = observed[first : last + 1]
    longest = current = 0
    for is_observed in inner:
        current = 0 if is_observed else current + 1
        longest = max(longest, current)
    return int(longest)


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

    Claves de configuracion relevantes bajo ``data`` (todas opcionales):

    - ``max_missing_after_ffill`` (por defecto ``0.10``): fraccion maxima de
      NaN admitida *despues* del relleno. Por si sola no detecta series casi
      vacias, porque el ffill las deja sin NaN.
    - ``min_coverage_before_ffill`` (por defecto ``0.0``): fraccion minima de
      observaciones **reales** exigida antes del relleno. Es el filtro que de
      verdad elimina exogenas con cobertura testimonial.
    - ``max_ffill_gap_days`` (por defecto ``null`` = sin limite): numero maximo
      de dias habiles que se permite propagar un valor de exogena.

    En todos los casos se imprime un diagnostico con la cobertura real y el
    mayor hueco interno de las columnas mas debiles.
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

    # Diagnostico ANTES del ffill: una vez propagado el ultimo valor conocido,
    # el porcentaje de NaN deja de informar sobre la calidad real de la serie.
    coverage_before = work[value_cols].notna().mean() if value_cols else pd.Series(dtype=float)
    gaps_before = (
        pd.Series({c: _max_internal_gap(work[c]) for c in value_cols}, dtype="int64")
        if value_cols
        else pd.Series(dtype="int64")
    )

    # El target tiene un limite explicito: no convertir una ausencia larga en
    # un precio ficticio. Las exogenas se mantienen causales mediante ffill.
    work["gold_spot"] = work["gold_spot"].ffill(limit=3)
    max_gap = d.get("max_ffill_gap_days")
    if max_gap is not None:
        max_gap = int(max_gap)
        if max_gap < 1:
            raise ValueError("max_ffill_gap_days debe ser un entero positivo o null")
    if value_cols:
        work[value_cols] = work[value_cols].ffill(limit=max_gap)

    max_missing = float(d.get("max_missing_after_ffill", 0.10))
    if not 0 <= max_missing <= 1:
        raise ValueError("max_missing_after_ffill debe estar entre 0 y 1")
    missing = work[value_cols].isna().mean() if value_cols else pd.Series(dtype=float)
    bad = set(missing[missing > max_missing].index)

    # Filtro de cobertura REAL (previa al ffill). Por defecto 0.0 para no
    # alterar el comportamiento historico; subirlo descarta columnas que solo
    # sobreviven porque el ffill propago un puñado de observaciones.
    min_coverage = float(d.get("min_coverage_before_ffill", 0.0))
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage_before_ffill debe estar entre 0 y 1")
    low_coverage = set(coverage_before[coverage_before < min_coverage].index)
    bad |= low_coverage

    if value_cols:
        weakest = coverage_before.sort_values().head(5)
        print("  [clean] cobertura real (pre-ffill) de las 5 exogenas mas debiles:")
        for col, cov in weakest.items():
            flag = " <- DESCARTADA" if col in bad else ""
            print(
                f"    {col}: {cov:.2%} observado, hueco interno maximo "
                f"{int(gaps_before.get(col, 0))} dias{flag}"
            )
    if bad:
        ordered_bad = [c for c in value_cols if c in bad]
        print(f"  [clean] columnas descartadas ({len(ordered_bad)}): {ordered_bad}")
        work = work.drop(columns=ordered_bad)

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

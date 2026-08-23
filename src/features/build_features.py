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

import warnings

import numpy as np
import pandas as pd

from src.config import get_config


def _target_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"gold_spot_lag{lag}"] = df["gold_spot"].shift(lag)
    return df


def _validate_positive_price(series: pd.Series) -> None:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
        raise ValueError("gold_spot debe contener valores positivos y finitos")


def _validate_lags(values: list[int], name: str) -> list[int]:
    if not values or any(isinstance(v, bool) or int(v) != v or int(v) < 1 for v in values):
        raise ValueError(f"{name} debe contener enteros positivos")
    return [int(v) for v in values]


def _target_returns(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    _validate_positive_price(df["gold_spot"])
    log_price = np.log(df["gold_spot"])
    for lag in lags:
        df[f"gold_ret_lag{lag}"] = log_price - log_price.shift(lag)
    return df


def _target_rolling(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    _validate_positive_price(df["gold_spot"])
    log_price = np.log(df["gold_spot"])
    for w in windows:
        min_periods = max(1, w // 2)
        df[f"gold_ret_roll{w}"] = log_price - log_price.shift(w)  # retorno w días
        df[f"gold_rollmean_{w}"] = df["gold_spot"].rolling(w, min_periods=min_periods).mean()
        df[f"gold_rollstd_{w}"] = df["gold_spot"].rolling(w, min_periods=min_periods).std()
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


def _exogenous_features(
    df: pd.DataFrame,
    exog: list[str],
    lags: list[int],
    fit_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Lags y retornos logaritmicos de las exogenas.

    ``fit_mask`` delimita las filas que pueden usarse para tomar decisiones de
    diseño (tipicamente, solo el tramo de entrenamiento). Sin ella, la simple
    decision de "¿esta exogena admite retornos logaritmicos?" dependeria de
    observaciones futuras, de modo que la *composicion misma del feature set*
    filtraria informacion del periodo de validacion o test.
    """
    for col in exog:
        if col in ("date", "gold_spot"):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        # Solo calcular retornos log si todos los valores observados son
        # estrictamente positivos (indices, precios y tipos pueden tener
        # escalas distintas). La decision se toma exclusivamente sobre el
        # tramo de entrenamiento cuando este disponible.
        decision = series[fit_mask] if fit_mask is not None else series
        observed_fit = decision.dropna()
        if observed_fit.empty:  # sin datos en train: se recurre a la serie completa
            observed_fit = series.dropna()
        observed_all = series.dropna()
        positive_in_fit = not observed_fit.empty and bool((observed_fit > 0).all())
        positive_overall = not observed_all.empty and bool((observed_all > 0).all())
        if positive_in_fit and not positive_overall:
            # La decision tomada con datos de entrenamiento queda invalidada por
            # observaciones posteriores no positivas: el logaritmo no esta
            # definido en todo el rango, asi que no se crea la columna. Se avisa
            # porque significa que la composicion del feature set depende del
            # futuro (leakage de esquema, no de valores).
            warnings.warn(
                f"'{col}' es positiva en entrenamiento pero no en el resto del "
                "rango: se omiten sus retornos logaritmicos. Congela el esquema "
                "de features en entrenamiento para evitar esta dependencia.",
                UserWarning,
                stacklevel=2,
            )
        use_log_returns = positive_in_fit and positive_overall
        # np.log se calcula una sola vez por columna, no una vez por lag.
        log_series = np.log(series) if use_log_returns else None
        for lag in lags:
            df[f"{col}_lag{lag}"] = series.shift(lag)
            if log_series is not None:
                df[f"{col}_ret_lag{lag}"] = log_series - log_series.shift(lag)
        # Indicador de dato original ausente (antes del ffill) si está disponible
    return df


def _gap_indicators(df: pd.DataFrame, exog: list[str], raw: pd.DataFrame | None) -> pd.DataFrame:
    """Marca con 1/0 si el valor original de la exógena faltaba ese día."""
    if raw is None:
        return df
    # Solo para columnas presentes en el dataframe limpio (las excluidas no tienen indicador)
    present = [
        c for c in exog if c in df.columns and c not in ("date", "gold_spot") and c in raw.columns
    ]
    if not present:
        return df
    raw_copy = raw.copy()
    raw_copy["date"] = pd.to_datetime(raw_copy["date"], errors="coerce")
    if raw_copy["date"].isna().any() or raw_copy["date"].duplicated().any():
        raise ValueError("raw debe tener fechas validas y unicas para crear indicadores")
    raw_series = raw_copy.set_index("date")[present]
    aligned = raw_series.reindex(pd.DatetimeIndex(df["date"]))
    missing_df = aligned.isna().astype("int8")
    missing_df.columns = [f"{c}_missing" for c in missing_df.columns]
    # Alinear por posición (no por índice): reset del índice para que el
    # concat por columnas no haga un join externo y duplique filas.
    missing_df = missing_df.reset_index(drop=True)
    # Concatenar de una vez (evita fragmentación del DataFrame)
    return pd.concat([df.reset_index(drop=True), missing_df], axis=1)


def _fit_mask(dates: pd.Series, cfg: dict) -> pd.Series | None:
    """Mascara booleana de las filas pertenecientes al tramo de entrenamiento.

    Devuelve ``None`` si la configuracion no define ``split.train``, en cuyo
    caso el llamante usa la serie completa (comportamiento historico).
    """
    train_range = cfg.get("split", {}).get("train")
    if not train_range or len(train_range) < 2:
        return None
    try:
        train_end = pd.Timestamp(train_range[1])
    except (TypeError, ValueError):
        return None
    mask = dates <= train_end
    return mask if bool(mask.any()) else None


def build_features(
    df: pd.DataFrame, cfg: dict | None = None, raw: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Genera el feature set completo a partir del dataframe limpio.

    El PerformanceWarning de pandas por fragmentación (inserción repetida
    de columnas) se suprime: es un aviso de rendimiento, no de corrección,
    y el DataFrame se consolida al final con una copia.
    """
    cfg = cfg or get_config()
    f = cfg["features"]
    if "date" not in df.columns or "gold_spot" not in df.columns:
        raise ValueError("El dataframe debe contener date y gold_spot")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("El dataframe debe estar ordenado y tener fechas validas unicas")
    _validate_positive_price(df["gold_spot"])
    target_lags = _validate_lags(f.get("target_lags", []), "target_lags")
    rolling_windows = _validate_lags(f.get("rolling_windows", []), "rolling_windows")
    exogenous_lags = _validate_lags(f.get("exogenous_lags", []), "exogenous_lags")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

        out = df.copy()
        out = _target_lags(out, target_lags)
        out = _target_returns(out, target_lags)
        out = _target_rolling(out, rolling_windows)
        if f.get("calendar", True):
            out = _calendar_features(out)

        exog = [c for c in df.columns if c not in ("date", "gold_spot")]
        out = _exogenous_features(out, exog, exogenous_lags, _fit_mask(dates, cfg))
        out = _gap_indicators(out, exog, raw)

        # Defensa en profundidad: eliminar columnas completamente vacías
        # (p.ej. indicadores _missing de features excluidas en versiones previas)
        all_null = out.columns[out.isna().all()]
        if len(all_null) > 0:
            out = out.drop(columns=all_null)

        # Consolidar el DataFrame (mejora el rendimiento de acceso posterior)
        out = out.copy()

    return out


def make_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Crea targets a futuro usando observaciones de días hábiles.

    Los targets se crean antes del split y se eliminan las últimas filas sin
    observacion futura. Ningun target se usa como feature posteriormente.
    """
    if "gold_spot" not in df.columns:
        raise ValueError("El dataframe debe contener gold_spot")
    horizons = _validate_lags(horizons, "horizons")
    out = df.copy()
    for h in horizons:
        out[f"target_{h}"] = out["gold_spot"].shift(-h)
    return out.dropna(subset=[f"target_{h}" for h in horizons]).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame, horizons: list[int] | None = None) -> list[str]:
    """Devuelve features y excluye siempre todas las columnas ``target_*``.

    Aunque el llamador solicite un solo horizonte, no se puede dejar otro
    target futuro en la matriz: hacerlo introduciria leakage accidental.
    ``horizons`` se conserva por compatibilidad y se valida si se proporciona.
    """
    if horizons is not None:
        _validate_lags(horizons, "horizons")
    return [
        column
        for column in df.columns
        if column not in {"date", "gold_spot"} and not column.startswith("target_")
    ]


if __name__ == "__main__":
    from src.data.load_data import build_interim, load_raw

    cfg = get_config()
    raw = load_raw(cfg)
    clean = build_interim(cfg)
    feats = build_features(clean, cfg, raw)
    feats = make_targets(feats, cfg["target"]["horizons"])
    from pathlib import Path

    out_path = Path(cfg["data"]["processed_dir"])
    out_path.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path / "features.parquet", index=False)
    print(f"[features] guardado en {out_path}/features.parquet shape={feats.shape}")
    print("[features] columnas:", len(get_feature_columns(feats, cfg["target"]["horizons"])))

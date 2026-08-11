"""Script CLI de predicción batch (fase 21).

Uso:
    python scripts/predict.py --date 2025-09-12
    python scripts/predict.py --csv data/processed/test.parquet --out reports/batch_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import get_config, path_from_root  # noqa: E402
from src.data.load_data import load_raw  # noqa: E402
from src.features.build_features import build_features, make_targets  # noqa: E402
from src.models.train_model import load_model_artifacts  # noqa: E402


def predict_row(features_row: pd.Series, model, pp, sel_cols, horizon: int = 1) -> float:
    """Predice el target a `horizon` días a partir de una fila de features."""
    X = pp.transform(features_row[sel_cols].to_numpy(dtype=np.float64).reshape(1, -1))
    return float(model.predict(X)[0])


def predict_for_date(date_str: str, model, pp, sel_cols, cfg: dict) -> float:
    """Predice para una fecha: reconstruye las features hasta ese día.

    Reutiliza el pipeline de entrenamiento exacto: clean_daily_series
    (limpieza + ffill + exclusión por cobertura) -> build_features ->
    make_targets -> warm-up. Lanza ValueError si la fecha no tiene datos.
    """
    from src.data.load_data import clean_daily_series

    target_date = pd.Timestamp(date_str)
    raw = load_raw(cfg)

    # Rechazar fechas posteriores al último dato disponible (no hay con qué predecir)
    last_date = raw["date"].max()
    if target_date > last_date:
        raise ValueError(
            f"La fecha {date_str} es posterior al último dato disponible "
            f"({last_date.date()}). No se puede predecir sin datos."
        )

    raw = raw[raw["date"] <= target_date]

    if raw.empty:
        raise ValueError(f"No hay datos hasta la fecha {date_str}")

    # Limpieza idéntica a la de entrenamiento (ffill, exclusión, sin futuro)
    clean = clean_daily_series(raw, cfg)
    if clean.empty:
        raise ValueError(f"Sin gold_spot disponible hasta {date_str}")

    feats = build_features(clean, cfg, raw)
    feats = make_targets(feats, cfg["target"]["horizons"])
    # Warm-up: mismas reglas que en train (drop_warmup con 260)
    from src.data.split import drop_warmup

    feats = drop_warmup(feats, warmup=260)

    if feats.empty or feats[sel_cols].isna().any().any():
        raise ValueError(f"Datos insuficientes para construir features en {date_str}")

    row = feats.iloc[-1]
    return predict_row(row, model, pp, sel_cols, cfg["target"]["primary_horizon"])


def main():
    ap = argparse.ArgumentParser(description="Predicción del precio del oro (USD/oz)")
    ap.add_argument("--date", type=str, help="Fecha (YYYY-MM-DD) para una predicción puntual")
    ap.add_argument("--csv", type=str, help="CSV/parquet con filas de features a predecir")
    ap.add_argument(
        "--out",
        type=str,
        default="reports/batch_predictions.csv",
        help="Ruta de salida para modo --csv",
    )
    args = ap.parse_args()

    cfg = get_config()
    model, pp, sel_cols = load_model_artifacts(cfg)
    h = cfg["target"]["primary_horizon"]

    if args.date:
        try:
            pred = predict_for_date(args.date, model, pp, sel_cols, cfg)
        except (ValueError, IndexError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(2)
        print(f"Predicción gold_spot en {args.date} +{h}d: {pred:.2f} USD/oz")
    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: no existe {csv_path}", file=sys.stderr)
            sys.exit(2)
        df = pd.read_parquet(csv_path) if csv_path.suffix == ".parquet" else pd.read_csv(csv_path)
        missing = [c for c in sel_cols if c not in df.columns]
        if missing:
            print(f"Error: faltan columnas en el CSV: {missing[:10]}...", file=sys.stderr)
            sys.exit(2)
        df["prediction"] = df.apply(lambda r: predict_row(r, model, pp, sel_cols, h), axis=1)
        out_path = path_from_root(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Predicciones guardadas en {out_path} ({len(df)} filas)")
    else:
        ap.error("Indique --date o --csv")


if __name__ == "__main__":
    main()

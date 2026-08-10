"""Script CLI de predicción batch (fase 21).

Uso:
    python scripts/predict.py --date 2025-09-12
    python scripts/predict.py --csv data/processed/test.parquet --out reports/batch_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.load_data import load_raw
from src.data.split import drop_warmup, temporal_split
from src.features.build_features import build_features, make_targets
from src.models.train_model import load_model_artifacts


def predict_row(features_row: pd.Series, model, pp, sel_cols, horizon: int = 1) -> float:
    X = pp.transform(features_row[sel_cols].to_numpy(dtype=np.float64).reshape(1, -1))
    return float(model.predict(X)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, help="Fecha (YYYY-MM-DD) para una predicción")
    ap.add_argument("--csv", type=str, help="CSV/parquet con filas a predecir")
    ap.add_argument("--out", type=str, default="reports/batch_predictions.csv")
    args = ap.parse_args()

    cfg = get_config()
    model, pp, sel_cols = load_model_artifacts(cfg)
    h = cfg["target"]["primary_horizon"]

    if args.date:
        raw = load_raw(cfg)
        clean = build_features(
            pd.concat([raw[raw["date"] <= args.date].tail(400), raw[raw["date"] == args.date]]),
            cfg, raw)
        # simplificado: usar última fila del dataset de features completo
        feats = pd.read_parquet("data/processed/features.parquet")
        row = feats[feats["date"] <= args.date].iloc[-1]
        pred = predict_row(row, model, pp, sel_cols, h)
        print(f"Predicción gold_spot en {args.date} +{h}d: {pred:.2f} USD/oz")
    elif args.csv:
        df = pd.read_parquet(args.csv) if args.csv.endswith(".parquet") else pd.read_csv(args.csv)
        df["prediction"] = df.apply(lambda r: predict_row(r, model, pp, sel_cols, h), axis=1)
        df.to_csv(args.out, index=False)
        print(f"Predicciones guardadas en {args.out} ({len(df)} filas)")
    else:
        ap.error("Indique --date o --csv")


if __name__ == "__main__":
    main()

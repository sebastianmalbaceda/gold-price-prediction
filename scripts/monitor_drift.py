"""Script de monitorización de drift (fase 23).

Compara la distribución de features de una ventana reciente contra la de
entrenamiento (KS test) y emite un JSON con alertas.

Uso:
    python scripts/monitor_drift.py --window 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from src.config import get_config  # noqa: E402
from src.data.split import drop_warmup, temporal_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=60, help="Últimos N días a comparar contra train")
    args = ap.parse_args()

    cfg = get_config()
    feats = pd.read_parquet("data/processed/features.parquet")
    feats = drop_warmup(feats, warmup=260)
    parts = temporal_split(feats, cfg)

    with open(cfg["model"]["feature_list_path"]) as f:
        sel_cols = json.load(f)

    train = pd.concat([parts["train"], parts["val"]]).sort_values("date")
    recent = feats.tail(args.window)

    alerts = []
    for c in sel_cols:
        a, b = train[c].dropna(), recent[c].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        stat, p = ks_2samp(a, b)
        if p < 0.01:
            alerts.append({"feature": c, "ks": round(float(stat), 3), "p": round(float(p), 4)})

    out = {
        "window_days": int(args.window),
        "n_features_checked": len(sel_cols),
        "n_drift_alerts": len(alerts),
        "alerts": alerts,
    }
    with open("reports/drift_report.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Features con drift: {len(alerts)}/{len(sel_cols)}")
    for a in alerts[:15]:
        print(" ", a)


if __name__ == "__main__":
    main()

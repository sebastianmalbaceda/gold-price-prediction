"""Monitorizacion de drift de features mediante prueba KS.

Uso:
    python scripts/monitor_drift.py --window 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from src.config import get_config, path_from_root  # noqa: E402
from src.data.split import drop_warmup, temporal_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Detecta drift de features frente a train+val")
    ap.add_argument("--window", type=int, default=60, help="Ultimos N dias a comparar contra train")
    ap.add_argument("--alpha", type=float, default=0.01, help="Nivel de alerta KS")
    args = ap.parse_args()
    if args.window < 2:
        ap.error("--window debe ser >= 2")
    if not 0 < args.alpha < 1:
        ap.error("--alpha debe estar entre 0 y 1")

    cfg = get_config()
    features_path = path_from_root("data/processed/features.parquet")
    if not features_path.is_file():
        raise SystemExit(f"No existe el conjunto de features: {features_path}")
    feats = pd.read_parquet(features_path)
    feats = drop_warmup(feats, warmup=260)
    if args.window > len(feats):
        ap.error(f"--window no puede superar las {len(feats)} filas disponibles")
    parts = temporal_split(feats, cfg)

    feature_path = Path(cfg["model"]["feature_list_path"])
    try:
        sel_cols = json.loads(feature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"No se pudo leer la lista de features: {exc}") from exc
    missing = [column for column in sel_cols if column not in feats.columns]
    if missing:
        raise SystemExit(f"Features ausentes del dataset: {missing[:10]}")

    train = pd.concat([parts["train"], parts["val"]]).sort_values("date")
    recent = feats.sort_values("date").tail(args.window)
    alerts = []
    checked = 0
    for column in sel_cols:
        a = pd.to_numeric(train[column], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(recent[column], errors="coerce").dropna().to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 10 or len(b) < 10:
            continue
        checked += 1
        statistic, p_value = ks_2samp(a, b)
        if p_value < args.alpha:
            alerts.append(
                {
                    "feature": column,
                    "ks": round(float(statistic), 3),
                    "p": round(float(p_value), 6),
                }
            )

    out = {
        "window_days": int(args.window),
        "n_recent_rows": int(len(recent)),
        "alpha": float(args.alpha),
        "n_features_checked": checked,
        "n_drift_alerts": len(alerts),
        "alerts": alerts,
    }
    output = path_from_root("reports/drift_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Features con drift: {len(alerts)}/{checked}")
    for alert in alerts[:15]:
        print(" ", alert)


if __name__ == "__main__":
    main()

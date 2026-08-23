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
from src.utils import atomic_write_text  # noqa: E402


def benjamini_hochberg(p_values: list[float], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Control de la tasa de falsos descubrimientos (FDR) de Benjamini-Hochberg.

    Contrastar decenas de features con un ``alpha`` por hipotesis dispara los
    falsos positivos: con 83 contrastes independientes y ``alpha=0.01``, la
    probabilidad de al menos una alerta espuria es ``1 - 0.99**83 ~ 56 %``.
    BH ajusta los p-valores para que el ``alpha`` se interprete sobre la
    familia completa de contrastes.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Mascara booleana de rechazos y p-valores ajustados, en el orden de
        entrada.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    adjusted_sorted = np.minimum.accumulate((p[order] * n / ranks)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted <= alpha, adjusted


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
        ap.error(f"--window no puede superar las {int(len(feats))} filas disponibles")
    parts = temporal_split(feats, cfg)

    feature_path = Path(cfg["model"]["feature_list_path"])
    try:
        sel_cols = json.loads(feature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"No se pudo leer la lista de features: {exc}") from exc
    missing = [column for column in sel_cols if column not in feats.columns]
    if missing:
        raise SystemExit(f"Features ausentes del dataset: {missing[:10]}")

    reference = pd.concat([parts["train"], parts["val"]]).sort_values("date")
    recent = feats.sort_values("date").tail(args.window).copy()

    # La ventana reciente debe ser ESTRICTAMENTE posterior a la referencia. Si
    # hay pocos datos tras validacion, o la ventana es mayor que el tramo de
    # test, ambos grupos comparten observaciones y el contraste KS deja de
    # tener sentido: compararia una muestra consigo misma y ocultaria el drift.
    recent_start = recent["date"].min()
    reference = reference.loc[reference["date"] < recent_start]
    if len(reference) < 30:
        ap.error(
            "La ventana reciente se solapa con la referencia o deja menos de 30 "
            f"filas de referencia (inicio de ventana: {recent_start.date()}). "
            "Reduzca --window o amplie el historico."
        )

    tests = []
    for column in sel_cols:
        a = pd.to_numeric(reference[column], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(recent[column], errors="coerce").dropna().to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 10 or len(b) < 10:
            continue
        statistic, p_value = ks_2samp(a, b)
        # Se conserva el p-valor SIN redondear: redondear antes de decidir
        # puede dejar fuera un contraste que si supera el umbral.
        tests.append((column, float(statistic), float(p_value)))

    checked = len(tests)
    rejected, adjusted = benjamini_hochberg([p for _, _, p in tests], args.alpha)
    alerts = [
        {
            "feature": column,
            "ks": round(statistic, 3),
            "p": round(p_value, 6),
            "p_adjusted_bh": round(float(p_adj), 6),
        }
        for (column, statistic, p_value), is_drift, p_adj in zip(
            tests, rejected, adjusted, strict=True
        )
        if is_drift
    ]
    n_raw_alerts = int(sum(1 for _, _, p in tests if p < args.alpha))

    out = {
        "window_days": int(args.window),
        "n_recent_rows": int(len(recent)),
        "recent_start": str(recent_start.date()),
        "recent_end": str(recent["date"].max().date()),
        "reference_start": str(reference["date"].min().date()),
        "reference_end": str(reference["date"].max().date()),
        "n_reference_rows": int(len(reference)),
        "alpha": float(args.alpha),
        "multiple_testing_correction": "benjamini_hochberg_fdr",
        "n_features_checked": checked,
        "n_alerts_uncorrected": n_raw_alerts,
        "n_drift_alerts": len(alerts),
        "alerts": alerts,
    }
    output = path_from_root("reports/drift_report.json")
    atomic_write_text(json.dumps(out, indent=2, ensure_ascii=False), output)
    print(
        f"Features con drift (FDR-BH, alpha={args.alpha}): {len(alerts)}/{checked} "
        f"[sin corregir habrian sido {n_raw_alerts}]"
    )
    print(
        f"Referencia {out['reference_start']} -> {out['reference_end']} "
        f"({len(reference)} filas) | ventana {out['recent_start']} -> {out['recent_end']}"
    )
    for alert in alerts[:15]:
        print(" ", alert)


if __name__ == "__main__":
    main()

# %% [markdown]
# # Fase 17: Evaluación final bloqueada en TEST
#
# Protocolo estricto:
# 1. Cargar el **test intacto** (nunca usado hasta ahora).
# 2. Aplicar **solo transform** del preprocesador entrenado (sin refit).
# 3. Inferir con el modelo final congelado.
# 4. Calcular métricas finales y comparar contra baselines.
# 5. Documentar.
#
# **Regla:** si el test obligara a cambiar decisiones, habría que crear un
# nuevo test bloqueado. No es el caso aquí.
#
# ---

# %%
"""Bootstrap del path del proyecto (funciona desde cualquier cwd)."""
import sys
from pathlib import Path

def _find_root():
    p = Path.cwd()
    for _ in range(5):
        if (p / "configs" / "config.yaml").exists():
            return p
        p = p.parent
    return Path.cwd()

sys.path.insert(0, str(_find_root()))

"""Fase 17: evaluación final en test con artefactos congelados."""
import json
import numpy as np
import pandas as pd

from src.config import get_config
from src.data.split import temporal_split, drop_warmup
from src.evaluation.metrics import regression_metrics, metrics_table
from src.models.train_model import load_model_artifacts
from src.utils import save_json

cfg = get_config()
feats = pd.read_parquet("data/processed/features.parquet")
feats = drop_warmup(feats, warmup=260)
parts = temporal_split(feats, cfg)

model, pp, sel_cols = load_model_artifacts(cfg)
h = cfg["target"]["primary_horizon"]

te = parts["test"]
X_te = pp.transform(te[sel_cols].to_numpy(dtype=np.float64))
y_te = te[f"target_{h}"].to_numpy(dtype=np.float64)
y_pred = model.predict(X_te)

final = regression_metrics(y_te, y_pred, h)
print("=== MÉTRICAS FINALES EN TEST (bloqueado) ===")
print(f"MAE = {final['mae']:.2f} USD/oz")
print(f"RMSE = {final['rmse']:.2f} USD/oz")
print(f"sMAPE = {final['smape']:.3f}%")
print(f"R² = {final['r2']:.4f}")
print(f"Directional Accuracy = {final['directional_accuracy']:.2f}%")

# Comparación con baselines en el mismo test
last_price = parts["train"]["gold_spot"].iloc[-1]  # último valor de train+val
naive = np.full(len(y_te), last_price)
naive_m = regression_metrics(y_te, naive, h)
print("\n=== Baseline naive (último valor conocido) en test ===")
print(f"MAE = {naive_m['mae']:.2f} | sMAPE = {naive_m['smape']:.3f}% | DA = {naive_m['directional_accuracy']:.2f}%")

improvement = (1 - final["mae"] / naive_m["mae"]) * 100
print(f"\nGanancia vs naive: {improvement:.1f}% de reducción de MAE")

# Guardar predicciones y resultados
te_out = te[["date", "gold_spot", f"target_{h}"]].copy()
te_out["prediction"] = y_pred
te_out["residual"] = y_te - y_pred
te_out.to_csv("reports/test_predictions.csv", index=False)

save_json({"final_test": final, "naive": naive_m,
           "improvement_vs_naive_pct": improvement}, "test_results.json")

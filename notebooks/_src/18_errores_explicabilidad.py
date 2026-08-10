# %% [markdown]
# # Fase 18: Análisis de errores, explicabilidad e incertidumbre
#
# ## 18.1 Análisis de errores
# - Residuos en el tiempo: ¿eran predecibles? ¿se concentran en regímenes de alta volatilidad (2024)?
# - Errores por año y por nivel de precios.
# - Peores predicciones: sobre/subestimaciones extremas.
#
# ## 18.2 Explicabilidad
# - **Importancia de features** (modelo de árboles).
# - **SHAP** (muestra de test): qué mueve el precio predicho.
# - PDP para las variables más importantes.
#
# ## 18.3 Incertidumbre
# - **Intervalos por cuantiles de ensemble**: se entrena el mismo modelo con
#   varias semillas y se calcula el percentil 5-95 de las predicciones.
# - Cobertura empírica de los intervalos en test.
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

"""Fase 18: errores, SHAP e incertidumbre."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import get_config
from src.data.split import temporal_split, drop_warmup
from src.evaluation.metrics import regression_metrics
from src.models.train_model import load_model_artifacts
from src.utils import save_fig, save_json, set_seed

set_seed(42)
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
resid = y_te - y_pred

# --- 18.1 Errores ---
out = te[["date"]].copy()
out["y_true"], out["y_pred"], out["resid"] = y_te, y_pred, resid
out["year"] = out["date"].dt.year

fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes[0, 0].plot(out["date"], out["resid"], lw=0.5, alpha=0.7)
axes[0, 0].axhline(0, color="k", lw=0.8)
axes[0, 0].set_title("Residuos en el tiempo")
by_year = out.groupby("year")["resid"].agg(["mean", "std", "count"])
by_year.plot(y=["mean", "std"], ax=axes[0, 1], title="Error por año (media/desv.)")
axes[1, 0].hist(resid, bins=60)
axes[1, 0].set_title("Histograma de residuos")
worst = out.reindex(resid.abs().sort_values(ascending=False).index).head(10)
axes[1, 1].bar(range(len(worst)), worst["resid"].abs())
axes[1, 1].set_title("10 peores errores absolutos")
fig.tight_layout(); save_fig(fig, "error_analysis.png")

print("MAE por año:")
print(by_year["mean"].abs().round(2).to_string())
print("\nPeores 5 errores:")
print(worst[["date", "y_true", "y_pred", "resid"]].head().round(2).to_string(index=False))

# --- 18.2 SHAP (submuestra de 300) ---
sample_idx = np.random.default_rng(42).choice(len(X_te), size=min(300, len(X_te)), replace=False)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_te[sample_idx], check_additivity=False)
fig = plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_te[sample_idx], feature_names=sel_cols, show=False, max_display=20)
plt.title("SHAP - impacto en predicción (test)")
save_fig(fig, "shap_summary.png")

imp = pd.Series(np.abs(shap_values).mean(axis=0), index=sel_cols).sort_values(ascending=False)
print("\nTop 15 importancias SHAP:")
print(imp.head(15).round(4).to_string())
save_json(imp.head(30).round(5).to_dict(), "shap_importances.json")

# --- 18.3 Incertidumbre: ensemble de semillas ---
from xgboost import XGBRegressor
train_val = pd.concat([parts["train"], parts["val"]]).sort_values("date").reset_index(drop=True)
X_tv = pp.transform(train_val[sel_cols].to_numpy(dtype=np.float64))
y_tv = train_val[f"target_{h}"].to_numpy(dtype=np.float64)

preds = []
for seed in (7, 13, 21, 99, 123):
    m = XGBRegressor(**model.get_params(), random_state=seed, n_jobs=-1)
    m.fit(X_tv, y_tv)
    preds.append(m.predict(X_te))
preds = np.vstack(preds)
lo, hi = np.percentile(preds, [5, 95], axis=0)
coverage = np.mean((y_te >= lo) & (y_te <= hi))
print(f"\nCobertura empírica intervalo P5-P95: {coverage:.1%}")
print(f"Amplitud media del intervalo: {np.mean(hi - lo):.2f} USD/oz")

fig, ax = plt.subplots(figsize=(14, 5))
idx = np.arange(min(500, len(y_te)))
ax.plot(y_te[idx], "k-", lw=1, label="Real")
ax.plot(y_pred[idx], "b-", lw=1, label="Predicción")
ax.fill_between(idx, lo[idx], hi[idx], alpha=0.25, label="P5-P95")
ax.legend(); ax.set_title("Predicción con banda de incertidumbre (500 días de test)")
save_fig(fig, "uncertainty_band.png")

save_json({"coverage_p5_p95": float(coverage),
           "mean_interval_width": float(np.mean(hi - lo))}, "uncertainty.json")

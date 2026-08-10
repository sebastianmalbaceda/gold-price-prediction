# %% [markdown]
# # Fase 11-13: Baselines, modelado inicial y validación
#
# ## 11. Baselines
# Para forecasting, los baselines obligatorios son:
# - **Naive** (último valor): `y_pred = gold_spot(t)` → a todos los horizontes.
# - **Naive estacional** (mismo día de la semana anterior, h=5) / valor de hace 21 días (h=21).
# - **Media móvil** 21 días.
# - **ARIMA** (auto_arima-ish con statsmodels) como baseline estadístico clásico.
#
# ## 12. Modelado inicial
# Secuencia de complejidad:
# 1. Ridge (lineal regularizado) sobre features estándar.
# 2. Random Forest (no lineal).
# 3. XGBoost / LightGBM / CatBoost (boosting).
# 4. MLP (deep learning básico).
#
# ## 13. Validación
# - TimeSeriesSplit(5, gap=21) sobre train+val; preprocesador dentro de cada fold.
# - Métricas: MAE, RMSE, sMAPE, R², Directional Accuracy.
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

"""Fase 11-13: baselines y primeros modelos con CV temporal."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import get_config
from src.data.split import get_temporal_splitter, temporal_split, drop_warmup
from src.evaluation.metrics import regression_metrics, metrics_table
from src.features.build_features import get_feature_columns
from src.models.pipeline import fit_preprocessor
from src.models.train_model import get_X_y
from src.utils import set_seed

set_seed(42)
cfg = get_config()
feats = pd.read_parquet("data/processed/features.parquet")
feats = drop_warmup(feats, warmup=260)
parts = temporal_split(feats, cfg)
train_val = pd.concat([parts["train"], parts["val"]]).sort_values("date").reset_index(drop=True)
with open("models/feature_list.json") as f:
    import json
    sel_cols = json.load(f)

h = cfg["target"]["primary_horizon"]
tscv = get_temporal_splitter(cfg)

def cv_evaluate(model, label, horizon=h):
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train_val)):
        tr, va = train_val.iloc[tr_idx], train_val.iloc[va_idx]
        pp = fit_preprocessor(tr[sel_cols].to_numpy(dtype=np.float64))
        Xtr = pp.transform(tr[sel_cols].to_numpy(dtype=np.float64))
        Xva = pp.transform(va[sel_cols].to_numpy(dtype=np.float64))
        m = model.__class__(**model.get_params()) if hasattr(model, "get_params") else model
        m.fit(Xtr, tr[f"target_{h}"].to_numpy())
        yp = m.predict(Xva)
        rows.append(regression_metrics(va[f"target_{h}"].to_numpy(), yp, h))
    res = pd.DataFrame(rows)
    print(f"[CV] {label:20s} MAE={res['mae'].mean():8.2f} ±{res['mae'].std():6.2f} | "
          f"RMSE={res['rmse'].mean():8.2f} | sMAPE={res['smape'].mean():5.2f}% | "
          f"DA={res['directional_accuracy'].mean():5.1f}%")
    return res

# --- Baselines (sin features) ---
def baseline_naive(train_val, h):
    """Predice el último valor conocido: gold_spot(t)."""
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train_val)):
        va = train_val.iloc[va_idx]
        # El 'último valor' = gold_spot del día anterior al inicio del fold val
        last = train_val.iloc[tr_idx]["gold_spot"].iloc[-1]
        yp = np.full(len(va), last)
        rows.append(regression_metrics(va[f"target_{h}"].to_numpy(), yp, h))
    res = pd.DataFrame(rows)
    print(f"[CV] {'Naive (último valor)':20s} MAE={res['mae'].mean():8.2f} | "
          f"RMSE={res['rmse'].mean():8.2f} | sMAPE={res['smape'].mean():5.2f}%")
    return res

naive_res = baseline_naive(train_val, h)

# --- Baselines ML ---
ridge_res = cv_evaluate(Ridge(alpha=1.0), "Ridge")
rf_res = cv_evaluate(RandomForestRegressor(n_estimators=200, max_depth=12,
                                           min_samples_leaf=5, n_jobs=-1,
                                           random_state=42), "RandomForest")
xgb_res = cv_evaluate(XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                   subsample=0.85, colsample_bytree=0.7,
                                   random_state=42, n_jobs=-1), "XGBoost")

print("\n=== Resumen CV (métricas medias por fold) ===")
summary = pd.concat([
    naive_res.mean().rename("Naive"),
    ridge_res.mean().rename("Ridge"),
    rf_res.mean().rename("RandomForest"),
    xgb_res.mean().rename("XGBoost"),
], axis=1).T[["mae", "rmse", "smape", "directional_accuracy"]]
print(summary.round(3).to_string())
summary.to_csv("reports/cv_baselines_initial.csv")

# %% [markdown]
# # Fase 15-16: Selección de modelo y entrenamiento final
#
# ## 15. Model selection
# - Se comparan las familias **solo con CV/validation** (test intacto).
# - Criterios: MAE medio, estabilidad (std entre folds), sMAPE, directional accuracy,
#   coste de entrenamiento, interpretabilidad y facilidad de despliegue.
# - Se actualiza `configs/params.yaml` con los mejores hiperparámetros.
#
# ## 16. Entrenamiento final
# - Decisiones congeladas (features, preprocesador, modelo, hiperparámetros).
# - Se reentrena con **train + validation** (fase de selección terminada).
# - Se guardan: modelo, preprocesador, lista de features, configuración.
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

"""Fase 15-16: comparativa final en validation y entrenamiento con train+val."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.config import get_config
from src.data.split import temporal_split, drop_warmup
from src.evaluation.metrics import regression_metrics, metrics_table
from src.models.pipeline import fit_preprocessor
from src.models.train_model import save_model_artifacts
from src.utils import set_seed

set_seed(42)
cfg = get_config()
feats = pd.read_parquet("data/processed/features.parquet")
feats = drop_warmup(feats, warmup=260)
parts = temporal_split(feats, cfg)
with open("models/feature_list.json") as f:
    sel_cols = json.load(f)

h = cfg["target"]["primary_horizon"]
tr, va = parts["train"], parts["val"]
Xtr0 = tr[sel_cols].to_numpy(dtype=np.float64)
ytr0 = tr[f"target_{h}"].to_numpy(dtype=np.float64)
Xva = va[sel_cols].to_numpy(dtype=np.float64)
yva = va[f"target_{h}"].to_numpy(dtype=np.float64)

def evaluate_on_val(model):
    pp = fit_preprocessor(Xtr0)
    model.fit(pp.transform(Xtr0), ytr0)
    yp = model.predict(pp.transform(Xva))
    return pp, regression_metrics(yva, yp, h)

with open("reports/tuning_results.json") as f:
    tuned = json.load(f)

families = {
    "ridge": Ridge(**tuned["ridge"]["params"]),
    "rf": RandomForestRegressor(**tuned["rf"]["params"], random_state=42, n_jobs=-1),
    "xgb": XGBRegressor(**tuned["xgb"]["params"], random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(**tuned["lgbm"]["params"], random_state=42, n_jobs=-1, verbose=-1),
    "cat": CatBoostRegressor(**tuned["cat"]["params"], random_seed=42, verbose=0),
}

rows, fitted = [], {}
for name, model in families.items():
    pp, m = evaluate_on_val(model)
    fitted[name] = (model, pp)
    print(f"[val] {name:6s} MAE={m['mae']:8.2f} RMSE={m['rmse']:8.2f} "
          f"sMAPE={m['smape']:5.2f}% R2={m['r2']:.4f} DA={m['directional_accuracy']:.1f}%")
    rows.append({"model": name, **m})

val_table = metrics_table(rows)
print("\n=== Comparativa en VALIDATION (h=1) ===")
print(val_table.round(3).to_string(index=False))
val_table.to_csv("reports/validation_comparison.csv", index=False)

# --- Decisión: se elige el mejor por MAE con buen equilibrio ---
best_name = val_table.sort_values("mae").iloc[0]["model"]
print(f"\nModelo seleccionado: {best_name}")

# 16. Entrenamiento final con train + val
train_val = pd.concat([tr, va]).sort_values("date").reset_index(drop=True)
X_tv = train_val[sel_cols].to_numpy(dtype=np.float64)
y_tv = train_val[f"target_{h}"].to_numpy(dtype=np.float64)
pp = fit_preprocessor(X_tv)
final_model = families[best_name][0]
final_model.fit(pp.transform(X_tv), y_tv)

save_model_artifacts(final_model, pp, sel_cols,
                     metrics={"selected_model": best_name,
                              "val_metrics": rows,
                              "horizon": h,
                              "config": cfg["model"]["version"]}, cfg=cfg)
print(f"\nArtefactos guardados. Modelo final: {best_name}")

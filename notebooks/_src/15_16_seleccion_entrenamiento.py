# %% [markdown]
# # Fase 15-16: Selección de modelo y entrenamiento final
#
# ## 15. Model selection
#
# - Se comparan las familias **solo con CV/validation** (test intacto).
# - Criterios: MAE medio, estabilidad (std entre folds), sMAPE, directional accuracy,
#   coste de entrenamiento, interpretabilidad y facilidad de despliegue.
# - Se actualiza `configs/params.yaml` con los mejores hiperparámetros.
#
# ## 16. Entrenamiento final
#
# - Decisiones congeladas (features, preprocesador, modelo, hiperparámetros).
# - Se reentrena con **train + validation** (fase de selección terminada).
# - Se guardan: modelo, preprocesador, lista de features, configuración.
#
# ---

# %%
"""Configuración del notebook (raíz del proyecto)."""
import sys
from pathlib import Path

def _find_root():
    p = Path.cwd()
    for _ in range(5):
        if (p / "configs" / "config.yaml").exists():
            return p
        p = p.parent
    return Path.cwd()

ROOT = _find_root()
sys.path.insert(0, str(ROOT))
from src.utils import set_seed
set_seed(42)

# %%
"""Carga de datos y de los resultados del tuning.

Leemos `reports/tuning_results.json` (fase 14) para instanciar cada familia
con sus mejores hiperparámetros. Comprobamos que el archivo existe.
"""
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
from src.utils import path_from_root

cfg = get_config()
feats = pd.read_parquet(path_from_root("data/processed/features.parquet"))
feats = drop_warmup(feats, warmup=260)
parts = temporal_split(feats, cfg)
with open(path_from_root("models", "feature_list.json")) as f:
    sel_cols = json.load(f)

with open(path_from_root("reports", "tuning_results.json")) as f:
    tuned = json.load(f)

h = cfg["target"]["primary_horizon"]
tr, va = parts["train"], parts["val"]
Xtr0 = tr[sel_cols].to_numpy(dtype=np.float64)
ytr0 = tr[f"target_{h}"].to_numpy(dtype=np.float64)
Xva = va[sel_cols].to_numpy(dtype=np.float64)
yva = va[f"target_{h}"].to_numpy(dtype=np.float64)
print(f"train: {len(tr)} | val: {len(va)} | features: {len(sel_cols)}")

# %%
"""Evaluación de cada familia en validation.

Cada modelo se entrena con train (preprocesador ajustado solo con train)
y se evalúa en validation. Esta es la comparativa final para elegir.
"""
families = {
    "ridge": Ridge(**tuned["ridge"]["params"]),
    "rf": RandomForestRegressor(**tuned["rf"]["params"], random_state=42, n_jobs=-1),
    "xgb": XGBRegressor(**tuned["xgb"]["params"], random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(**tuned["lgbm"]["params"], random_state=42, n_jobs=-1, verbose=-1),
    "cat": CatBoostRegressor(**tuned["cat"]["params"], random_seed=42, verbose=0),
}

def evaluate_on_val(model):
    pp = fit_preprocessor(Xtr0)
    model.fit(pp.transform(Xtr0), ytr0)
    yp = model.predict(pp.transform(Xva))
    return pp, regression_metrics(yva, yp, h)

rows = []
for name, model in families.items():
    pp, m = evaluate_on_val(model)
    print(f"[val] {name:6s} MAE={m['mae']:8.2f} RMSE={m['rmse']:8.2f} "
          f"sMAPE={m['smape']:5.2f}% R2={m['r2']:.4f} DA={m['directional_accuracy']:.1f}%")
    rows.append({"model": name, **m})

val_table = metrics_table(rows)
print("\n=== Comparativa en VALIDATION (h=1) ===")
print(val_table.round(3).to_string(index=False))
val_table.to_csv(path_from_root("reports", "validation_comparison.csv"), index=False)

# %%
"""Decisión del modelo final.

Elegimos por MAE con equilibrio de estabilidad/coste/interpretabilidad.
Ridge domina claramente (MAE ~37 vs ~200 de los árboles). Actualizamos
`configs/params.yaml` con la decisión documentada.
"""
best_name = val_table.sort_values("mae").iloc[0]["model"]
print(f"Modelo seleccionado: {best_name}")

import yaml
params_path = path_from_root("configs", "params.yaml")
params = yaml.safe_load(open(params_path, encoding="utf-8"))
params["model_selection"] = {
    "selected": best_name,
    "reason": ("Mejor MAE en CV temporal y validation. Lineal regularizado "
               "generaliza mejor en niveles no estacionarios."),
    "val_mae_cv": float(tuned[best_name]["best_mae_cv"]),
    "val_mae_holdout": float(val_table.sort_values("mae").iloc[0]["mae"]),
}
with open(params_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(params, f, allow_unicode=True, sort_keys=False)
print(f"configs/params.yaml actualizado con la decisión: {best_name}")

# %%
"""Entrenamiento final con train + validation.

Una vez congelada la decisión, reentrenamos con todo el dato disponible
(train+val) para maximizar la muestra. Se guardan los artefactos: modelo,
preprocesador y lista de features (la API los usará).
"""
train_val = pd.concat([tr, va]).sort_values("date").reset_index(drop=True)
X_tv = train_val[sel_cols].to_numpy(dtype=np.float64)
y_tv = train_val[f"target_{h}"].to_numpy(dtype=np.float64)
pp = fit_preprocessor(X_tv)
final_model = families[best_name]  # el modelo ya está entrenado en la celda 3
final_model.fit(pp.transform(X_tv), y_tv)

save_model_artifacts(final_model, pp, sel_cols,
                     metrics={"selected_model": best_name,
                              "val_metrics": rows,
                              "horizon": h,
                              "config": cfg["model"]["version"]}, cfg=cfg)
print(f"\nArtefactos guardados. Modelo final: {best_name}")

# %%
"""Visualización de la comparativa final.

Gráfico de barras con MAE y R² por familia en validation. Se aprecia el
contraste entre el modelo lineal y los ensembles.
"""
import matplotlib.pyplot as plt

from src.utils import save_fig

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
names = val_table["model"].tolist()
axes[0].bar(names, val_table["mae"], color=["#2c7fb8" if n == best_name else "#999" for n in names])
axes[0].set_title("MAE en validation (USD/oz)")
axes[0].tick_params(axis="x", rotation=20)
axes[1].bar(names, val_table["r2"], color=["#2c7fb8" if n == best_name else "#999" for n in names])
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_title("R² en validation")
axes[1].tick_params(axis="x", rotation=20)
fig.tight_layout()
save_fig(fig, "model_selection.png")

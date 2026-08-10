# %% [markdown]
# # Fase 8-10: Preprocessing, feature engineering y feature selection
#
# ## 8. Preprocessing seguro
# - Limpieza base (fases 2-3): solo días hábiles, forward-fill, features con cobertura suficiente → `data/interim`.
# - **Escalado (RobustScaler) ajustado SOLO con train** y aplicado con `transform` a val/test.
# - No hay imputación con estado: el ffill ya usa solo pasado.
#
# ## 9. Feature engineering (serie temporal)
# - **Lags del target:** 1, 2, 3, 5, 10, 21 (días hábiles).
# - **Retornos logarítmicos** del target a esos lags.
# - **Rolling:** retorno y media/desv. móvil a 5, 21, 63, 126 días.
# - **Exógenas:** lag 1 de cada una + retorno log del lag (siempre positivas).
# - **Calendario:** año, mes, día de semana, trimestre, día del año.
# - **Indicadores de ausencia:** si la exógena faltaba en origen, marca 1/0.
#
# Todas las ventanas miran **solo hacia atrás** (sin futuro) → sin leakage.
#
# ## 10. Feature selection
# - **Reglas:** se eliminan features excluidas por cobertura (config) y el target/date.
# - **Filtro:** correlación de Pearson; si dos features correlacionan >0.98, se elimina una.
# - **Embedded (fase 15):** importancia de XGBoost para el informe.
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

"""Fase 8-10: preprocesamiento, features y selección por correlación."""
import numpy as np
import pandas as pd

from src.config import get_config
from src.data.split import drop_warmup, temporal_split
from src.features.build_features import get_feature_columns
from src.models.pipeline import fit_preprocessor

cfg = get_config()
feats = pd.read_parquet("data/processed/features.parquet")
feats = drop_warmup(feats, warmup=260)
parts = temporal_split(feats, cfg)

feat_cols = get_feature_columns(feats, cfg["target"]["horizons"])
print("Features totales:", len(feat_cols))

X_train = parts["train"][feat_cols].to_numpy(dtype=np.float64)
print("\nNaN en train (features):", np.isnan(X_train).sum())
assert not np.isnan(X_train).any(), "Hay NaNs en train"

# 8. Preprocesador ajustado SOLO con train
pp = fit_preprocessor(X_train)
print("Preprocesador ajustado en train:", type(pp[0]).__name__)

# 10.2 Selección por correlación (redundancia > 0.98)
corr = parts["train"][feat_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high = [(upper.columns[i], upper.columns[j], round(upper.iloc[i, j], 3))
        for i in range(len(upper)) for j in range(i + 1, len(upper))
        if upper.iloc[i, j] > 0.98]
print(f"\nPares con correlación > 0.98: {len(high)}")
for a, b, v in high[:15]:
    print(f"  {a} ~ {b}: {v}")

# Eliminar el miembro de cada par con menor varianza (redundancia)
drop_set = set()
for a, b, _ in high:
    va, vb = parts["train"][a].var(), parts["train"][b].var()
    drop_set.add(a if va <= vb else b)
sel_cols = [c for c in feat_cols if c not in drop_set]
print(f"\nFeatures tras filtro de redundancia: {len(feat_cols)} -> {len(sel_cols)}")
print("Eliminadas:", sorted(drop_set))

# Guardar selección para fases siguientes
import json
from pathlib import Path
Path("models").mkdir(exist_ok=True)
with open("models/feature_list.json", "w") as f:
    json.dump(sel_cols, f, indent=2)
print("Lista de features guardada en models/feature_list.json")

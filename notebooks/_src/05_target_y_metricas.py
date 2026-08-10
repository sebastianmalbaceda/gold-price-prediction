# %% [markdown]
# # Fase 5-6: Target, features, métricas y criterios de éxito
#
# ## 5. Target y variables
# - **Target:** `gold_spot` (precio spot del oro, USD/oz) a horizontes h ∈ {1, 5, 21} días hábiles.
# - **Tipo de tarea:** regresión (forecasting de niveles).
# - **Etiquetado:** valor futuro real observado `gold_spot(t+h)` (retraso de etiqueta = h días).
# - **Variables:** ver diccionario de datos en `docs/data_card.md`.
# - **Riesgo de leakage:** mitigado con lags/rollings calculados solo con pasado y targets desplazados a futuro.
#
# ## 6. Métricas y criterios de éxito
# ### 6.1 Métrica primaria (regresión/forecasting)
# - **MAE** (USD/oz) → interpretable para el usuario.
# - **RMSE** → penaliza errores grandes (volatilidad).
# - **sMAPE** → error porcentual simétrico, comparable entre horizontes.
# - **R²** complementario.
#
# ### 6.2 Métricas secundarias
# - **Directional Accuracy** (acierto del signo del cambio) — crítica para uso en trading.
# - Análisis de residuos, error por segmento temporal.
#
# ### 6.3 Umbral de decisión
# - No aplica umbral de clasificación (regresión). Política de abstención: si el modelo estima incertidumbre alta (intervalos por cuantiles de ensemble), se puede rechazar la predicción.
#
# ### 6.4 Criterios de aceptación
# | Métrica | Baseline naive | Objetivo (test) |
# |---|---|---|
# | MAE h=1 (USD/oz) | ~último precio | < naive |
# | sMAPE h=1 | ~1.0% | < 1.0% |
# | sMAPE h=21 | ~4-5% | < naive |
# | Directional h=1 | ~50% | > 52% |
#
# La regla de oro: **ganancia mínima sobre el baseline naive** en MAE/sMAPE en test.
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

"""Fase 5-6: construcción del feature set y verificación de targets."""
import numpy as np
import pandas as pd

from src.config import get_config
from src.data.load_data import build_interim, load_raw
from src.features.build_features import (build_features, get_feature_columns,
                                         make_targets)

cfg = get_config()
raw = load_raw(cfg)
clean = build_interim(cfg)

feats = build_features(clean, cfg, raw)
feats = make_targets(feats, cfg["target"]["horizons"])
print("Features shape:", feats.shape)

feat_cols = get_feature_columns(feats, cfg["target"]["horizons"])
print("Nº features:", len(feat_cols))
print("\nPrimeras 10 features:", feat_cols[:10])

# Verificación anti-leakage: el target_1 en t debe ser gold_spot en t+1
chk = feats[["date", "gold_spot", "target_1"]].dropna().head(5)
print("\nComprobación de desplazamiento temporal:")
print(chk.to_string(index=False))

# Guardar para fases posteriores
feats.to_parquet("data/processed/features.parquet", index=False)
print("\nGuardado en data/processed/features.parquet")

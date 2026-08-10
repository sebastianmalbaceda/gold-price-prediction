# %% [markdown]
# # Fase 0-1: Problema, contexto y diseño técnico
#
# **Gold Price Prediction** — Serie temporal financiera (forecasting de regresión).
#
# ## 0. Problema y contexto
#
# ### 0.1 Contexto
# - **Problema de negocio:** estimar el precio spot del oro (*gold_spot*, USD/oz) a 1, 5 y 21 días hábiles vista.
# - **Usuario final:** analista de mercados / gestor de cartera que necesita una referencia cuantitativa de precios futuros.
# - **Proceso actual sin IA:** heurísticas, reglas técnicas (soportes/resistencias) y juicio de analistas.
# - **Decisión que apoya:** sizing de posiciones, timing de compra/venta, *stress testing* de carteras y alertas.
# - **Valor esperado:** referencia objetiva + métricas de incertidumbre; **coste**: los errores de predicción se traducen en P&L, pero este es un proyecto educativo/analítico (sin trading automático).
# - **Alternativas no ML:** *naive* (último precio), media móvil, ARIMA.
#
# ### 0.2 Formulación técnica
# - **Tipo:** Forecasting de regresión univariante con covariables exógenas (múltiples horizontes).
# - **Unidad de predicción:** día hábil (mercado COMEX/NYMEX).
# - **Entrada disponible en el momento de predecir `t`:** precio del oro y 59 exógenas en `t` (y pasadas) — nunca futuras.
# - **Salida esperada:** precio del oro en `t+h`.
# - **Horizonte:** h ∈ {1, 5, 21} días hábiles. Frecuencia de inferencia: diaria (batch).
# - **Límites:** latencia < 1 s por predicción, memoria < 2 GB, interpretabilidad media-alta (SHAP).
#
# ## 1. Diseño técnico y reproducibilidad
#
# ### 1.1 Estructura del proyecto
# ```
# gold-price-prediction-v2/
# ├── configs/            # config.yaml (rutas, split, features) + params.yaml (hiperparámetros)
# ├── data/raw|interim|processed/
# ├── notebooks/          # 01_... a 12_... (una fase por notebook)
# ├── src/                # data, features, models, evaluation, api
# ├── models/             # artefactos .joblib
# ├── reports/figures/    # gráficos y métricas
# ├── tests/              # pytest
# └── docs/               # model card, informe técnico, manual API
# ```
#
# ### 1.2 Reproducibilidad
# - Semilla global 42 (`src/utils.set_seed`).
# - Python 3.11.9, dependencias fijadas en `requirements.txt`.
# - Dataset versionado: `data/raw/gold-price-prediction-dataset.csv` (inmutable).
# - Configuración de split y features en `configs/config.yaml` (separada del código).
# - Experimentos registrados en `reports/` (JSON + figuras).
#
# ## Resumen de la fase
# | Concepto | Valor |
# |---|---|
# | Problema | Regresión de precio spot del oro a 1/5/21 días |
# | Unidad | Día hábil |
# | Métrica primaria | RMSE / MAE en USD, sMAPE |
# | Split | Temporal estricto (train 2000-2019, val 2020-2022, test 2023-2025) |
# | Entorno | Python 3.11 + scikit-learn/XGBoost/LightGBM/CatBoost |
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

"""Fase 0-1: verificación del entorno y estructura."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__, "| numpy:", np.__version__, "| sklearn:", sklearn.__version__)
print("Proyecto:", Path.cwd())

from src.config import get_config, get_params

cfg = get_config()
print("\nVentana de datos:", cfg["data"]["start_date"], "->", cfg["data"]["end_date"])
print("Split:", cfg["split"])
print("Horizontes:", cfg["target"]["horizons"])
print("Modelos configurados:", list(get_params()["models"].keys()))

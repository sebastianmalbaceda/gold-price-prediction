# %% [markdown]
# # Fase 4: EDA y entendimiento del dominio
#
# ## 4.1 EDA del target (gold_spot)
#
# - Serie larga con **tendencia alcista estructural** (1979 → 2025) y **cambios de régimen** (2008, 2011, 2020, 2024).
# - Precio en niveles: NO es estacionaria (ADF con p>0.05). Los retornos logarítmicos sí lo son.
# - Volatilidad agrupada (*volatility clustering*), típica de activos financieros.
#
# ## 4.2 EDA de features
#
# - Correlación esperada con el dólar (DXY) negativa y con tipos reales negativa.
# - Muchas features correlacionadas entre sí (metales, índices) → redundancia.
#
# ## 4.3 EDA específico de series temporales
#
# - Tendencia, estacionalidad semanal débil, autocorrelación alta en niveles y casi nula en retornos.
#
# ## 4.4 Hipótesis
#
# 1. Los **lags y retornos** del propio oro dominarán la predicción a 1 día.
# 2. DXY, tipos reales y VIX aportan señal exógena a horizontes medios.
# 3. La métrica más informativa para el usuario es **MAE/sMAPE** (USD por onza).
# 4. El split debe ser **temporal estricto**; los retornos permiten un modelo más estable.
# 5. Riesgo principal: *non-stationarity* → los modelos de árboles con lags gestionan niveles; los lineales requieren retornos.
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
"""Carga y limpieza del dataset para EDA.

Reutilizamos el pipeline de limpieza (`clean_daily_series`) que: recorta a
la ventana 2000-2025, elimina fines de semana, excluye features con
cobertura insuficiente y hace forward-fill (solo pasado).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.config import get_config
from src.data.load_data import clean_daily_series, load_raw
from src.utils import save_fig

cfg = get_config()
raw = load_raw(cfg)
df = clean_daily_series(raw, cfg)
print("Filas tras limpieza:", len(df), "| columnas:", df.shape[1])
print("Rango:", df["date"].min().date(), "->", df["date"].max().date())

# %%
"""Estadísticos descriptivos del target.

Media, desviación, cuartiles, asimetría y curtosis del precio del oro en la
ventana 2000-2025. La asimetría positiva y la alta curtosis son típicas de
activos con rallies (cola derecha).
"""
g = df["gold_spot"]
print(g.describe().round(2).to_string())
print("Asimetría:", round(stats.skew(g), 2), "| Curtosis:", round(stats.kurtosis(g), 2))

# %%
"""Gráfico del target: niveles y retornos logarítmicos.

- Panel superior izquierdo: precio en niveles (tendencia alcista + regímenes).
- Panel superior derecho: retorno log diario (volatilidad variable).
- Paneles inferiores: histogramas de niveles y retornos (cola ancha en retornos).
"""
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes[0, 0].plot(df["date"], g, lw=0.7)
axes[0, 0].set_title("gold_spot (USD/oz) - niveles")
axes[0, 0].set_ylabel("USD/oz")

rets = np.log(g).diff().dropna()
axes[0, 1].plot(df["date"].iloc[1:], rets, lw=0.4, alpha=0.7)
axes[0, 1].set_title("Retorno log diario")

axes[1, 0].hist(rets, bins=80)
axes[1, 0].set_title("Histograma retornos")

axes[1, 1].hist(g, bins=60)
axes[1, 1].set_title("Histograma niveles")
fig.tight_layout()
save_fig(fig, "eda_target.png")

# %%
"""Análisis de estacionariedad (test ADF).

El test Augmented Dickey-Fuller contrasta H0: la serie tiene raíz unitaria
(no estacionaria). Si p < 0.05 rechazamos H0 (serie estacionaria).
En niveles el oro NO es estacionario (p alto); en retornos SÍ (p < 0.01).
"""
from statsmodels.tsa.stattools import adfuller

for name, s in [("niveles", g.dropna()), ("retornos", rets.dropna())]:
    adf, p = adfuller(s)[:2]
    print(f"ADF {name}: stat={adf:.2f}, p={p:.4f} -> "
          f"{'estacionaria' if p < 0.05 else 'NO estacionaria'}")

# %%
"""Autocorrelación: niveles vs retornos.

- ACF de niveles: decae muy lentamente (persistencia extrema, ~1 en lag 1).
- ACF de retornos: ruido blanco (sin autocorrelación) → el signo diario es
  casi impredecible; de ahí la importancia de los lags para el nivel.
"""
from pandas.plotting import autocorrelation_plot

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
autocorrelation_plot(g, ax=axes[0])
axes[0].set_title("ACF niveles")
autocorrelation_plot(rets, ax=axes[1])
axes[1].set_title("ACF retornos")
fig.tight_layout()
save_fig(fig, "eda_acf.png")

# %%
"""Correlaciones con el target (niveles y retornos).

- Niveles: el oro correlaciona alto con plata, metales e índices de materias
  primas; negativo con el dólar (DXY) y tipos reales.
- Retornos: correlaciones mucho más débiles (señal diaria pequeña).
"""
exog = [c for c in df.columns if c not in ("date", "gold_spot")]
corr_levels = df[exog].corrwith(g).sort_values(ascending=False)
print("Top correlaciones con gold_spot (niveles):")
print(corr_levels.head(8).round(3).to_string())
print("\nPeores correlaciones (niveles):")
print(corr_levels.tail(5).round(3).to_string())

ret_df = df.copy()
ret_df["gold_ret"] = rets
corr_rets = ret_df[exog].corrwith(ret_df["gold_ret"])
print("\nTop correlaciones retorno oro vs retornos exógenas:")
print(corr_rets.sort_values(ascending=False).head(10).round(3).to_string())

# %%
"""Estacionalidad mensual y volatilidad agrupada.

- Precio medio por mes: estacionalidad débil (el oro no tiene un mes claramente
  dominante, aunque octubre-enero suelen ser fuertes).
- Volatilidad anualizada (ventana 21d): claramente agrupada (2008, 2011-13,
  2020, 2024-25), típico de activos financieros.
"""
df["month"] = df["date"].dt.month
monthly = df.groupby("month")["gold_spot"].mean()
fig, ax = plt.subplots(figsize=(8, 4))
monthly.plot(kind="bar", ax=ax)
ax.set_title("Precio medio del oro por mes (2000-2025)")
save_fig(fig, "eda_seasonal.png")

vol21 = rets.rolling(21).std() * np.sqrt(252)
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(df["date"].iloc[1:], vol21, lw=0.8)
ax.set_title("Volatilidad anualizada (ventana 21d)")
save_fig(fig, "eda_vol.png")

# %%
"""Mapa de correlaciones entre features (heatmap).

Muchas variables financieras están altamente correlacionadas entre sí
(metales, índices, tipos). Esto informa la selección de features por
redundancia (fase 10): eliminaremos pares con |ρ| > 0.98.
"""
import seaborn as sns

# Solo features presentes en el dataset limpio (algunas se excluyeron por cobertura)
available = [c for c in ["gold_spot", "silver_spot", "dxy_index", "us10y_yield",
                           "vix_index", "wti_spot", "sp500_futures", "bitcoin_price",
                           "usdjpy_exchange", "gold_futures"] if c in df.columns]
key = available or ["gold_spot", "silver_spot", "dxy_index", "us10y_yield"]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df[key].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlaciones entre variables clave")
plt.tight_layout()
save_fig(fig, "eda_corr_heatmap.png")

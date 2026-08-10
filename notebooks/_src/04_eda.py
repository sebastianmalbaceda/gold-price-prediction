# %% [markdown]
# # Fase 4: EDA y entendimiento del dominio
#
# ## 4.1 EDA del target (gold_spot)
# - Serie larga con **tendencia alcista estructural** (1979 → 2025) y **cambios de régimen** (2008, 2011, 2020, 2024).
# - Precio en niveles: NO es estacionaria (ADF con p>0.05). Los retornos logarítmicos sí lo son.
# - Volatilidad agrupada (*volatility clustering*), típica de activos financieros.
#
# ## 4.2 EDA de features
# - Correlación esperada con el dólar (DXY) negativa y con tipos reales negativa.
# - Muchas features correlacionadas entre sí (metales, índices) → redundancia.
#
# ## 4.3 EDA específico de series temporales
# - Tendencia, estacionalidad semanal débil, autocorrelación alta en niveles y casi nula en retornos.
#
# ## 4.4 Hipótesis
# 1. Los **lags y retornos** del propio oro dominarán la predicción a 1 día.
# 2. DXY, tipos reales y VIX aportan señal exógena a horizontes medios.
# 3. La métrica más informativa para el usuario es **MAE/sMAPE** (USD por onza).
# 4. El split debe ser **temporal estricto**; los retornos permiten un modelo más estable.
# 5. Riesgo principal: *non-stationarity* → los modelos de árboles con lags gestionan niveles; los lineales requieren retornos.
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

"""Fase 4: EDA - estadísticos, tendencia, estacionalidad, autocorrelación."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.data.load_data import clean_daily_series, load_raw
from src.utils import save_fig

cfg = {"data": {"start_date": "2000-01-01", "end_date": "2025-09-12"}}
from src.config import get_config
cfg = get_config()

raw = load_raw(cfg)
df = clean_daily_series(raw, cfg)
print("Filas tras limpieza:", len(df), "| columnas:", df.shape[1])

# --- 4.1 Target ---
g = df["gold_spot"]
print("\n[target] describir:")
print(g.describe().round(2).to_string())
print("Asimetría:", round(stats.skew(g), 2), "| Curtosis:", round(stats.kurtosis(g), 2))

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

# --- 4.3 Autocorrelación ---
from pandas.plotting import autocorrelation_plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
autocorrelation_plot(g, ax=axes[0]); axes[0].set_title("ACF niveles")
autocorrelation_plot(rets, ax=axes[1]); axes[1].set_title("ACF retornos")
fig.tight_layout(); save_fig(fig, "eda_acf.png")

# --- Estacionariedad ---
from statsmodels.tsa.stattools import adfuller
for name, s in [("niveles", g.dropna()), ("retornos", rets.dropna())]:
    adf, p = adfuller(s)[:2]
    print(f"ADF {name}: stat={adf:.2f}, p={p:.4f} -> {'estacionaria' if p < 0.05 else 'NO estacionaria'}")

# --- 4.2 Correlaciones clave con el target (niveles y retornos) ---
exog = [c for c in df.columns if c not in ("date", "gold_spot")]
corr_levels = df[exog].corrwith(g).sort_values(ascending=False)
print("\nTop correlaciones con gold_spot (niveles):")
print(corr_levels.head(8).round(3).to_string())
print("\nPeores correlaciones (niveles):")
print(corr_levels.tail(5).round(3).to_string())

ret_df = df.copy()
ret_df["gold_ret"] = rets
corr_rets = ret_df[exog].corrwith(ret_df["gold_ret"])
print("\nTop correlaciones retorno oro vs retornos exógenas:")
print(corr_rets.sort_values(ascending=False).head(10).round(3).to_string())

# --- Estacionalidad mensual ---
df["month"] = df["date"].dt.month
monthly = df.groupby("month")["gold_spot"].mean()
fig, ax = plt.subplots(figsize=(8, 4))
monthly.plot(kind="bar", ax=ax)
ax.set_title("Precio medio del oro por mes (2000-2025)")
save_fig(fig, "eda_seasonal.png")

# --- Volatilidad agrupada ---
vol21 = rets.rolling(21).std() * np.sqrt(252)
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(df["date"].iloc[1:], vol21, lw=0.8)
ax.set_title("Volatilidad anualizada (ventana 21d)")
save_fig(fig, "eda_vol.png")

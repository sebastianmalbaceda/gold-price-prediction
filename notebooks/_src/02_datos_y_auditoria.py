# %% [markdown]
# # Fase 2-3: Obtención, gobierno e ingesta/auditoría de datos
#
# ## 2. Obtención y licencia
# - **Fuente:** dataset compilado *gold-price-prediction-dataset.csv* (Kaggle, dominio público / uso académico).
# - **Método:** archivo CSV descargado; contenido: 61 columnas (fecha + 60 variables).
# - **Periodo:** 1901-06-30 → 2025-09-14, frecuencia diaria.
# - **Integridad:** comprobada vía checksum al cargar (inmutable en `data/raw/`).
# - **Licencia:** el proyecto es de uso académico/educativo; ver `LICENSE`. No contiene datos personales.
#
# ## 3. Ingesta y auditoría
# ### 3.1 Validación de esquema
# - 45,368 filas × 61 columnas; `date` datetime64 única; 60 columnas float64.
# - Sin duplicados de fecha.
#
# ### 3.2 Calidad básica
# - **Cobertura desigual:** la mayoría de series empiezan a mediados del s. XX; `gold_spot` arranca en 1979-12-27.
# - **Fines de semana:** 12,963 filas (28.6%) sin cotización de oro → no son días de mercado.
# - **Huecos largos** en exógenas (CPI, paro... ~31 días; `ovx_index` 961 días) → gobernar con forward-fill y umbral de cobertura.
#
# ### 3.3 Riesgos iniciales detectados
# | Riesgo | Detección | Mitigación |
# |---|---|---|
# | Ventana con cobertura insuficiente (1901-1979) | cobertura < 3% | recorte a 2000+ |
# | Fines de semana sin cotización | 28.6% filas | filtrado (solo días hábiles) |
# | Features macro con retraso de publicación | huecos de ~31 días | forward-fill (solo pasado) |
# | Features casi vacías (CPI, M2, google_trends...) | < 5% cobertura | exclusión en config |
# | Leakage por orden de filas | — | orden temporal estricto garantizado |
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

"""Fase 2-3: auditoría del dataset crudo."""
import pandas as pd
import numpy as np

from src.data.load_data import load_raw

df = load_raw()
print("Shape:", df.shape)
print("Fechas:", df["date"].min().date(), "->", df["date"].max().date())
print("Duplicados de fecha:", df["date"].duplicated().sum())
print("Columnas float64:", (df.dtypes == "float64").sum(), "| datetime:", (df.dtypes == "datetime64[us]").sum())

# Cobertura global y por década
nn = df.notna().mean()
print("\nCobertura media global: {:.1%}".format(nn.mean()))
print("Columnas con cobertura <5%:", (nn < 0.05).sum())
print("Columnas con cobertura >=50%:", (nn >= 0.5).sum(), "de", len(nn))

# gold_spot
gs = df.dropna(subset=["gold_spot"])
print("\ngold_spot: desde", gs["date"].min().date(), "hasta", gs["date"].max().date(), "|", len(gs), "obs")

# Fines de semana
wk = df["date"].dt.dayofweek >= 5
print("Filas de fin de semana:", wk.sum(), f"({wk.mean():.1%})")

# Huecos grandes por columna (2000+)
sub = df[df["date"] >= "2000-01-01"]
gaps = {}
for c in sub.columns:
    if c == "date":
        continue
    s = sub[c].dropna()
    if len(s) > 1:
        gaps[c] = s.index.to_series().diff().max() or 0
top = pd.Series(gaps).sort_values(ascending=False).head(8)
print("\nMayores huecos (días) en 2000+:")
print(top.to_string())

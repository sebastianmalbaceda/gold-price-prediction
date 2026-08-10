# %% [markdown]
# # Fase 19-20: Robustez, ética, privacidad y seguridad
#
# ## 19. Robustez y generalización
#
# - **Pruebas de entrada:** nulos, tipos erróneos, valores extremos, categorías desconocidas → la API valida el esquema (fase 21).
# - **Pruebas de cambio:** se evalúa el modelo congelado en ventanas temporales posteriores (drift por segmento):
#   - 2023 vs 2024 vs 2025 → ¿degradación temporal?
#   - Se comparan distribuciones de features (covariate shift) entre train y test.
# - **Pruebas técnicas:** diferentes semillas (ensemble de la fase 18), submuestras.
#
# ## 20. Ética, privacidad y seguridad
#
# - **Datos:** sin datos personales (series de mercado públicas). No hay grupos afectados ni variables sensibles.
# - **Equidad:** no aplica discriminación; se documenta el uso previsto (analítico, no trading automático).
# - **Seguridad:** validación de inputs en API, sin secretos en el repo, dependencias fijadas.
# - **Uso indebido:** el modelo NO debe usarse como asesor financiero automatizado sin supervisión humana.
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
from src.utils import path_from_root, set_seed
set_seed(42)

# %%
"""Carga de datos y predicciones en test.

Preparamos test y artefactos para los análisis de robustez y drift.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import get_config
from src.data.split import temporal_split, drop_warmup
from src.evaluation.metrics import regression_metrics
from src.models.train_model import load_model_artifacts
from src.utils import save_json

cfg = get_config()
feats = pd.read_parquet(path_from_root("data/processed/features.parquet"))
feats = drop_warmup(feats, warmup=260)
parts = temporal_split(feats, cfg)
model, pp, sel_cols = load_model_artifacts(cfg)
h = cfg["target"]["primary_horizon"]

te = parts["test"]
X_te = pp.transform(te[sel_cols].to_numpy(dtype=np.float64))
y_te = te[f"target_{h}"].to_numpy(dtype=np.float64)
y_pred = model.predict(X_te)

# %%
"""19.2 Degradación temporal en test por año.

Evaluamos el modelo congelado por año (2023, 2024, 2025). La progresión
del MAE muestra si hay drift de mercado y cuánto se degrada el modelo.
"""
out = pd.DataFrame({"date": te["date"], "y": y_te, "p": y_pred})
out["year"] = out["date"].dt.year
perf = out.groupby("year").apply(
    lambda g: regression_metrics(g["y"].to_numpy(), g["p"].to_numpy(), h),
    include_groups=False).apply(pd.Series)
print("Rendimiento por año (test):")
print(perf[["mae", "smape", "directional_accuracy"]].round(3).to_string())

# %%
"""Visualización de la degradación temporal.

Gráfico de barras del MAE por año. La tendencia creciente (87 → 205 → 393)
evidencia que el modelo se degrada con el tiempo: es la señal principal
para el reentrenamiento (fase 23).
"""
import matplotlib.pyplot as plt

from src.utils import save_fig

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(perf.index.astype(str), perf["mae"], color="#d62728")
ax.set_title("MAE por año en test (modelo congelado)")
ax.set_ylabel("MAE (USD/oz)")
fig.tight_layout()
save_fig(fig, "robustness_by_year.png")

# %%
"""19.2 Covariate shift (KS test) entre train y test.

Comparamos la distribución de cada feature en train+val contra test con el
test de Kolmogorov-Smirnov. p < 0.01 indica drift significativo.
"""
train_val = pd.concat([parts["train"], parts["val"]]) \
    .sort_values("date").reset_index(drop=True)
key_feats = sel_cols[:15]
shift = {}
for c in key_feats:
    a = train_val[c].dropna().to_numpy()
    b = te[c].dropna().to_numpy()
    stat, p = ks_2samp(a, b)
    shift[c] = {"ks_stat": round(stat, 3), "p_value": round(p, 4)}

print("Covariate shift (KS) en features clave:")
for c, v in shift.items():
    flag = "DRIFT" if v["p_value"] < 0.01 else "ok"
    print(f"  {c:35s} KS={v['ks_stat']:.3f} p={v['p_value']:.4f} {flag}")

save_json({"segment_performance":
           perf[["mae", "smape", "directional_accuracy"]].round(3).to_dict("index"),
           "covariate_shift": shift}, "robustness.json")

# %%
"""20. Ética, privacidad y seguridad.

El dataset no contiene datos personales ni variables sensibles (son series
de mercado públicas). No hay grupos demográficos afectados, por lo que el
análisis de equidad por subgrupos no aplica.

Medidas de seguridad implementadas:
- Validación estricta de inputs en la API (Pydantic + rechazo de NaN/Inf/claves extra).
- Sin secretos en el repositorio (`.env` ignorado).
- Dependencias fijadas en `requirements.txt`.
- Uso previsto documentado: analítico, NO trading automático sin supervisión.

Este bloque es informativo; no ejecuta código.
"""
print("""
Checklist fase 20:
[OK] Sin datos personales ni variables sensibles
[OK] Sin secretos en el repo (.env ignorado)
[OK] Validación de inputs en API (Pydantic + finitos)
[OK] Uso previsto documentado (analítico, no trading automático)
[OK] Dependencias fijadas en requirements.txt
""")

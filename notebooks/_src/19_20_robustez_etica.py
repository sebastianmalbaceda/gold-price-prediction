# %% [markdown]
# # Fase 19-20: Robustez, ética, privacidad y seguridad
#
# ## 19. Robustez y generalización
# - **Pruebas de entrada:** nulos, tipos erróneos, valores extremos, categorías desconocidas → la API valida el esquema (fase 21).
# - **Pruebas de cambio:** se evalúa el modelo congelado en ventanas temporales posteriores (drift por segmento):
#   - 2023 vs 2024 vs 2025 → ¿degradación temporal?
#   - Se comparan distribuciones de features (covariate shift) entre train y test.
# - **Pruebas técnicas:** diferentes semillas (ensemble de la fase 18), submuestras.
#
# ## 20. Ética, privacidad y seguridad
# - **Datos:** sin datos personales (series de mercado públicas). No hay grupos afectados ni variables sensibles.
# - **Equidad:** no aplica discriminación; se documenta el uso previsto (analítico, no trading automático).
# - **Seguridad:** validación de inputs en API, sin secretos en el repo, dependencias fijadas.
# - **Uso indebido:** el modelo NO debe usarse como asesor financiero automatizado sin supervisión humana.
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

"""Fase 19-20: drift temporal y pruebas de robustez."""
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import get_config
from src.data.split import temporal_split, drop_warmup
from src.evaluation.metrics import regression_metrics
from src.models.train_model import load_model_artifacts
from src.utils import save_json, set_seed

set_seed(42)
cfg = get_config()
feats = pd.read_parquet("data/processed/features.parquet")
feats = drop_warmup(feats, warmup=260)
parts = temporal_split(feats, cfg)
model, pp, sel_cols = load_model_artifacts(cfg)
h = cfg["target"]["primary_horizon"]

# --- 19.2 Degradación temporal en test por semestre ---
te = parts["test"]
X_te = pp.transform(te[sel_cols].to_numpy(dtype=np.float64))
y_te = te[f"target_{h}"].to_numpy(dtype=np.float64)
y_pred = model.predict(X_te)

out = pd.DataFrame({"date": te["date"], "y": y_te, "p": y_pred})
out["period"] = pd.cut(out["date"], bins=6, labels=[f"S{i}" for i in range(1, 7)])
perf = out.groupby("period", observed=True).apply(
    lambda g: regression_metrics(g["y"].to_numpy(), g["p"].to_numpy(), h),
    include_groups=False).apply(pd.Series)
print("Rendimiento por segmento temporal (test):")
print(perf[["mae", "smape", "directional_accuracy"]].round(3).to_string())

# --- Covariate shift: KS test train vs test en features clave ---
train_val = pd.concat([parts["train"], parts["val"]]).sort_values("date").reset_index(drop=True)
key_feats = sel_cols[:12]  # muestra
shift = {}
for c in key_feats:
    a = train_val[c].dropna().to_numpy()
    b = te[c].dropna().to_numpy()
    stat, p = ks_2samp(a, b)
    shift[c] = {"ks_stat": round(stat, 3), "p_value": round(p, 4)}
print("\nCovariate shift (KS) en features clave:")
for c, v in shift.items():
    flag = "DRIFT" if v["p_value"] < 0.01 else "ok"
    print(f"  {c:35s} KS={v['ks_stat']:.3f} p={v['p_value']:.4f} {flag}")

save_json({"segment_performance": perf[["mae", "smape", "directional_accuracy"]].round(3).to_dict("index"),
           "covariate_shift": shift}, "robustness.json")

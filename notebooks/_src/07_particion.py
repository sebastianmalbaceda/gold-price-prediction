# %% [markdown]
# # Fase 7: Partición temporal y protocolo experimental
#
# Para **forecasting** se exige **orden temporal estricto**:
# - **Train:** 2000-01-01 → 2019-12-31 (aprende todo: scaler, modelo)
# - **Validation:** 2020-01-01 → 2022-12-31 (selección de modelos/hiperparámetros)
# - **Test:** 2023-01-01 → 2025-09-12 (**bloqueado**, solo se usa una vez al final)
#
# Además se elimina el *warm-up*: las primeras ~260 filas no tienen lags
# completos y se descartan ANTES de particionar (evita contaminación).
#
# ## Protocolo experimental
# 1. Los índices de partición se guardan (no se re-generan).
# 2. Nunca se toca test durante tuning.
# 3. CV: `TimeSeriesSplit(5 folds, gap=21)` sobre train+val.
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

"""Fase 7: partición temporal estricta + TimeSeriesSplit."""
import pandas as pd

from src.config import get_config
from src.data.split import drop_warmup, get_temporal_splitter, temporal_split

cfg = get_config()
feats = pd.read_parquet("data/processed/features.parquet")
feats = drop_warmup(feats, warmup=260)
print("Shape tras warm-up:", feats.shape)

parts = temporal_split(feats, cfg)
for k, v in parts.items():
    print(f"{k:6s}: {len(v):5d} filas | {v['date'].min().date()} -> {v['date'].max().date()}")

# Guardar particiones
for k, v in parts.items():
    v.to_parquet(f"data/processed/{k}.parquet", index=False)

# Verificar que no hay solape
tr, va, te = (parts[k]["date"] for k in ("train", "val", "test"))
assert tr.max() < va.min() and va.max() < te.min(), "Solape entre particiones"
print("\nSin solape temporal: OK")

# TimeSeriesSplit de ejemplo
tscv = get_temporal_splitter(cfg)
all_dates = pd.concat([tr, va]).sort_values()
n = len(all_dates)
print(f"\nTimeSeriesSplit({tscv.n_splits} folds, gap={tscv.gap}) sobre {n} filas:")
for i, (tr_idx, va_idx) in enumerate(tscv.split(range(n))):
    print(f"  fold {i}: train {len(tr_idx):5d} | val {len(va_idx):5d}")

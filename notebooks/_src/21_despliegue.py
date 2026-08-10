# %% [markdown]
# # Fase 21: Empaquetado, inferencia y despliegue
#
# ## 21.1 Pipeline de inferencia
#
# ```
# Input crudo (date + exógenas)
# → validación de esquema (Pydantic)
# → limpieza (ffill, solo días hábiles)
# → feature engineering (mismo código de train)
# → preprocesador entrenado (transform)
# → modelo final
# → respuesta (predicción + horizonte + timestamp)
# ```
#
# **Regla de oro:** la inferencia usa EXACTAMENTE el preprocesado y features
# entrenados (mismo código, mismos artefactos). Nada de reimplementaciones.
#
# ## 21.2 Formas de entrega
#
# - **API REST** en `src/api/main.py` (FastAPI) → sirve predicciones a 1 día.
# - **Script CLI** `scripts/predict.py` para batch.
# - Notebook reproducible (este).
#
# ## 21.3 Validación de entradas
#
# - Esquema Pydantic: fecha, lista de valores de las features usadas.
# - Errores claros, tipos validados, valores fuera de rango rechazados con mensaje.
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
"""Preparación: comprobamos que los artefactos existen.

Antes de arrancar la API verificamos que el modelo, el preprocesador y la
lista de features están guardados (fases 15-16). Si no, la API devolverá
503.
"""
import json

from src.models.train_model import load_model_artifacts

model, pp, sel_cols = load_model_artifacts()
print(f"Modelo: {type(model).__name__} | features: {len(sel_cols)}")
print("Artefactos OK. Listos para arrancar la API.")

# %%
"""Arranque de la API en segundo plano (celda autocontenida).

Lanzamos uvicorn en un puerto libre. Esperamos a que responda `/health`
y detenemos el proceso al final. Todo en una sola celda para no dejar
bloques try abiertos entre celdas.
"""
import subprocess
import sys
import time

import requests

PORT = 8000
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.main:app",
     "--host", "127.0.0.1", "--port", str(PORT)],
    cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

try:
    ok = False
    for _ in range(40):
        time.sleep(0.5)
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
            if r.status_code == 200:
                ok = True
                break
        except requests.ConnectionError:
            continue
    if not ok:
        raise RuntimeError("La API no arrancó en 20s")
    print("Health:", r.json())

    # --- Prueba de predicción válida ---
    import pandas as pd

    from src.config import get_config

    cfg = get_config()
    feats = pd.read_parquet(path_from_root("data/processed/features.parquet"))
    with open(path_from_root("models", "feature_list.json")) as f:
        feats_list = json.load(f)

    last = feats.iloc[-1]
    payload = {"date": str(last["date"].date()),
               "features": {c: (None if pd.isna(last[c]) else float(last[c]))
                            for c in feats_list}}
    resp = requests.post(f"http://127.0.0.1:{PORT}/predict", json=payload, timeout=10)
    print("\nPOST /predict ->", resp.status_code)
    print(json.dumps(resp.json(), indent=2, default=str))

    # --- Prueba de predicción de dirección (fase 16b) ---
    resp_dir = requests.post(f"http://127.0.0.1:{PORT}/predict_direction",
                             json=payload, timeout=10)
    print("\nPOST /predict_direction ->", resp_dir.status_code)
    print(json.dumps(resp_dir.json(), indent=2, default=str))

    # --- Pruebas de validación de inputs ---
    bad = {"date": "2025-01-01", "features": {"inexistente": 1.0}}
    resp = requests.post(f"http://127.0.0.1:{PORT}/predict", json=bad, timeout=10)
    print("\nInput inválido (feature inexistente) ->",
          resp.status_code, resp.json().get("detail", "")[:120])

    bad_date = {**payload, "date": "2025-13-45"}
    resp = requests.post(f"http://127.0.0.1:{PORT}/predict", json=bad_date, timeout=10)
    print("Fecha inválida ->", resp.status_code,
          resp.json().get("detail", "")[:120])

    bad_extra = {**payload, "features": {**payload["features"], "extra": 1.0}}
    resp = requests.post(f"http://127.0.0.1:{PORT}/predict", json=bad_extra, timeout=10)
    print("Claves extra ->", resp.status_code,
          resp.json().get("detail", "")[:120])
finally:
    proc.terminate()
    print("\nAPI detenida.")

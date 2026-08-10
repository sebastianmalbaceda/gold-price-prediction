# %% [markdown]
# # Fase 21: Empaquetado, inferencia y despliegue
#
# ## 21.1 Pipeline de inferencia
# ```
# Input crudo (date + exógenas)
# → validación de esquema (Pydantic)
# → limpieza (ffill, solo días hábiles)
# → feature engineering (mismo código de train)
# → preprocesador entrenado (transform)
# → modelo final
# → respuesta (predicción + horizonte + timestamp)
# ```
# **Regla de oro:** la inferencia usa EXACTAMENTE el preprocesado y features
# entrenados (mismo código, mismos artefactos). Nada de reimplementaciones.
#
# ## 21.2 Formas de entrega
# - **API REST** en `src/api/main.py` (FastAPI) → sirve predicciones a 1 día.
# - **Script CLI** `scripts/predict.py` para batch.
# - Notebook reproducible (este).
#
# ## 21.3 Validación de entradas
# - Esquema Pydantic: fecha, lista de valores de las 37 features usadas.
# - Errores claros, tipos validados, valores fuera de rango rechazados con mensaje.
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

"""Fase 21: arranque del servidor de inferencia y prueba local."""
import subprocess, sys, time
import requests

# Arrancar la API en segundo plano
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.main:app",
     "--host", "127.0.0.1", "--port", "8000"],
    cwd=".", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

try:
    # Esperar a que arranque
    for _ in range(30):
        time.sleep(1)
        try:
            r = requests.get("http://127.0.0.1:8000/health", timeout=2)
            if r.status_code == 200:
                break
        except requests.ConnectionError:
            continue
    print("Health:", r.json())

    # Construir payload con los últimos valores conocidos de las features
    import json
    import pandas as pd
    from src.config import get_config
    from src.data.split import drop_warmup, temporal_split
    from src.features.build_features import get_feature_columns

    cfg = get_config()
    feats = pd.read_parquet("data/processed/features.parquet")
    feats = drop_warmup(feats, warmup=260)
    with open("models/feature_list.json") as f:
        sel_cols = json.load(f)

    last = feats.iloc[-1]
    payload = {"date": str(last["date"].date()),
               "features": {c: (None if pd.isna(last[c]) else float(last[c]))
                            for c in sel_cols}}
    resp = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
    print("\nPOST /predict ->", resp.status_code)
    print(json.dumps(resp.json(), indent=2, default=str))

    # Prueba de input inválido
    bad = {"date": "2025-01-01", "features": {"inexistente": 1.0}}
    resp = requests.post("http://127.0.0.1:8000/predict", json=bad, timeout=10)
    print("\nInput inválido ->", resp.status_code, resp.json().get("detail", "")[:200])
finally:
    proc.terminate()
    print("\nAPI detenida.")

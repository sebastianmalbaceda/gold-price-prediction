"""API REST de inferencia (fase 21): FastAPI con validación de inputs.

Sirve predicciones del precio spot del oro a h=1 día hábil.

Ejecución:
    uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_config, path_from_root
from src.models.train_model import load_model_artifacts

app = FastAPI(
    title="Gold Price Prediction API",
    description="Predicción del precio spot del oro (USD/oz) a 1 día hábil.",
    version="1.0.0",
)

# Artefactos cargados una sola vez (lazy al arrancar)
_model = None
_preprocessor = None
_features: list[str] = []


def _ensure_loaded():
    global _model, _preprocessor, _features
    if _model is None:
        try:
            cfg = get_config()
            _model, _preprocessor, _features = load_model_artifacts(cfg)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Modelo no disponible: {e}. Ejecute primero el entrenamiento (fases 15-16).",
            )


class PredictRequest(BaseModel):
    """Entrada: fecha y valores de las features en el momento t.

    Las features son exactamente las usadas en entrenamiento
    (ver models/feature_list.json).
    """
    date: str = Field(..., description="Fecha (YYYY-MM-DD) del momento de predicción")
    features: dict[str, float] = Field(..., description="Valores de las features en t")


class PredictResponse(BaseModel):
    date: str
    horizon_days: int
    prediction_usd_per_oz: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "gold_spot"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _ensure_loaded()
    try:
        date = pd.Timestamp(req.date)
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="Fecha inválida (use YYYY-MM-DD)")

    # Validación del esquema: deben venir EXACTAMENTE las features del modelo
    missing = [c for c in _features if c not in req.features]
    if missing:
        raise HTTPException(status_code=422,
                            detail=f"Faltan features: {missing[:10]}...")
    extra = [c for c in req.features if c not in _features]
    if extra:
        raise HTTPException(status_code=422,
                            detail=f"Features no esperadas: {extra[:10]}...")

    # Construir vector en el orden exacto de entrenamiento
    try:
        row = [float(req.features[c]) for c in _features]
    except (TypeError, ValueError):
        raise HTTPException(status_code=422,
                            detail="Todas las features deben ser numéricas")

    # Rechazar NaN/Inf (el modelo no los acepta y no son datos válidos)
    if not all(math.isfinite(v) for v in row):
        raise HTTPException(status_code=422,
                            detail="Las features deben ser finitas (sin NaN/Inf)")

    X = np.asarray([row], dtype=np.float64)
    X_scaled = _preprocessor.transform(X)
    pred = float(_model.predict(X_scaled)[0])
    return PredictResponse(
        date=str(date.date()),
        horizon_days=1,
        prediction_usd_per_oz=round(pred, 2),
        model_version=get_config()["model"]["version"],
    )

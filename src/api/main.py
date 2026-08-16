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

from src.config import get_config
from src.models.classifier import load_classifier_artifacts
from src.models.train_model import load_model_artifacts

app = FastAPI(
    title="Gold Price Prediction API",
    description="Predicción del precio spot del oro (USD/oz) y su dirección (sube/baja).",
    version="1.1.0",
)

# Artefactos cargados una sola vez (lazy al arrancar)
_model = None
_preprocessor = None
_features: list[str] = []
_clf = None
_clf_pp = None
_clf_features: list[str] = []
_feature_ranges: dict | None = None

# Rangos absolutos plausibles por tipo de variable (rechazan valores
# físicamente imposibles sin bloquear el drift real de mercado).
# Se aplican SOLO a variables de nivel (no a retornos, que son acotados).
ABSOLUTE_RANGES = {
    "us10y_yield": (-5.0, 20.0),
    "us2y_yield": (-5.0, 20.0),
    "dxy_index": (50.0, 200.0),
    "dxy_future": (50.0, 200.0),
    "vix_index": (5.0, 150.0),
    "gold_spot": (100.0, 10000.0),
}


def _validate_absolute_range(features: dict[str, float], feature_list: list[str]) -> None:
    """Rechaza valores físicamente imposibles para variables de nivel.

    A diferencia de la validación por rango de train (frágil ante drift),
    usa rangos absolutos amplios: solo bloquea errores groseros o ataques,
    sin rechazar el drift legítimo del mercado.
    """
    for c in feature_list:
        # Solo validar variables de NIVEL (sin sufijos de retorno/lag/missing)
        if any(c.endswith(s) for s in ("_ret_lag1", "_lag1", "_missing")):
            continue
        r = ABSOLUTE_RANGES.get(c)
        if r is None:
            continue
        v = features.get(c)
        if v is None:
            continue
        lo, hi = r
        if v < lo or v > hi:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Valor fuera de rango plausible para '{c}': {v:.2f} "
                    f"(rango esperado [{lo}, {hi}])"
                ),
            )


def _ensure_loaded():
    global _model, _preprocessor, _features
    if _model is None:
        try:
            cfg = get_config()
            _model, _preprocessor, _features = load_model_artifacts(cfg)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Modelo no disponible: " f"{e}. Ejecute primero el entrenamiento (fases 15-16)."
                ),
            )


def _ensure_clf_loaded():
    global _clf, _clf_pp, _clf_features
    if _clf is None:
        try:
            _clf, _clf_pp, _clf_features = load_classifier_artifacts()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Clasificador no disponible: {e}. Ejecute el notebook 16.",
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


class DirectionResponse(BaseModel):
    date: str
    horizon_days: int
    probability_up: float
    direction: str  # "up" | "down"
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
        raise HTTPException(status_code=422, detail=f"Faltan features: {missing[:10]}...")
    extra = [c for c in req.features if c not in _features]
    if extra:
        raise HTTPException(status_code=422, detail=f"Features no esperadas: {extra[:10]}...")

    # Construir vector en el orden exacto de entrenamiento
    try:
        row = [float(req.features[c]) for c in _features]
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Todas las features deben ser numéricas")

    # Rechazar NaN/Inf (el modelo no los acepta y no son datos válidos)
    if not all(math.isfinite(v) for v in row):
        raise HTTPException(status_code=422, detail="Las features deben ser finitas (sin NaN/Inf)")

    # Validar rango plausible (evita extrapolaciones absurdas)
    _validate_absolute_range(req.features, _features)

    X = np.asarray([row], dtype=np.float64)
    X_scaled = _preprocessor.transform(X)
    pred = float(_model.predict(X_scaled)[0])
    return PredictResponse(
        date=str(date.date()),
        horizon_days=1,
        prediction_usd_per_oz=round(pred, 2),
        model_version=get_config()["model"]["version"],
    )


@app.post("/predict_direction", response_model=DirectionResponse)
def predict_direction(req: PredictRequest):
    """Predice la probabilidad de que el oro SUBE en t+1.

    Usa el clasificador de dirección (RandomForest calibrado) con las
    mismas features que la regresión. La señal es débil (AUC ~0.55):
    el resultado debe interpretarse como una leve inclinación, no como
    una certeza.
    """
    _ensure_clf_loaded()
    try:
        date = pd.Timestamp(req.date)
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="Fecha inválida (use YYYY-MM-DD)")

    missing = [c for c in _clf_features if c not in req.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Faltan features: {missing[:10]}...")
    extra = [c for c in req.features if c not in _clf_features]
    if extra:
        raise HTTPException(status_code=422, detail=f"Features no esperadas: {extra[:10]}...")

    try:
        row = [float(req.features[c]) for c in _clf_features]
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Todas las features deben ser numéricas")
    if not all(math.isfinite(v) for v in row):
        raise HTTPException(status_code=422, detail="Las features deben ser finitas (sin NaN/Inf)")

    # Validar rango plausible (evita extrapolaciones absurdas)
    _validate_absolute_range(req.features, _clf_features)

    X = np.asarray([row], dtype=np.float64)
    X_scaled = _clf_pp.transform(X)
    prob_up = float(_clf.predict_proba(X_scaled)[0][1])
    return DirectionResponse(
        date=str(date.date()),
        horizon_days=1,
        probability_up=round(prob_up, 4),
        direction="up" if prob_up >= 0.5 else "down",
        model_version=get_config()["model"]["version"],
    )

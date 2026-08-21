"""API REST de inferencia (fase 21) con validacion de entradas."""

from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_config
from src.models.classifier import load_classifier_artifacts
from src.models.train_model import load_model_artifacts

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gold Price Prediction API",
    description="Prediccion del precio spot del oro (USD/oz) y su direccion (sube/baja).",
    version="1.1.0",
)

# Artefactos cargados una sola vez (lazy al arrancar).
_model = None
_preprocessor = None
_features: list[str] = []
_clf = None
_clf_pp = None
_clf_features: list[str] = []
MAX_FEATURES = 256
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Rangos absolutos amplios para bloquear errores de entrada evidentes sin
# confundir drift de mercado con datos imposibles.
ABSOLUTE_RANGES = {
    "us10y_yield": (-5.0, 20.0),
    "us2y_yield": (-5.0, 20.0),
    "dxy_index": (50.0, 200.0),
    "dxy_future": (50.0, 200.0),
    "vix_index": (5.0, 150.0),
    "gold_spot": (100.0, 10000.0),
}


def _parse_date(value: str) -> date:
    """Acepta exclusivamente fechas ISO ``YYYY-MM-DD``."""
    if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
        raise HTTPException(status_code=422, detail="Fecha invalida (use YYYY-MM-DD)")
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Fecha invalida (use YYYY-MM-DD)") from exc


def _validate_feature_count(features: dict[str, float]) -> None:
    if len(features) > MAX_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"Se recibieron demasiadas features (maximo {MAX_FEATURES})",
        )


def _validate_absolute_range(features: dict[str, float], feature_list: list[str]) -> None:
    """Valida niveles base y sus lags, no retornos ni indicadores de ausencia."""
    for column in feature_list:
        if column.endswith("_missing") or re.search(r"_ret_lag\d+$", column):
            continue
        base = re.sub(r"_lag\d+$", "", column)
        limits = ABSOLUTE_RANGES.get(base)
        if limits is None:
            continue
        value = features.get(column)
        if value is None:
            continue
        low, high = limits
        if value < low or value > high:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Valor fuera de rango plausible para '{column}': {value:.2f} "
                    f"(rango esperado [{low}, {high}])"
                ),
            )


def _validate_features(features: dict[str, float], feature_list: list[str]) -> np.ndarray:
    """Comprueba esquema, finitud y rangos y devuelve el vector ordenado."""
    _validate_feature_count(features)
    missing = [column for column in feature_list if column not in features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Faltan features: {missing[:10]}...")
    extra = [column for column in features if column not in feature_list]
    if extra:
        raise HTTPException(status_code=422, detail=f"Features no esperadas: {extra[:10]}...")
    try:
        row = [float(features[column]) for column in feature_list]
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="Todas las features deben ser numericas"
        ) from exc
    if not all(math.isfinite(value) for value in row):
        raise HTTPException(status_code=422, detail="Las features deben ser finitas (sin NaN/Inf)")
    _validate_absolute_range(features, feature_list)
    return np.asarray([row], dtype=np.float64)


def _check_artifact_compatibility(model, preprocessor, features: list[str]) -> None:
    if not features:
        raise ValueError("La lista de features esta vacia")
    pp_count = getattr(preprocessor, "n_features_in_", None)
    if pp_count is not None and int(pp_count) != len(features):
        raise ValueError("El preprocesador y la lista de features no coinciden")
    model_count = getattr(model, "n_features_in_", None)
    if model_count is not None and int(model_count) != len(features):
        raise ValueError("El modelo y la lista de features no coinciden")


def _ensure_loaded() -> None:
    global _model, _preprocessor, _features
    if _model is not None:
        return
    try:
        cfg = get_config()
        model, preprocessor, features = load_model_artifacts(cfg)
        _check_artifact_compatibility(model, preprocessor, features)
        _model, _preprocessor, _features = model, preprocessor, features
    except FileNotFoundError as exc:
        logger.warning("Artefactos de regresion no disponibles: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Modelo de regresion no disponible; ejecute el entrenamiento.",
        ) from exc
    except Exception as exc:  # joblib puede lanzar varios errores.
        logger.exception("No se pudieron cargar los artefactos de regresion")
        raise HTTPException(status_code=503, detail="Artefactos de regresion invalidos.") from exc


def _ensure_clf_loaded() -> None:
    global _clf, _clf_pp, _clf_features
    if _clf is not None:
        return
    try:
        clf, clf_pp, clf_features = load_classifier_artifacts()
        _check_artifact_compatibility(clf, clf_pp, clf_features)
        _clf, _clf_pp, _clf_features = clf, clf_pp, clf_features
    except FileNotFoundError as exc:
        logger.warning("Artefactos de clasificacion no disponibles: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Clasificador no disponible; ejecute el notebook de direccion.",
        ) from exc
    except Exception as exc:
        logger.exception("No se pudieron cargar los artefactos de clasificacion")
        raise HTTPException(
            status_code=503, detail="Artefactos de clasificacion invalidos."
        ) from exc


def _model_version() -> str:
    return str(get_config()["model"].get("version", "unknown"))


class PredictRequest(BaseModel):
    """Entrada: fecha y valores de las features en el momento t."""

    model_config = ConfigDict(extra="forbid")
    date: str = Field(..., min_length=10, max_length=10)
    features: dict[str, float] = Field(...)


class PredictResponse(BaseModel):
    date: str
    horizon_days: int
    prediction_usd_per_oz: float
    model_version: str


class DirectionResponse(BaseModel):
    date: str
    horizon_days: int
    probability_up: float
    direction: str
    model_version: str


@app.get("/health")
def health():
    """Liveness: el proceso esta activo, aunque aun no tenga modelos cargados."""
    return {
        "status": "ok",
        "model": "gold_spot",
        "regression_loaded": _model is not None,
        "direction_loaded": _clf is not None,
    }


@app.get("/ready")
def ready():
    """Readiness: requiere que ambos artefactos de inferencia sean utilizables."""
    _ensure_loaded()
    _ensure_clf_loaded()
    return {"status": "ready", "model": "gold_spot"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predice el nivel de oro a un día hábil."""
    parsed_date = _parse_date(req.date)
    _ensure_loaded()
    X = _validate_features(req.features, _features)
    try:
        prediction = float(_model.predict(_preprocessor.transform(X))[0])
    except Exception as exc:
        logger.exception("Fallo durante inferencia de regresion")
        raise HTTPException(status_code=503, detail="No se pudo ejecutar la inferencia.") from exc
    if not math.isfinite(prediction):
        raise HTTPException(status_code=503, detail="El modelo devolvio una prediccion no finita")
    return PredictResponse(
        date=parsed_date.isoformat(),
        horizon_days=1,
        prediction_usd_per_oz=round(prediction, 2),
        model_version=_model_version(),
    )


@app.post("/predict_direction", response_model=DirectionResponse)
def predict_direction(req: PredictRequest):
    """Predice la probabilidad de que el oro suba en t+1."""
    parsed_date = _parse_date(req.date)
    _ensure_clf_loaded()
    X = _validate_features(req.features, _clf_features)
    try:
        probability = float(_clf.predict_proba(_clf_pp.transform(X))[0][1])
    except Exception as exc:
        logger.exception("Fallo durante inferencia de clasificacion")
        raise HTTPException(status_code=503, detail="No se pudo ejecutar la inferencia.") from exc
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise HTTPException(
            status_code=503, detail="El clasificador devolvio una probabilidad invalida"
        )
    return DirectionResponse(
        date=parsed_date.isoformat(),
        horizon_days=1,
        probability_up=round(probability, 4),
        direction="up" if probability >= 0.5 else "down",
        model_version=_model_version(),
    )

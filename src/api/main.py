"""API REST de inferencia (fase 21) con validacion de entradas."""

from __future__ import annotations

import functools
import logging
import math
import os
import re
import secrets
import threading
from datetime import date

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
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
# Limite del cuerpo HTTP. Se aplica ANTES de deserializar: sin el, el limite de
# MAX_FEATURES solo actua cuando Pydantic ya ha materializado el diccionario
# completo en memoria, lo que deja abierta una denegacion de servicio por
# volumen. 256 features numericas caben holgadamente en 128 KiB.
MAX_BODY_BYTES = int(os.environ.get("API_MAX_BODY_BYTES", 128 * 1024))
# Clave de API opcional. Si no se define GOLD_API_KEY la API queda abierta
# (comportamiento historico, valido solo en local); definirla activa la
# autenticacion por cabecera ``X-API-Key`` en los endpoints de inferencia.
_API_KEY = os.environ.get("GOLD_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Los artefactos se cargan de forma perezosa desde varios hilos del threadpool
# de Starlette; sin cerrojo, un hilo puede observar ``_model`` ya asignado
# mientras ``_preprocessor`` sigue a None.
_load_lock = threading.Lock()
_clf_load_lock = threading.Lock()

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


def require_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    """Exige ``X-API-Key`` unicamente si ``GOLD_API_KEY`` esta configurada."""
    if not _API_KEY:
        return
    if not api_key or not secrets.compare_digest(api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Clave de API invalida o ausente")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Rechaza cuerpos demasiado grandes antes de deserializar el JSON."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Cuerpo demasiado grande (maximo {MAX_BODY_BYTES} bytes)"},
                )
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Content-Length invalido"})
    return await call_next(request)


def _parse_date(value: str) -> date:
    """Acepta exclusivamente fechas ISO ``YYYY-MM-DD``."""
    if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
        raise HTTPException(status_code=422, detail="Fecha invalida (use YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
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
    """Carga perezosa y thread-safe de los artefactos de regresion."""
    global _model, _preprocessor, _features
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:  # otro hilo termino la carga mientras esperabamos
            return
        try:
            cfg = get_config()
            model, preprocessor, features = load_model_artifacts(cfg)
            _check_artifact_compatibility(model, preprocessor, features)
        except FileNotFoundError as exc:
            logger.warning("Artefactos de regresion no disponibles: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Modelo de regresion no disponible; ejecute el entrenamiento.",
            ) from exc
        except Exception as exc:  # joblib puede lanzar varios errores.
            logger.exception("No se pudieron cargar los artefactos de regresion")
            raise HTTPException(
                status_code=503, detail="Artefactos de regresion invalidos."
            ) from exc
        # ``_model`` se asigna EL ULTIMO: es el centinela que leen los demas
        # hilos, asi que no debe hacerse visible antes que sus dependencias.
        _preprocessor, _features = preprocessor, features
        _model = model


def _ensure_clf_loaded() -> None:
    """Carga perezosa y thread-safe de los artefactos de clasificacion."""
    global _clf, _clf_pp, _clf_features
    if _clf is not None:
        return
    with _clf_load_lock:
        if _clf is not None:
            return
        try:
            clf, clf_pp, clf_features = load_classifier_artifacts()
            _check_artifact_compatibility(clf, clf_pp, clf_features)
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
        _clf_pp, _clf_features = clf_pp, clf_features
        _clf = clf


@functools.lru_cache(maxsize=1)
def _model_version() -> str:
    """Version del modelo, cacheada: antes se leia el YAML en cada peticion.

    Una configuracion sin seccion ``model`` provocaba un ``KeyError`` y, por
    tanto, un HTTP 500 en cada prediccion. Los metadatos no son esenciales para
    responder, asi que se degradan a ``unknown`` con un aviso en el log.
    """
    try:
        return str(get_config()["model"].get("version", "unknown"))
    except (KeyError, TypeError, AttributeError):
        logger.warning("model.version no configurado; se reporta 'unknown'")
        return "unknown"


@functools.lru_cache(maxsize=1)
def _primary_horizon() -> int:
    """Horizonte principal declarado en configuracion.

    Antes estaba codificado a 1 en las respuestas, de modo que la API habria
    mentido sobre su propio contrato si se cambiaba ``target.primary_horizon``.
    """
    try:
        return int(get_config()["target"]["primary_horizon"])
    except (KeyError, TypeError, ValueError):
        logger.warning("target.primary_horizon no configurado; se asume 1")
        return 1


class PredictRequest(BaseModel):
    """Entrada: fecha y valores de las features en el momento t."""

    model_config = ConfigDict(extra="forbid")
    date: str = Field(..., min_length=10, max_length=10)
    # ``max_length`` lo aplica Pydantic durante la validacion del modelo, de
    # modo que un diccionario desmesurado se rechaza sin recorrerlo entero.
    features: dict[str, float] = Field(..., max_length=MAX_FEATURES)


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


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest):
    """Predice el nivel de oro al horizonte principal configurado."""
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
        horizon_days=_primary_horizon(),
        prediction_usd_per_oz=round(prediction, 2),
        model_version=_model_version(),
    )


@app.post(
    "/predict_direction",
    response_model=DirectionResponse,
    dependencies=[Depends(require_api_key)],
)
def predict_direction(req: PredictRequest):
    """Probabilidad de que el oro suba al horizonte principal configurado."""
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
        horizon_days=_primary_horizon(),
        probability_up=round(probability, 4),
        direction="up" if probability >= 0.5 else "down",
        model_version=_model_version(),
    )

# Manual de la API - Gold Price Prediction

API REST para predecir el **precio spot del oro (USD/oz) a 1 día hábil** y su
**dirección (sube/baja)**.

## Arranque

```bash
# Desarrollo/local solamente
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Documentación interactiva (OpenAPI): http://127.0.0.1:8000/docs

**Producción**: ejecutar detrás de un reverse proxy con TLS, autenticación,
rate limiting, límite de cuerpo y logs; no exponer Uvicorn directamente.

## Endpoints

### `GET /health`
```json
{"status": "ok", "model": "gold_spot"}
```

### `GET /ready`
```json
{"status": "ready", "model": "gold_spot"}
```
Devuelve `503` si faltan o son invalidos los artefactos de regresion o
clasificacion. `/health` solo comprueba liveness.

### `POST /predict` - regresión de nivel

**Request**
```json
{
  "date": "2025-08-14",
  "features": {
    "us10y_yield": 4.01,
    "gold_spot_lag1": 2940.5,
    "silver_spot": 33.2,
    "...": 0.0
  }
}
```
- `date`: fecha (YYYY-MM-DD) del momento de predicción.
- `features`: **todas** las features del modelo (lista en `models/feature_list.json`, 83 columnas).

> La API no ingiere datos crudos ni realiza limpieza, `ffill`, cálculo de lags,
> retornos o ventanas móviles. El cliente debe enviar el vector completo de
> features ya calculadas, con el mismo esquema y versión que el entrenamiento.

**Response 200**
```json
{
  "date": "2025-08-14",
  "horizon_days": 1,
  "prediction_usd_per_oz": 2968.35,
  "model_version": "1.1.0"
}
```

### `POST /predict_direction` - clasificación sube/baja

Mismo request que `/predict`. Devuelve la **probabilidad calibrada** de que el
oro suba en t+1 y la dirección con umbral 0.5.

**Response 200**
```json
{
  "date": "2025-08-14",
  "horizon_days": 1,
  "probability_up": 0.5371,
  "direction": "up",
  "model_version": "1.1.0"
}
```

> **Interpretación honesta**: el clasificador tiene AUC ~ 0.55 en test. La
> señal es débil (mercado eficiente); `probability_up` debe leerse como una
> **leve inclinación**, no como una certeza. No usar como señal de trading
> automático sin supervisión.

**Errores**
| Código | Caso |
|---|---|
| 400 | cabecera `Content-Length` inválida |
| 401 | cabecera `X-API-Key` ausente o inválida cuando `GOLD_API_KEY` está definida |
| 422 | fecha inválida, features faltantes/extra, valores NaN/Inf/no numéricos |
| 413 | `Content-Length` declarado superior al límite configurado |
| 503 | modelo/clasificador no entrenado (ejecutar fases 15-16 y notebook 16) |

## Validación de entradas
- Esquema Pydantic estricto (`PredictRequest`).
- El servidor verifica que el conjunto de features coincida exactamente con
  `models/feature_list.json` (mismo orden que entrenamiento).
- El campo `features` está limitado por Pydantic a 256 entradas.
- Rechaza NaNs, Inf, claves extra, rangos fisicamente imposibles y fechas que no cumplen `YYYY-MM-DD`.
- Los artefactos joblib deben proceder de una fuente confiable; su carga es una deserializacion de Python.

## Controles de seguridad de la API

- Si se define `GOLD_API_KEY`, `/predict` y `/predict_direction` exigen la
  cabecera `X-API-Key`. Si no se define, ambos endpoints quedan abiertos por
  compatibilidad con el comportamiento histórico; úselo solo en local.
- `API_MAX_BODY_BYTES` establece el límite del cuerpo HTTP declarado mediante
  `Content-Length`; su valor predeterminado es 131072 bytes (128 KiB). El
  middleware devuelve HTTP 413 cuando la cabecera declara un tamaño superior.
- En despliegues públicos se requiere un reverse proxy con TLS, autenticación,
  rate limiting, límite de cuerpo y logs; no exponer Uvicorn directamente.

## Compatibilidad entrenamiento/servicio
La API carga los artefactos serializados (`final_model.joblib`, `preprocessor.joblib`,
`direction_classifier.joblib`) y aplica `transform` del preprocesador entrenado:
**cero divergencia** entre entrenamiento e inferencia (regla de oro de la fase 21).

## Tests
```bash
pytest tests/ -q        # incluye validación de la API (TestClient)
```

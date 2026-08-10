# Manual de la API — Gold Price Prediction

API REST para predecir el **precio spot del oro (USD/oz) a 1 día hábil**.

## Arranque

```bash
uvicorn src.api.main:app --reload          # desarrollo
uvicorn src.api.main:app --host 0.0.0.0 --port 8000   # producción
```

Documentación interactiva (OpenAPI): http://127.0.0.1:8000/docs

## Endpoints

### `GET /health`
```json
{"status": "ok", "model": "gold_spot"}
```

### `POST /predict`

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
- `features`: **todas** las features del modelo (lista en `models/feature_list.json`, 85 columnas).

**Response 200**
```json
{
  "date": "2025-08-14",
  "horizon_days": 1,
  "prediction_usd_per_oz": 2968.35,
  "model_version": "1.0.0"
}
```

**Errores**
| Código | Caso |
|---|---|
| 422 | fecha inválida, features faltantes, valores NaN |
| 503 | modelo no entrenado (ejecutar fases 15-16 primero) |

## Validación de entradas
- Esquema Pydantic estricto (`PredictRequest`).
- El servidor verifica que el conjunto de features coincida exactamente con
  `models/feature_list.json` (mismo orden que entrenamiento).
- Rechaza NaNs y fechas malformadas con mensaje descriptivo.

## Compatibilidad entrenamiento/servicio
La API carga los artefactos serializados (`final_model.joblib`, `preprocessor.joblib`)
y aplica `transform` del preprocesador entrenado: **cero divergencia** entre
entrenamiento e inferencia (regla de oro de la fase 21).

## Tests
```bash
pytest tests/ -q        # incluye validación de la API (TestClient)
```

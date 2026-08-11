# Manual de la API - Gold Price Prediction

API REST para predecir el **precio spot del oro (USD/oz) a 1 día hábil** y su
**dirección (sube/baja)**.

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
- `features`: **todas** las features del modelo (lista en `models/feature_list.json`, 110 columnas).

**Response 200**
```json
{
  "date": "2025-08-14",
  "horizon_days": 1,
  "prediction_usd_per_oz": 2968.35,
  "model_version": "1.0.0"
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
  "model_version": "1.0.0"
}
```

> [!] **Interpretación honesta**: el clasificador tiene AUC ~ 0.55 en test. La
> señal es débil (mercado eficiente); `probability_up` debe leerse como una
> **leve inclinación**, no como una certeza. No usar como señal de trading
> automático sin supervisión.

**Errores**
| Código | Caso |
|---|---|
| 422 | fecha inválida, features faltantes/extra, valores NaN/Inf/no numéricos |
| 503 | modelo/clasificador no entrenado (ejecutar fases 15-16 y notebook 16) |

## Validación de entradas
- Esquema Pydantic estricto (`PredictRequest`).
- El servidor verifica que el conjunto de features coincida exactamente con
  `models/feature_list.json` (mismo orden que entrenamiento).
- Rechaza NaNs, Inf, claves extra y fechas malformadas con mensaje descriptivo.

## Compatibilidad entrenamiento/servicio
La API carga los artefactos serializados (`final_model.joblib`, `preprocessor.joblib`,
`direction_classifier.joblib`) y aplica `transform` del preprocesador entrenado:
**cero divergencia** entre entrenamiento e inferencia (regla de oro de la fase 21).

## Tests
```bash
pytest tests/ -q        # incluye validación de la API (TestClient)
```

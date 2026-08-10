# Model Card — Gold Price Prediction (Ridge)

## Resumen
Modelo de regresión para predecir el **precio spot del oro (USD/oz)** a 1 día hábil
(horizontes 5 y 21 implementados en features). Entrenado con datos 2000-2022,
evaluado en test 2023-2025.

## Uso previsto
- Referencia cuantitativa de nivel para analistas (sizing, stress testing, alertas).
- Investigación/educación en forecasting financiero.
- Predicción batch/diaria vía API o CLI.

## Uso NO previsto
- **Trading automático** o decisiones de compra/venta sin supervisión humana
  (la directional accuracy ≈ 47-48% no supera el azar).
- Horizontes > 21 días.
- Predicción en regímenes sin precedentes (el modelo extrapola mal).

## Datos
- 45,368 filas crudas (1901-2025) → ventana 2000-2025 → 6,705 días hábiles.
- 60 features exógenas (tipos, FX, materias primas, índices, macro).
- Fuente: compilación pública; sin datos personales.

## Métricas (test bloqueado 2023-2025)

| Métrica | Valor |
|---|---|
| MAE | 204.36 USD/oz |
| RMSE | 241.81 USD/oz |
| sMAPE | 8.28% |
| R² | 0.758 |
| DA | 47.4% |
| vs naive | −76.8% MAE |

## Subgrupos / segmentos
| Segmento | MAE |
|---|---|
| 2023 | 86.6 |
| 2024 | 204.8 |
| 2025 | 392.6 |

Degradación creciente: los segmentos recientes (rally 2024-25) están fuera de
la distribución de entrenamiento.

## Limitaciones
1. Dirección diaria impredecible (DA ≈ azar).
2. Drift de mercado: requiere reentrenamiento periódico.
3. Intervalos de incertidumbre mal calibrados (cobertura P5-P95 ≈ 1%).
4. Features macro publicadas con retraso (ffill).

## Consideraciones éticas
- Sin datos personales ni variables sensibles.
- El modelo no debe presentarse como asesor financiero; el error medio de
  ~204 USD/oz (y hasta 600 en 2025) debe comunicarse a los usuarios.

## Mantenimiento
- Reentrenar cuando el MAE rodante (60d) supere 1.5× el MAE de test o cuando
  el KS-drift sea persistente (script `scripts/monitor_drift.py`).
- Versionado de artefactos (`models/`), rollback conservando el modelo anterior.
- Propietario: equipo de datos (proyecto educativo).

## Versiones
- v1.0.0 (2025): Ridge α=0.0022, 85 features, RobustScaler, h=1.

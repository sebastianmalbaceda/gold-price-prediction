# Changelog

Todas las versiones notables de este proyecto se documentan aquí.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.1.0/) y el
proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [1.0.0] - 2025-08-11

### Añadido

- Pipeline completo de predicción del precio spot del oro (USD/oz) a 1, 5 y
  21 días hábiles con 60+ variables exógenas financieras.
- 20 notebooks ejecutados que cubren las 23 fases metodológicas
  (definición del problema, auditoría de datos, EDA, target/features,
  métricas, partición temporal, preprocessing, feature engineering,
  selección, baselines, modelado, validación, tuning, selección de modelo,
  entrenamiento final, test bloqueado, análisis de errores, robustez,
  ética, despliegue, documentación y monitorización).
- Modelo de regresión (Ridge) para niveles: MAE 204.70 USD/oz en test
  (2023-2025), R² = 0.757.
- Clasificador de dirección (RandomForest calibrado): AUC 0.552 en test.
- Modelo de volatilidad realizada con position sizing: Sharpe 1.88 vs 1.61
  de buy & hold.
- Auditoría de datos temporales: frecuencias de actualización, lookahead
  bias en variables macro, multicolinealidad (VIF) y rango de fechas óptimo.
- API REST (FastAPI) con endpoints `/predict` y `/predict_direction`,
  validación estricta de entradas.
- CLI de predicción (`scripts/predict.py`) y monitorización de drift
  (`scripts/monitor_drift.py`).
- Tests unitarios (25), CI en GitHub Actions (pytest + flake8 + black),
  Dockerfile y docker-compose.
- Documentación completa: README, guía metodológica, informe técnico,
  model card, data card y manual de API.

### Corregido

- Columna constante (`policy_uncertainty_missing`) eliminada del modelo
  (varianza 0 en train y test).
- Incoherencias de métricas entre documentación y reports resueltas.
- Lookahead bias en variables macro documentado con impacto cuantificado
  (+0.003 AUC) y corrección propuesta (`PUBLICATION_LAG`).

## [0.9.0] - 2025-08-10

### Añadido

- Clasificador de dirección (sube/baja) reutilizando el pipeline existente.
- Backtest con retornos alineados (hoy -> mañana), costes de transacción
  y tests de significancia estadística.
- Verificación de robustez: gap train/val/test, learning curve con
  validación fija (25 puntos), walk-forward 2019-2025 y análisis por
  régimen de mercado.
- Reestructuración del repositorio: notebooks autocontenidos, guía
  metodológica en `docs/methodology-guide.md`, eliminación de código
  redundante.

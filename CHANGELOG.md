# Changelog

Todas las versiones notables de este proyecto se documentan aquÃ­.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.1.0/) y el
proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).


## [1.1.0] - 2025-08-18

### Añadido

- Extension opcional de deep learning con MLP tabular y GRU causal.
- Deteccion automatica de CUDA/CPU y registro del dispositivo usado.
- Notebook `16c_deep_learning_clasificacion.ipynb` y metadatos de entrenamiento.
- Dependencias opcionales en `requirements-dl.txt` y `.[dl]`.

### Evaluacion

- MLP: validation AUC 0.544, test AUC 0.538.
- GRU: validation AUC 0.529, test AUC 0.528.
- No reemplazan al clasificador clasico porque no mejoran su AUC fuera de muestra.

## [1.0.0] - 2025-08-11

### AÃ±adido

- Pipeline completo de predicciÃ³n del precio spot del oro (USD/oz) a 1, 5 y
  21 dÃ­as hÃ¡biles con 60+ variables exÃ³genas financieras.
- 20 notebooks ejecutados que cubren las 23 fases metodolÃ³gicas
  (definiciÃ³n del problema, auditorÃ­a de datos, EDA, target/features,
  mÃ©tricas, particiÃ³n temporal, preprocessing, feature engineering,
  selecciÃ³n, baselines, modelado, validaciÃ³n, tuning, selecciÃ³n de modelo,
  entrenamiento final, test bloqueado, anÃ¡lisis de errores, robustez,
  Ã©tica, despliegue, documentaciÃ³n y monitorizaciÃ³n).
- Modelo de regresiÃ³n (Ridge) para niveles: MAE 204.70 USD/oz en test
  (2023-2025), R^2 = 0.757.
- Clasificador de direcciÃ³n (RandomForest calibrado): AUC 0.552 en test.
- Modelo de volatilidad realizada con position sizing: Sharpe 1.88 vs 1.61
  de buy & hold.
- AuditorÃ­a de datos temporales: frecuencias de actualizaciÃ³n, lookahead
  bias en variables macro, multicolinealidad (VIF) y rango de fechas Ã³ptimo.
- API REST (FastAPI) con endpoints `/predict` y `/predict_direction`,
  validaciÃ³n estricta de entradas.
- CLI de predicciÃ³n (`scripts/predict.py`) y monitorizaciÃ³n de drift
  (`scripts/monitor_drift.py`).
- Tests unitarios (25), CI en GitHub Actions (pytest + flake8 + black),
  Dockerfile y docker-compose.
- DocumentaciÃ³n completa: README, guÃ­a metodolÃ³gica, informe tÃ©cnico,
  model card, data card y manual de API.

### Corregido

- Columna constante (`policy_uncertainty_missing`) eliminada del modelo
  (varianza 0 en train y test).
- Incoherencias de mÃ©tricas entre documentaciÃ³n y reports resueltas.
- Lookahead bias en variables macro documentado con impacto cuantificado
  (+0.003 AUC) y correcciÃ³n propuesta (`PUBLICATION_LAG`).

## [0.9.0] - 2025-08-10

### AÃ±adido

- Clasificador de direcciÃ³n (sube/baja) reutilizando el pipeline existente.
- Backtest con retornos alineados (hoy -> maÃ±ana), costes de transacciÃ³n
  y tests de significancia estadÃ­stica.
- VerificaciÃ³n de robustez: gap train/val/test, learning curve con
  validaciÃ³n fija (25 puntos), walk-forward 2019-2025 y anÃ¡lisis por
  rÃ©gimen de mercado.
- ReestructuraciÃ³n del repositorio: notebooks autocontenidos, guÃ­a
  metodolÃ³gica en `docs/methodology-guide.md`, eliminaciÃ³n de cÃ³digo
  redundante.

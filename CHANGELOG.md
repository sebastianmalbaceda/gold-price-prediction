# Changelog

Todas las versiones notables de este proyecto se documentan aqui.

El formato sigue Keep a Changelog y Semantic Versioning.

## [1.1.0] - 2025-08-18

### Aniadido

- Extension de deep learning con MLP tabular y GRU causal.
- Deteccion automatica de CUDA y CPU con registro del dispositivo utilizado.
- Notebook 16c y metadatos de entrenamiento.
- PyTorch incluido en los requisitos normales del proyecto.

### Evaluacion

- MLP: validation AUC 0.550, test AUC 0.534.
- GRU: validation AUC 0.543, test AUC 0.510.
- No reemplazan al clasificador clasico porque no mejoran su AUC fuera de muestra.

## [1.0.0] - 2025-08-11

### Aniadido

- Pipeline completo de prediccion del precio spot del oro (USD/oz) a 1, 5 y
  21 dias habiles con 59 variables exogenas financieras.
- 20 notebooks ejecutados que cubren las 23 fases metodologicas.
- Modelo de regresion Ridge para niveles: MAE 99.75 USD/oz en test,
  R2 = 0.935.
- Clasificador de direccion calibrado: AUC 0.532 en test. Esta cifra proviene
  de `models/direction_metrics.json`, que no está versionado; no es auditable
  sin reejecutar el pipeline.
- Modelo de volatilidad realizada con position sizing: Sharpe 1.83 vs 1.61
  de buy and hold.
- Auditoria de datos temporales, API REST, CLI, monitorizacion, tests,
  CI, Docker y documentacion academica.

### Corregido

- Columna constante policy_uncertainty_missing eliminada del modelo.
- Incoherencias de metricas entre documentacion y reports resueltas.
- Lookahead bias de variables macro documentado con impacto cuantificado.

## [0.9.0] - 2025-08-10

### Aniadido

- Clasificador de direccion y backtest con retornos alineados.
- Verificacion de robustez, walk-forward y analisis por regimen.
- Reorganizacion del repositorio y notebooks autocontenidos.

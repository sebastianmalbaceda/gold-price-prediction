"""
Gold Price Prediction - paquete fuente.

Serie temporal financiera: predicción del precio spot del oro (gold_spot)
con 60 variables exógenas (tipos, FX, materias primas, macro, sentimiento).

Estructura:
    src/config.py       - carga de configuraciones YAML
    src/data/           - carga y limpieza de datos
    src/features/       - feature engineering (lags, rolling, calendario)
    src/models/         - entrenamiento, tuning y persistencia
    src/evaluation/     - métricas, baselines y análisis de errores
    src/api/            - servicio de inferencia REST (FastAPI)
"""

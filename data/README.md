# Datos del proyecto Gold Price Prediction
# =========================================
#
# Este directorio contiene los datos en tres niveles (inmutabilidad):
#
# - `raw/`       -> datos ORIGINALES descargados. NUNCA se modifican.
#   - `gold-price-prediction-dataset.csv`: 45,368 filas diarias (1901-06-30 a
#     2025-09-14) x 61 columnas (fecha + gold_spot + 60 exógenas financieras:
#     tipos, FX, materias primas, índices, macro).
#
# - `interim/`   -> datos transformados en pasos intermedios.
#   - `gold_clean.parquet`: serie limpia (2000-2025, días hábiles, ffill de
#     exógenas, features excluidas por cobertura).
#
# - `processed/` -> conjunto final usado para entrenamiento/evaluación.
#   - `features.parquet`: features + targets (lags, retornos, rolling,
#     calendario, indicadores de ausencia).
#   - `train.parquet` / `val.parquet` / `test.parquet`: particiones
#     temporales estrictas (2000-2019 / 2020-2022 / 2023-2025).
#
# Regeneración (idempotente, desde la raíz del proyecto):
#   python -m src.data.load_data          # raw -> interim
#   python -m src.features.build_features # interim -> processed
#
# Licencia: el dataset es una compilación pública de series de mercado para
# uso académico. No contiene datos personales. Ver LICENSE.

# Gold Price Prediction — Serie Temporal Financiera

Predicción del **precio spot del oro (USD/oz)** a 1, 5 y 21 días hábiles mediante
aprendizaje supervisado de regresión con 60 variables exógenas financieras
(tipos de interés, divisas, materias primas, índices, macroeconomía).

Proyecto completo de ML siguiendo el índice maestro
[`Indice-para-proyectos-de-ML-IA.md`](Indice-para-proyectos-de-ML-IA.md):
las 23 fases (definir → auditar → dividir → aprender solo con train → seleccionar
con validación/CV → comprobar una vez con test → empaquetar → monitorizar).

---

## 📈 Resultados principales

### Métricas finales en TEST bloqueado (2023-01 → 2025-09, 684 días hábiles)

| Modelo | MAE (USD/oz) | RMSE (USD/oz) | sMAPE | R² | Directional Acc. |
|---|---|---|---|---|---|
| **Naive (último valor)** | 879.67 | 1,007.6 | 42.6% | — | 0.7% |
| **Ridge (final)** | **204.36** | **241.81** | **8.28%** | **0.758** | 47.4% |
| XGBoost | 199.7 (val) | 212.0 | 11.7% (val) | −3.3 | 51.2% |

- **Reducción del MAE del 76.8%** frente al baseline naive.
- R² = 0.758: el modelo captura la tendencia y el nivel del oro.
- La **Directional Accuracy ≈ 47-48%** (peor que 50%) indica que el signo del
  movimiento diario es esencialmente impredecible con estos datos (eficiencia
  de mercado): el modelo es útil para **nivel**, no para *timing* diario.

### Degradación temporal (test, modelo congelado)

| Año | MAE | sMAPE |
|---|---|---|
| 2023 | 86.6 | 4.5% |
| 2024 | 204.8 | 8.9% |
| 2025 | 392.6 | 13.4% |

→ Drift de mercado fuerte (2024-25: rally del oro de 1,800 a 2,700+ USD/oz).
**Reentrenamiento periódico obligatorio** (fase 23).

### Explicabilidad (SHAP, top 5)

1. `gold_spot_lag21` (precio del oro hace 1 mes) — domina
2. `gold_ret_roll21` (retorno mensual)
3. `silver_spot` (plata — hermana del oro)
4. `sp500_futures` (apetito de riesgo)
5. `consumer_confidence`, `us_gdp` (macroeconomía)

---

## 🧠 Decisiones técnicas clave

| Decisión | Justificación |
|---|---|
| **Split temporal estricto** (train 2000-2019 / val 2020-2022 / test 2023-2025) | Forecasting: nunca datos futuros en train |
| **Modelo final: Ridge** (lineal regularizado) | MAE 37 (val) vs 199-248 de XGBoost/LGBM/CatBoost/RF: los árboles no extrapolan niveles no estacionarios |
| **Features**: lags + retornos + rolling del target, exógenas con lag 1 + retorno, calendario, indicadores de ausencia | Sin leakage (todo hacia atrás) |
| **Forward-fill** (solo pasado) para exógenas con huecos | Datos macro publicados con retraso |
| **Ventana 2000-2025** (6,705 días hábiles) | Cobertura fiable; antes de 2000 no hay datos |
| **85 features** tras filtro de redundancia (|ρ|>0.98) | Reduce colinealidad |

---

## 🚀 Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Python 3.11+ · numpy, pandas, scikit-learn, xgboost, lightgbm, catboost,
statsmodels, optuna, shap, matplotlib, seaborn, fastapi, uvicorn, pytest, jupyterlab.

## ▶️ Ejecución

```bash
# 0) Auditoría y limpieza -> data/interim
python -m src.data.load_data

# 1) Features + targets -> data/processed
python -m src.features.build_features

# 2) Entrenar modelo final (usa modelos/feature_list.json)
#    (los notebooks 01-17 hacen el flujo completo paso a paso)

# 3) API REST
uvicorn src.api.main:app --reload
#    GET  /health
#    POST /predict  {"date": "2025-08-14", "features": {...}}

# 4) Predicción CLI
python scripts/predict.py --date 2025-09-12

# 5) Monitorización de drift
python scripts/monitor_drift.py --window 60

# 6) Tests
pytest tests/ -q
```

## 📓 Notebooks (una fase por notebook)

| Notebook | Fase |
|---|---|
| `01_problema_y_diseno.ipynb` | 0-1. Problema, contexto, diseño y reproducibilidad |
| `02_datos_y_auditoria.ipynb` | 2-3. Obtención, licencia, ingesta y auditoría |
| `04_eda.ipynb` | 4. EDA y dominio (target, features, TS) |
| `05_target_y_metricas.ipynb` | 5-6. Target, variables, métricas y éxito |
| `07_particion.ipynb` | 7. Partición temporal y protocolo |
| `08_10_preprocessing_features.ipynb` | 8-10. Preprocessing, features y selección |
| `11_13_baselines_modelado.ipynb` | 11-13. Baselines, modelado y CV |
| `14_tuning.ipynb` | 14. Hyperparameter tuning (Optuna) |
| `15_16_seleccion_entrenamiento.ipynb` | 15-16. Selección y entrenamiento final |
| `17_test_final.ipynb` | 17. Test final bloqueado |
| `18_errores_explicabilidad.ipynb` | 18. Errores, SHAP e incertidumbre |
| `19_20_robustez_etica.ipynb` | 19-20. Robustez, ética y seguridad |
| `21_despliegue.ipynb` | 21. Empaquetado y API |
| `22_documentacion.ipynb` | 22. Documentación |
| `23_monitorizacion.ipynb` | 23. Monitorización y reentrenamiento |

## 📂 Estructura

```
├── configs/            # config.yaml (rutas/split/features) + params.yaml (hiperparámetros)
├── data/raw|interim|processed/
├── notebooks/          # 15 notebooks ejecutados (fuentes en notebooks/_src/)
├── src/                # data/, features/, models/, evaluation/, api/
├── models/             # final_model.joblib, preprocessor.joblib, feature_list.json
├── reports/            # métricas (JSON/CSV) y figures/ (EDA, SHAP, errores)
├── scripts/            # predict.py, monitor_drift.py, build_notebooks.py
├── tests/              # pytest
└── docs/               # informe técnico, model card, data card, manual API
```

## ⚠️ Limitaciones

1. **No sirve para trading automático**: la dirección diaria no es predecible
   (DA ≈ 47-48%). Uso analítico/de referencia.
2. **Drift de mercado**: el error crece con el tiempo (2025: MAE 393).
   Requiere reentrenamiento y monitorización continua.
3. **Incertidumbre mal calibrada**: los intervalos P5-P95 (bootstrap) solo
   cubren ~1% de las observaciones en test; la volatilidad real 2024-25
   excede la incertidumbre paramétrica del modelo.
4. **Horizonte único desplegado**: la API sirve h=1; los horizontes 5 y 21
   están implementados en las features pero no expuestos en la API.
5. **Datos macro con retraso**: CPI, PIB, etc. se forward-fillean; el modelo
   no ve las revisiones posteriores (como en producción real).

## 📄 Licencia y datos

- Dataset: `data/raw/gold-price-prediction-dataset.csv` (compilación pública,
  uso académico). No contiene datos personales.
- Código: ver `LICENSE`.

## 📚 Referencias

- Índice maestro de fases: `Indice-para-proyectos-de-ML-IA.md`
- Informe técnico completo: [`docs/informe_tecnico.md`](docs/informe_tecnico.md)
- Model card: [`docs/model_card.md`](docs/model_card.md)
- Data card / diccionario: [`docs/data_card.md`](docs/data_card.md)
- Manual API: [`docs/api_manual.md`](docs/api_manual.md)

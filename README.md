# Gold Price Prediction - Serie Temporal Financiera

Predicción del **precio spot del oro (USD/oz)** a 1, 5 y 21 días hábiles mediante
aprendizaje supervisado de regresión con 60 variables exógenas financieras
(tipos de interés, divisas, materias primas, índices, macroeconomía).

Proyecto completo de ML siguiendo el índice maestro
[`docs/methodology-guide.md`](docs/methodology-guide.md):
las 23 fases (definir -> auditar -> dividir -> aprender solo con train -> seleccionar
con validación/CV -> comprobar una vez con test -> empaquetar -> monitorizar).

---

## Resultados principales

### Verificación fuerte de generalización (fase 16b)

Diagnóstico honesto con 4 pruebas (detalle en `notebooks/16_direccion_clasificacion.ipynb`):

| Prueba | Resultado | Conclusión |
|---|---|---|
| Gap train/val/test (MAE) | train 14.7  |  val 21.5  |  test 204.7 | El gap train->test es **drift de mercado**, no overfitting severo (val gap pequeño) |
| Learning curve (CV temporal) | val MAE 15->42->89->58->42 al crecer train | **No hay overfitting severo**: el val no empeora sistemáticamente con más datos |
| Gap por familia | Ridge 23  |  RF 264  |  XGB 200 | Ridge es la familia con **menor overfitting** (por eso ganó) |
| vs naive-persistencia | naive MAE=17.6 vs modelo 204.7 | **El modelo NO supera a "mañana = hoy"** en h=1 |

**Conclusión honesta**: el R^2 alto en train (0.998) y test (0.757) refleja que el
modelo sigue la **tendencia**, pero el cambio diario del oro es **ruido
impredecible** (mercado eficiente). Para decisión de sube/baja, la regresión
no es útil (dirección implícita 45%). **Por eso se añadió el clasificador de
dirección** (ver siguiente sección).

### Clasificador de dirección (sube/baja) - fase 16b

Modelo de clasificación binaria (RandomForest calibrado) que predice
P(gold(t+1) > gold(t)) reutilizando las mismas features y split:

| Horizonte | CV AUC | Test AUC | Test ACC | P(sube) |
|---|---|---|---|---|
| h=1 | 0.528 | **0.552** | 0.551 | 0.537 |
| h=5 | 0.532 | 0.511 | 0.586 | 0.573 |
| h=21 | 0.571 | 0.535 | 0.689 | 0.677 |

- **h=1 (desplegado): Test AUC=0.552, ACC=0.551, recall=0.790, PR-AUC=0.585**.
- Señal **débil pero real** (AUC > 0.5), coherente con eficiencia de mercado.
- La señal proviene de momentum (`gold_ret_lag1`, `gold_ret_roll63`) y riesgo
  geopolítico (`geopolitical_risk`, `policy_uncertainty`).
- **Uso recomendado**: inclinación leve (p.ej. para alertas), NO como señal de
  trading automático. La probabilidad calibrada se sirve en `/predict_direction`.

### ¿Es rentable seguir la tendencia predicha? (fase 17b - backtest riguroso)

Análisis completo en `notebooks/17b_acierto_y_backtest.ipynb`:

**Acierto del clasificador (test, umbral 0.5):**

| Métrica | Valor |
|---|---|
| Acierto cuando REALMENTE sube (TPR) | **83.4%** |
| Acierto cuando REALMENTE baja (TNR) | 23.7% |
| Precisión al predecir sube | 55.8% |
| Acierto global | 55.7% |

El modelo está **sesgado a predecir sube** (80% de las veces): acierta muy
bien las subidas pero falla las bajadas. Subir el umbral a 0.55-0.60 mejora
la precisión (59-67%) a costa de operar menos.

**Backtest (retorno correctamente alineado hoy->mañana, costes 0.1%/op):**

| Estrategia | Retorno | Sharpe | vs Buy&Hold |
|---|---|---|---|
| Buy & hold (referencia) | **+82.9%** | - | - |
| LONG filtrado (clf, umbral 0.5) | +74.4% | 1.63 | -8.5 pp |
| LONG filtrado (RF profundo, umbral 0.51) | +48.7% | 1.60 | -34 pp |
| LONG filtrado (RF profundo, umbral 0.55) | +18.7% | 0.98 | -64 pp |

**Significancia estadística:**

- AUC test RF profundo = **0.571** (IC95% bootstrap [0.456, 0.545] - roza 0.5).
- Test de permutación: p < 0.001 (la señal no es casualidad en test).
- **CV temporal: AUC medio ~ 0.516** (folds 0.51-0.53) -> la señal **NO es estable** fuera de test.
- t-test de retornos estrategia vs buy&hold: **p = 0.25 (no significativo)** -> la
  estrategia NO rinde más que mantener.

**Conclusión honesta (fase 17b):**

1. El clasificador **distingue dirección mejor que el azar** en test (AUC 0.57),
   y la confianza es informativa: P(sube|pred) sube de 54% a **64-77%** cuando
   el modelo está seguro.
2. **PERO no es rentable frente a comprar y mantener**: la señal es débil e
   inestable (CV ~0.52), y el coste de oportunidad de salir del mercado supera
   la ganancia de precisión, sobre todo en mercados con tendencia alcista.
3. **Uso recomendado**: indicador de **riesgo/inclinación** (alertas, sizing
   marginal, gestión de exposición), NO como estrategia de trading. La
   regresión de niveles sirve como referencia de tendencia a medio plazo.

### Verificación de robustez: overfitting/underfitting (fase 17c)

Diagnóstico completo en `notebooks/17c_robustez_direccion.ipynb`:

**¿Overfitting?** El modelo original (RF depth=8) **SÍ sobreajustaba**:

| Modelo | Train AUC | Test AUC | Gap |
|---|---|---|---|
| RF profundo (depth=None) | 1.000 | 0.571 | **0.429** |
| RF original (depth=8) | 0.911 | 0.559 | **0.352** |
| **RF regularizado (depth=4, leaf=20)** | **0.695** | **0.569** | **0.126** |

La regularización reduce el overfitting **sin perder test AUC** (0.569).
La learning curve muestra que el gap se reduce al crecer train (0.29->0.19).

**¿Underfitting? NO.** Más datos mejoran el test AUC (0.537 con 30% de train
-> 0.557 con 100%). El modelo no es demasiado simple.

**Generalización real (walk-forward 2019-2025, reentrenando cada año):**

- AUC medio: **0.539 +/- 0.057** (solo 2/7 años superan 0.55).
- La estrategia gana a buy&hold en **1/7 años**; estrategia media +5.5%/año
  vs buy&hold +15.3%/año.
- Por régimen: alcista AUC=0.548  |  bajista AUC=0.438  |  lateral AUC=0.484.
  El modelo funciona mejor con tendencia alcista (momentum).

**Conclusión de robustez**: la señal es **real pero débil** (AUC ~0.54 en
walk-forward). El modelo regularizado no memoriza ni es demasiado simple:
es **lo mejor posible con estos datos**, y lo mejor posible es una señal
modesta. Uso práctico recomendado: **indicador de riesgo** (reducir
exposición cuando P<0.5, alertas), no estrategia de trading.

### Extension de deep learning (fase 16c)

El notebook `16c_deep_learning_clasificacion.ipynb` compara una MLP tabular y
una GRU causal con el clasificador clasico. La ejecucion detecta CUDA y usa
la GPU NVIDIA disponible; en entornos sin CUDA utiliza CPU automaticamente.

| Modelo | Validation AUC | Test AUC | Dispositivo observado |
|---|---:|---:|---|
| MLP | 0.544 | 0.538 | CUDA |
| GRU | 0.529 | 0.528 | CUDA |

La MLP gana por validation, pero no supera al RandomForest regularizado de la
fase 17c. Por rigor, se conserva como experimento DL y no reemplaza el modelo
operativo. Instalar con `pip install -r requirements-dl.txt` o
`pip install -e .[dl]`.

### Predicción de volatilidad y gestión de riesgo (fase 17d)

La **volatilidad** es mucho más predecible que la dirección (autocorrelación
lag-1 = 0.985, *volatility clustering*). Notebook `17d_volatilidad_riesgo.ipynb`:

| Modelo | MAE (vol) | R^2 |
|---|---|---|
| Naive (vol actual) | 0.0553 | -0.212 |
| **RF volatilidad (h=5)** | **0.0510** | **+0.058** |

- El modelo supera al naive (**+7.9% MAE**) y captura los picos de 2024-25.
- **Position sizing por volatilidad** (exposición inversa a vol prevista,
  target 12% anualizado):

| Estrategia | Retorno | Sharpe | Max Drawdown | Exposición |
|---|---|---|---|---|
| Buy & hold | +86.4% | 1.61 | -11.3% | 100% |
| **Sizing por volatilidad** | +76.8% | **1.88** | **-10.3%** | 80.7% |

El sizing **mejora el Sharpe (1.88 vs 1.61) y reduce el drawdown** con menos
exposición - la gestión de riesgo basada en volatilidad SÍ añade valor
práctico, a diferencia de la señal de dirección pura.

### Auditoría de datos temporales (fase 18b)

Análisis exhaustivo en `notebooks/18b_auditoria_datos.ipynb`:

**Frecuencias de actualización** (días entre cambios): 26 diarias  |  13 mensuales
(CPI, paro, M2, retail)  |  1 semanal  |  PIB trimestral (92 días).

**Lookahead bias detectado**: CPI, paro y PIB se actualizan **siempre el día 1
del mes** en el dataset, pero en realidad se publican con 5-30 días de retraso.
El forward-fill rellena huecos pero NO corrige este adelanto. Impacto medido:
+0.003 AUC (pequeño, la señal principal es el momentum del oro). Corrección
propuesta: `PUBLICATION_LAG` (desplazar cada macro k días hábiles).

**Multicolinealidad**: 24/110 features con VIF > 10 (us_gdp 79.9, sp500 73.3,
usdcny 72.0, gold_spot_lag21 60.9). Esperable en series financieras (lags del
mismo activo); tolerada por Ridge (regularización L2) y árboles.

**Rango de fechas óptimo confirmado**: ventana 2000-2025 con cobertura 100%
en todas las features tras warm-up. Antes de 2000 las series no existen.

### Métricas de la regresión en TEST bloqueado (2023-01 -> 2025-09, 684 días hábiles)

| Modelo | MAE (USD/oz) | RMSE (USD/oz) | sMAPE | R^2 | Directional Acc. |
|---|---|---|---|---|---|
| **Naive (último valor)** | 879.67 | 1,007.6 | 42.6% | - | 0.7% |
| **Ridge (final)** | **204.70** | **242.18** | **8.29%** | **0.7569** | 47.3% |
| XGBoost | 199.7 (val) | 212.0 | 11.7% (val) | -3.3 | 51.2% |

- **Reducción del MAE del 76.8%** frente al baseline naive del informe original
  (último valor de train+val, 2022).
- R^2 = 0.757: el modelo captura la tendencia y el nivel del oro.
- **Limitación documentada**: frente al naive-persistencia diario
  (gold_spot(t), MAE=17.6) el modelo NO gana en h=1; su valor está en el
  seguimiento de tendencia a medio plazo y como referencia de nivel.

### Degradación temporal (test, modelo congelado)

| Año | MAE | sMAPE |
|---|---|---|
| 2023 | 86.6 | 4.5% |
| 2024 | 204.8 | 8.9% |
| 2025 | 392.6 | 13.4% |

-> Drift de mercado fuerte (2024-25: rally del oro de 1,800 a 2,700+ USD/oz).
**Reentrenamiento periódico obligatorio** (fase 23).

### Explicabilidad (SHAP, top 5)

1. `gold_spot_lag21` (precio del oro hace 1 mes) - domina
2. `gold_ret_roll21` (retorno mensual)
3. `silver_spot` (plata - hermana del oro)
4. `sp500_futures` (apetito de riesgo)
5. `consumer_confidence`, `us_gdp` (macroeconomía)

---

## Decisiones técnicas clave

| Decisión | Justificación |
|---|---|
| **Split temporal estricto** (train 2000-2019 / val 2020-2022 / test 2023-2025) | Forecasting: nunca datos futuros en train |
| **Modelo final: Ridge** (lineal regularizado) | MAE 37 (val) vs 199-248 de XGBoost/LGBM/CatBoost/RF: los árboles no extrapolan niveles no estacionarios |
| **Features**: lags + retornos + rolling del target, exógenas con lag 1 + retorno, calendario, indicadores de ausencia | Sin leakage (todo hacia atrás) |
| **Forward-fill** (solo pasado) para exógenas con huecos | Datos macro publicados con retraso |
| **Ventana 2000-2025** (6,705 días hábiles) | Cobertura fiable; antes de 2000 no hay datos |
| **85 features** tras filtro de redundancia (|rho|>0.98) | Reduce colinealidad |

---

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Python 3.11+  |  numpy, pandas, scikit-learn, xgboost, lightgbm, catboost,
statsmodels, optuna, shap, matplotlib, seaborn, fastapi, uvicorn, pytest, jupyterlab.

## Ejecución

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
#    POST /predict           {"date": "2025-08-14", "features": {...}}
#    POST /predict_direction  {"date": "2025-08-14", "features": {...}}  -> P(sube)

# 4) Predicción CLI
python scripts/predict.py --date 2025-09-12

# 5) Monitorización de drift
python scripts/monitor_drift.py --window 60

# 6) Tests
pytest tests/ -q
```

## Notebooks (una fase por notebook)

| Notebook | Fase |
|---|---|
| `01_problema_y_diseno.ipynb` | 0-1. Problema, contexto, diseño y reproducibilidad |
| `02_datos_y_auditoria.ipynb` | 2-3. Obtención, licencia, ingesta y auditoría |
| `04_eda.ipynb` | 4. EDA y dominio (target, features, TS, scatterplots, pairplot, outliers, Spearman, PACF) |
| `05_target_y_metricas.ipynb` | 5-6. Target, variables, métricas y éxito |
| `07_particion.ipynb` | 7. Partición temporal y protocolo |
| `08_10_preprocessing_features.ipynb` | 8-10. Preprocessing, features, selección (PCA, Mutual Information, clustering por correlación) |
| `11_13_baselines_modelado.ipynb` | 11-13. Baselines, modelado y CV |
| `14_tuning.ipynb` | 14. Hyperparameter tuning (Optuna) |
| `15_16_seleccion_entrenamiento.ipynb` | 15-16. Selección y entrenamiento final |
| `16_direccion_clasificacion.ipynb` | **16b. Verificación de generalización + clasificación de dirección (sube/baja)** |
| `16c_deep_learning_clasificacion.ipynb` | **16c. Deep learning opcional: MLP y GRU con CUDA/CPU** |
| `17b_acierto_y_backtest.ipynb` | **17b. Acierto por clase, backtest de rentabilidad y significancia estadística** |
| `17c_robustez_direccion.ipynb` | **17c. Verificación de robustez (overfitting/underfitting), walk-forward y régimen** |
| `17d_volatilidad_riesgo.ipynb` | **17d. Predicción de volatilidad y position sizing (gestión de riesgo)** |
| `17_test_final.ipynb` | 17. Test final bloqueado |
| `18_errores_explicabilidad.ipynb` | 18. Errores, SHAP e incertidumbre |
| `18b_auditoria_datos.ipynb` | **18b. Auditoría de datos temporales: frecuencias, lookahead bias, multicolinealidad** |
| `19_20_robustez_etica.ipynb` | 19-20. Robustez, ética y seguridad |
| `21_despliegue.ipynb` | 21. Empaquetado y API |
| `22_documentacion.ipynb` | 22. Documentación |
| `23_monitorizacion.ipynb` | 23. Monitorización y reentrenamiento |

## Tests y calidad

```bash
pytest tests/ -q        # 20 tests (split, métricas, features, limpieza, API)
python -m compileall -q src scripts tests   # verificación de sintaxis
```

CI en GitHub Actions (`.github/workflows/ci.yml`): tests + flake8 + black.
Pre-commit hooks en `.pre-commit-config.yaml` (opcional).

## Despliegue con Docker

```bash
docker build -t gold-api .
docker compose up -d     # API en http://localhost:8000
```

## Ficheros de repo

`LICENSE` (MIT)  |  `pyproject.toml`  |  `pytest.ini`  |  `Makefile`  | 
`.env.example`  |  `.pre-commit-config.yaml`  |  `Dockerfile`  | 
`docker-compose.yml`  |  `.github/workflows/ci.yml`  |  `data/README.md`

## Estructura

```
|---- configs/            # config.yaml (rutas/split/features) + params.yaml (hiperparámetros)
|---- data/raw|interim|processed/
|---- notebooks/          # 19 notebooks ejecutados (el corazón del proyecto)
|---- src/                # data/, features/, models/, evaluation/, api/
|---- models/             # final_model.joblib, preprocessor.joblib, feature_list.json
|---- reports/            # métricas (JSON/CSV) y figures/ (EDA, SHAP, errores)
|---- scripts/            # predict.py (CLI), monitor_drift.py, run_notebooks.sh
|---- tests/              # pytest
+---- docs/               # methodology-guide, informe técnico, model card, data card, manual API
```

## Limitaciones

1. **No sirve para trading automático**: la dirección diaria no es predecible
   (DA ~ 47-48%). Uso analítico/de referencia.
2. **Drift de mercado**: el error crece con el tiempo (2025: MAE 393).
   Requiere reentrenamiento y monitorización continua.
3. **Incertidumbre mal calibrada**: los intervalos P5-P95 (bootstrap) solo
   cubren ~1% de las observaciones en test; la volatilidad real 2024-25
   excede la incertidumbre paramétrica del modelo.
4. **Horizonte único desplegado**: la API sirve h=1; los horizontes 5 y 21
   están implementados en las features pero no expuestos en la API.
5. **Datos macro con retraso**: CPI, PIB, etc. se forward-fillean; el modelo
   no ve las revisiones posteriores (como en producción real).

## Licencia y datos

- Dataset: `data/raw/gold-price-prediction-dataset.csv` (compilación pública,
  uso académico). No contiene datos personales.
- Código: ver `LICENSE`.

## Referencias

- Metodología (23 fases): [`docs/methodology-guide.md`](docs/methodology-guide.md)
- Informe técnico completo: [`docs/informe_tecnico.md`](docs/informe_tecnico.md)
- Model card: [`docs/model_card.md`](docs/model_card.md)
- Data card / diccionario: [`docs/data_card.md`](docs/data_card.md)
- Manual API: [`docs/api_manual.md`](docs/api_manual.md)

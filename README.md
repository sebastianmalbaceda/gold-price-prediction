# Gold Price Prediction - Serie Temporal Financiera

Predicción del **precio spot del oro (USD/oz)** a 1, 5 y 21 días hábiles mediante
aprendizaje supervisado de regresión con 59 variables exógenas financieras
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
| Gap train/val/test (MAE) | train 15.2 | val 24.9 | test 99.7 | El salto temporal refleja **drift de mercado**, no solo overfitting |
| Learning curve (validación fija) | val MAE 256.6 -> 60.4 al crecer train | Más historia mejora la validación, aunque el régimen sigue cambiando |
| Modelo final vs baseline | Ridge test MAE=99.7 frente a naive diario=17.6 | El modelo de nivel no supera la persistencia diaria |
| vs naive-persistencia | naive MAE=17.6 vs modelo 99.7 | **El modelo NO supera a "mañana = hoy"** en h=1 |

**Conclusión honesta**: el R^2 alto en train (0.998) y test (0.935) refleja que el
modelo sigue el **nivel y la tendencia**, pero el cambio diario del oro es
**ruido difícil de predecir** (mercado eficiente). Para decisión de sube/baja, la regresión
no es útil (dirección implícita 45%). **Por eso se añadió el clasificador de
dirección** (ver siguiente sección).

### Clasificador de dirección (sube/baja) - fase 16b

Modelo de clasificación binaria (RandomForest calibrado) que predice
P(gold(t+1) > gold(t)) reutilizando las mismas features y split:

| Horizonte | CV AUC | Test AUC | Test ACC | P(sube) |
|---|---|---|---|---|
| h=1 | 0.529 | **0.573** | 0.558 | 0.537 |
| h=5 | 0.528 | 0.534 | 0.570 | 0.573 |
| h=21 | 0.584 | 0.509 | 0.484 | 0.677 |

- Diagnóstico RF no calibrado h=1: AUC=0.573, ACC=0.558.
- **Artefacto desplegado calibrado**: AUC=0.532, ACC=0.539, recall=0.984, PR-AUC=0.576, Brier=0.247.
- **Trazabilidad de métricas**: estas cifras del artefacto calibrado provienen de `models/direction_metrics.json`, que no está versionado; no son auditables sin reejecutar el pipeline.
- Señal **débil e inestable**: el artefacto calibrado obtiene AUC=0.532 y el walk-forward medio es 0.526.
- La señal proviene de momentum (`gold_ret_lag1`, `gold_ret_roll63`) y riesgo
  geopolítico (`geopolitical_risk`, `policy_uncertainty`).
- **Uso recomendado**: inclinación leve (p.ej. para alertas), NO como señal de
  trading automático. La probabilidad calibrada se sirve en `/predict_direction`.

### ¿Es rentable seguir la tendencia predicha? (fase 17b - backtest riguroso)

Análisis completo en `notebooks/17b_acierto_y_backtest.ipynb`:

**Acierto del clasificador (test, umbral 0.5):**

| Métrica | Valor |
|---|---|
| Acierto cuando REALMENTE sube (TPR) | **98.4%** |
| Acierto cuando REALMENTE baja (TNR) | 2.5% |
| Precisión al predecir sube | 53.9% |
| Acierto global | 53.9% |

El modelo está **sesgado a predecir sube** (98% de las veces con umbral 0.5):
acierta muchas subidas pero falla las bajadas. El umbral 0.55 reduce las
operaciones y eleva la precision condicional al 68.8%, pero la estrategia solo
obtiene +2.4% en este test.

**Backtest (retorno correctamente alineado hoy->mañana, costes 0.1%/op):**

| Estrategia | Retorno | Sharpe | vs Buy&Hold |
|---|---|---|---|
| Buy & hold (referencia) | **+82.9%** | - | - |
| LONG filtrado (clasificador calibrado, umbral 0.5) | +85.2% | 1.60 | +2.3 pp |
| LONG filtrado (clasificador calibrado, umbral 0.55) | +2.4% | 0.31 | -80.5 pp |
| LONG filtrado (RF profundo, umbral 0.53) | +28.0% | 1.24 | -54.9 pp |

**Significancia estadística:**

- AUC test RF profundo = **0.533** (IC95% bootstrap [0.490, 0.574], incluye 0.5).
- El IC95% bootstrap incluye 0.5, por lo que no se confirma una ventaja estadistica.
- **CV temporal: AUC medio 0.526 +/- 0.043** -> la señal **NO es estable** fuera de test.
- t-test de retornos estrategia vs buy&hold: **p = 0.079 (no significativo al 5%)** -> la
  estrategia NO rinde más que mantener.

**Conclusión honesta (fase 17b):**

1. El clasificador no demuestra una ventaja robusta sobre el azar en este test (AUC 0.533),
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
| RF profundo (depth=None) | 1.000 | 0.533 | **0.467** |
| RF original (depth=8) | 0.907 | 0.574 | **0.334** |
| **RF regularizado (depth=4, leaf=20)** | **0.684** | **0.560** | **0.124** |

La regularización reduce el overfitting y mantiene un AUC test de 0.560, aunque la señal sigue siendo débil.
La learning curve muestra que el gap se reduce al crecer train (0.301->0.187).

**¿Underfitting?** No se observa una red demasiado simple; la señal sigue siendo débil y dependiente del régimen.

**Generalización real (walk-forward 2019-2025, reentrenando cada año):**

- AUC medio: **0.526 +/- 0.043** (solo 1/7 años supera 0.55).
- La estrategia gana a buy&hold en **1/7 años**; estrategia media +5.1%/año
  vs buy&hold +15.3%/año.
- Por régimen: alcista AUC=0.538  |  bajista AUC=0.448  |  lateral AUC=0.474.
  El modelo funciona mejor con tendencia alcista (momentum).

**Conclusión de robustez**: la señal es **débil e inestable** (AUC ~0.526 en
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
| MLP | 0.550 | 0.534 | CUDA |
| GRU | 0.543 | 0.510 | CUDA |

La MLP gana por validation, pero no supera al RandomForest regularizado de la
fase 17c. Por rigor, se conserva como experimento DL y no reemplaza el modelo
operativo. La extension utiliza PyTorch incluido en `requirements.txt`. Para una instalacion CUDA especifica, siga las instrucciones de `docs/deep_learning.md`.

### Predicción de volatilidad y gestión de riesgo (fase 17d)

La **volatilidad** es mucho más predecible que la dirección (autocorrelación
lag-1 = 0.985, *volatility clustering*). Notebook `17d_volatilidad_riesgo.ipynb`:

| Modelo | MAE (vol) | R^2 |
|---|---|---|
| Naive (vol actual) | 0.0553 | -0.212 |
| **RF volatilidad (h=5)** | **0.0463** | **+0.241** |

- El modelo supera al naive (**+16.2% MAE**) y captura los picos de 2024-25.
- **Position sizing por volatilidad** (exposición inversa a vol prevista,
  target 12% anualizado):

| Estrategia | Retorno | Sharpe | Max Drawdown | Exposición |
|---|---|---|---|---|
| Buy & hold | +86.4% | 1.61 | -11.3% | 100% |
| **Sizing por volatilidad** | +73.2% | **1.83** | **-10.9%** | 81.6% |

El sizing **mejora el Sharpe (1.83 vs 1.61) y reduce el drawdown** con menos
exposición - la gestión de riesgo basada en volatilidad SÍ añade valor
práctico, a diferencia de la señal de dirección pura.

### Auditoría de datos temporales (fase 18b)

Análisis exhaustivo en `notebooks/18b_auditoria_datos.ipynb`:

**Frecuencias auditadas**: el artefacto actual audita 25 features limpias, todas
con mediana de un día entre cambios; las frecuencias macro crudas aún no se
persisten en `reports/data_audit.json`.

**Lookahead bias detectado**: CPI, paro y PIB se actualizan **siempre el día 1
del mes** en el dataset, pero en realidad se publican con 5-30 días de retraso.
El forward-fill rellena huecos pero NO corrige este adelanto. Impacto medido:
+0.000 AUC en el conjunto actual porque esas variables se excluyen por cobertura. Corrección
propuesta: `PUBLICATION_LAG` (desplazar cada macro k días hábiles).

**Multicolinealidad**: 37/83 features con VIF > 10 (commodities_bloomberg 206.7,
commodities_crb 177.8, gold_spot_lag21 176.9). Es esperable en series financieras;
Ridge y los árboles la toleran mediante regularización o particiones.

**Rango de fechas óptimo confirmado**: ventana 2000-2025 con cobertura 100%
en todas las features tras warm-up. Antes de 2000 las series no existen.

### Métricas de la regresión en TEST bloqueado (2023-01 -> 2025-09, 684 días hábiles)

| Modelo | MAE (USD/oz) | RMSE (USD/oz) | sMAPE | R^2 | Directional Acc. |
|---|---|---|---|---|---|
| **Naive (último valor)** | 879.67 | 1,007.6 | 42.6% | - | 0.7% |
| **Ridge (final)** | **99.75** | **125.08** | **3.94%** | **0.9352** | 46.1% |
| XGBoost | 252.8 (val) | 265.8 (val) | 15.1% (val) | -5.83 | 49.5% |

- **Reducción del MAE del 88.7%** frente al baseline naive congelado en 2022
  (último valor de train+val, 2022).
- R^2 = 0.935: el modelo captura el nivel y parte de la tendencia del oro.
- **Limitación documentada**: frente al naive-persistencia diario
  (gold_spot(t), MAE=17.6) el modelo NO gana en h=1; su valor está en el
  seguimiento de tendencia a medio plazo y como referencia de nivel.

### Degradación temporal (test, modelo congelado)

| Año | MAE | sMAPE |
|---|---|---|
| 2023 | 41.4 | 2.1% |
| 2024 | 99.7 | 4.2% |
| 2025 | 193.4 | 6.4% |

-> Drift de mercado fuerte (2024-25: rally del oro de 1,800 a 2,700+ USD/oz).
**Reentrenamiento periódico obligatorio** (fase 23).

### Explicabilidad (SHAP, top 5)

1. `gold_spot_lag21` (precio del oro hace 1 mes) - domina
2. `gold_ret_roll21` (retorno mensual)
3. `silver_spot` (plata - hermana del oro)
4. `commodities_crb` (materias primas)
5. `quarter` y `platinum_spot` (calendario y metal relacionado)

---

## Decisiones técnicas clave

| Decisión | Justificación |
|---|---|
| **Split temporal estricto** (train 2000-2019 / val 2020-2022 / test 2023-2025) | Forecasting: nunca datos futuros en train |
| **Modelo final: Ridge** (lineal regularizado) | MAE 60.4 (val) frente a 212.9-282.5 de los árboles: los árboles no extrapolan bien niveles no estacionarios |
| **Features**: lags + retornos + rolling del target, exógenas con lag 1 + retorno, calendario, indicadores de ausencia | Sin leakage (todo hacia atrás) |
| **Forward-fill** (solo pasado) para exógenas con huecos | Datos macro publicados con retraso |
| **Ventana 2000-2025** (6,705 días hábiles; 27 columnas limpias) | Cobertura fiable; antes de 2000 no hay datos |
| **83 features** seleccionadas tras el filtro de redundancia (|rho|>0.98) | Reduce colinealidad |

---

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Python 3.11+  |  numpy, pandas, scikit-learn, scipy, xgboost, lightgbm, catboost,
torch, statsmodels, optuna, shap, matplotlib, seaborn, fastapi, uvicorn, pytest, jupyterlab.

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
#    GET  /health            (liveness)
#    GET  /ready             (artefactos disponibles)
#    POST /predict           {"date": "2025-08-14", "features": {...}}
#    POST /predict_direction  {"date": "2025-08-14", "features": {...}}  -> P(sube)
#    La API exige el vector completo de features ya calculadas con el mismo
#    esquema que el entrenamiento; no ingiere datos crudos.

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
| `16c_deep_learning_clasificacion.ipynb` | **16c. Deep learning: MLP y GRU con CUDA/CPU** |
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
pytest tests/ -q        # tests de split, métricas, features, limpieza, API, DL y notebooks
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
`docker-compose.yml`  |  `.dockerignore`  |  `.github/workflows/ci.yml`  |  `data/README.md`  |  `models/README.md`  |  `docs/audit_report.md`

## Estructura

```
|---- configs/            # config.yaml (rutas/split/features) + params.yaml (hiperparámetros)
|---- data/raw|interim|processed/
|---- notebooks/          # notebooks ejecutados (el corazón del proyecto)
|---- src/                # data/, features/, models/, evaluation/, api/
|---- models/             # artefactos regenerables y README de procedencia
|---- reports/            # métricas (JSON/CSV) y figures/ (EDA, SHAP, errores)
|---- scripts/            # predict.py (CLI), monitor_drift.py, run_notebooks.sh
|---- tests/              # pytest
+---- docs/               # metodología, informe, cards, API, deep learning y auditoría
```

## Limitaciones

1. **No sirve para trading automático**: la dirección diaria es débil e inestable
   (AUC calibrado 0.532; DA 46.1%). Uso analítico/de referencia.
2. **Drift de mercado**: el error crece con el tiempo (2025: MAE 193).
   Requiere reentrenamiento y monitorización continua.
3. **Incertidumbre mal calibrada**: los intervalos P5-P95 (bootstrap) cubren
   aproximadamente 5% de las observaciones en test; la volatilidad real excede
   la incertidumbre paramétrica del modelo.
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

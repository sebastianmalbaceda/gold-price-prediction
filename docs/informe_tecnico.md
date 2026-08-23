# Informe Técnico - Gold Price Prediction

## 1. Formulación

**Problema:** estimar el precio spot del oro (*gold_spot*, USD/oz) en el día hábil `t+h`.

**Tipo:** regresión / forecasting de series temporales con covariables exógenas.

**Horizontes:** h en {1, 5, 21} días hábiles. **Unidad:** día hábil (COMEX/NYMEX).

**Entrada en t:** precio del oro y las exógenas disponibles; el fichero crudo contiene 59 y el conjunto limpio conserva 25 antes de la ingeniería (nunca se usan valores futuros).

**Decisión que apoya:** referencia de nivel para sizing/stress testing; NO trading automático.

## 2. Datos

- **Fuente:** `data/raw/gold-price-prediction-dataset.csv` (45,368 filas, 1901-06-30 -> 2025-09-14, 61 columnas).
- **Licencia:** compilación pública de uso académico; sin datos personales.
- **Auditoría (fases 2-3):**
  - Cobertura desigual: la mayoría de series empiezan a mitad del s. XX; `gold_spot` desde 1979-12-27.
  - 12,963 filas de fin de semana (28.6%) sin cotización -> eliminadas.
  - 40+ features con cobertura baja en 2000+ (CPI, PIB, M2, google_trends...) -> excluidas en `config.yaml`.
  - Huecos de ~31 días en macro (CPI, paro) -> forward-fill (solo pasado).
- **Ventana de trabajo:** 2000-01-01 -> 2025-09-12, 6,705 días hábiles, 27 columnas tras limpieza.

## 3. EDA (fase 4)

- `gold_spot`: tendencia alcista estructural, cambios de régimen (2008, 2011, 2020, 2024).
- **No estacionaria en niveles** (ADF p>0.05); retornos log estacionarios.
- Autocorrelación: altísima en niveles (rho(lag1)~0.999), casi nula en retornos -> predecir niveles es "fácil" (persistencia), predecir direcciones es difícil.
- Volatilidad agrupada (clustering) típica de activos financieros.
- Correlaciones esperadas: negativo con DXY, positivo con plata/metales.

## 4. Split (fase 7)

| Conjunto | Periodo | Filas | Uso |
|---|---|---|---|
| Train | 2000-01 -> 2019-12 | 4,957 | aprender todo |
| Validation | 2020-01 -> 2022-12 | 783 | selección/tuning |
| Test | 2023-01 -> 2025-09 | 684 | **bloqueado**, 1 sola vez |

- Warm-up de 260+ días eliminado antes de particionar (lags/rolling completos).
- CV: `TimeSeriesSplit(5 folds, gap=21 días)` sobre train+val.
- Índices no re-generados; test jamás tocado durante tuning.

## 5. Preprocessing y features (fases 8-10)

- Limpieza: días hábiles, ffill exógenas (solo pasado), exclusión por cobertura.
- Escalado: **RobustScaler (5-95%) ajustado SOLO con train**, aplicado con `transform` a val/test.
- Features (128 columnas derivadas -> **83 finales** tras filtro de redundancia |rho|>0.98 con prioridad a niveles):
  - Lags del target: 1, 2, 3, 5, 10, 21.
  - Retornos log del target a esos lags.
  - Rolling: retorno, media, desv. a 5/21/63/126 días.
  - Exógenas: lag 1 + retorno log; indicadores de ausencia original.
  - Calendario: año, mes, día de semana, trimestre, día del año.
- Anti-leakage: todas las ventanas miran solo hacia atrás; targets desplazados a futuro; sin agregados globales.

## 6. Baselines (fase 11)

| Baseline | CV MAE | sMAPE |
|---|---|---|
| Naive (último valor conocido) | 287.2 | 25.7% |
| Ridge | 122.9 | 13.2% |
| RandomForest | 222.8 | 22.2% |
| XGBoost | 202.5 | 20.6% |

Nota: el naive de CV usa el último valor del fold (1-2 años antes) -> MAE alto.
En test, el naive usa el último valor de train+val (2022) -> MAE 879.

## 7. Modelado y tuning (fases 12-14)

- Familias: Ridge, RandomForest, XGBoost, LightGBM, CatBoost.
- **Optuna (TPE, 77 trials totales)** con TimeSeriesSplit, optimizando MAE medio.
- Resultados CV: **Ridge MAE 70.0** (alpha~0.002) vs RF 212.1 / XGB 196.8 / LGBM 191.3 / Cat 220.6.
- Los árboles no extrapolan niveles no estacionarios (R^2 negativo en validation); el modelo lineal captura tendencia + lags.

## 8. Selección y entrenamiento final (fases 15-16)

- **Validation (2020-2022):** Ridge MAE=60.4, RMSE=72.5, sMAPE=3.44%, R^2=0.493, DA=47.4%.
  RF/XGB/LGBM/Cat: MAE 212.9-282.5 y R^2 negativo -> descartados.
- **Decisión:** Ridge con alpha=0.0022, 83 features, RobustScaler.
- Reentrenado con train+val (5,740 filas). Artefactos: `models/final_model.joblib`, `preprocessor.joblib`, `feature_list.json`.

## 9. Evaluación final en test (fase 17)

| Métrica | Valor |
|---|---|
| MAE | **99.75 USD/oz** |
| RMSE | 125.08 USD/oz |
| sMAPE | **3.94%** |
| R^2 | **0.9352** |
| Directional Accuracy | 46.1% |
| vs Naive congelado | **-88.7% MAE** |

## 9b. Verificación fuerte de generalización (fase 16b)

**¿Overfitting o underfitting?** Diagnóstico con 4 pruebas:

1. **Gap train/val/test**: MAE train=15.2, val=24.9, test=99.7. El salto temporal
se explica por drift de régimen y no implica por sí solo sobreajuste.
2. **Learning curve (validación fija)**: la validación baja de 256.6 a 60.4 al
añadir historia, aunque permanece la variabilidad temporal.
3. **Persistencia diaria**: naive MAE=17.6 frente a Ridge MAE=99.7 en test ->
**el modelo NO supera a "mañana = hoy" en h=1**; su valor es descriptivo para
niveles y tendencia.

**Conclusión técnica**: el R^2 alto (0.998 train / 0.935 test) mide el
seguimiento del nivel y parte de la tendencia, no una capacidad fiable de
timing diario. El cambio a 1 día sigue siendo difícil de predecir. El modelo de regresión es útil
como **referencia de nivel/tendencia**, no para timing.

## 9c. Clasificador de dirección (sube/baja) - fase 16b

Modelo binario: `y_dir = 1 si gold(t+h) > gold(t)`, con las mismas features
y split. La tabla siguiente muestra el RF no calibrado como diagnóstico; el
artefacto desplegado se calibra con isotonic y h=1.

| Horizonte | CV AUC | Test AUC | Test ACC | P(sube) test |
|---|---|---|---|---|
| h=1 | 0.529 | **0.573** | 0.558 | 0.537 |
| h=5 | 0.528 | 0.534 | 0.570 | 0.573 |
| h=21 | 0.584 | 0.509 | 0.484 | 0.677 |

- Diagnóstico no calibrado h=1: AUC=0.573, ACC=0.558. El artefacto desplegado calibrado obtiene **AUC=0.532, ACC=0.539, recall=0.984, PR-AUC=0.576, Brier=0.247**.
- **Trazabilidad de métricas**: estas cifras del artefacto calibrado provienen de `models/direction_metrics.json`, que no está versionado; no son auditables sin reejecutar el pipeline.
- Señal débil e inestable: el walk-forward medio es AUC=0.526 +/- 0.043.
- Features más informativas: momentum (`gold_ret_lag1`, `gold_ret_roll63`)
  y riesgo (`geopolitical_risk`, `policy_uncertainty`, `usdinr_exchange_ret_lag1`).
- **Interpretación**: inclinación leve (para alertas/sizing marginal), NO
  señal de trading automático. La probabilidad está calibrada (Brier 0.247
  cerca del óptimo para P~0.54).
- Artefactos: `models/direction_classifier.joblib`, endpoint `/predict_direction`.

## 9c-dl. Extension de deep learning (fase 16c)

Se compararon dos arquitecturas neuronales sobre el mismo split temporal y
las mismas 83 features: una MLP tabular y una GRU causal de 21 dias.
El entrenamiento utiliza AdamW, dropout, weight decay, clipping de gradiente
y early stopping basado exclusivamente en AUC de validation. El dispositivo
se detecta automaticamente; la ejecucion registrada uso CUDA en una NVIDIA
GeForce RTX 3050 Ti Laptop GPU.

| Modelo | Train AUC | Validation AUC | Test AUC |
|---|---:|---:|---:|
| MLP | 0.649 | 0.550 | 0.534 |
| GRU | 0.569 | 0.543 | 0.510 |

La MLP es la mejor por validation, pero no supera al RandomForest regularizado
(AUC test 0.560) ni al clasificador calibrado de referencia (AUC 0.532). Por tanto, deep learning se incorpora como experimento
reproducible y comparativo, no como sustituto del modelo operativo. Los
resultados no justifican aumentar la complejidad: con 5.740 observaciones y
señal financiera debil, la red aprende patrones limitados y conserva un gap
train-test apreciable.

El notebook guarda el dispositivo, la semilla, el mejor epoch, los pesos y
los metadatos. Los pesos `.pt` no se versionan por git; pueden regenerarse
ejecutando el notebook con las dependencias de `requirements.txt`.

## 9d. ¿Es rentable? Backtest y significancia (fase 17b)

**Acierto por clase (test, umbral 0.5):** TPR = 98.4% y TNR = 2.5%. El modelo
calibrado predice sube el 98% de las veces; el umbral 0.55 reduce operaciones,
pero la estrategia solo obtiene +2.4% en este test.

**Backtest (retorno hoy->mañana, costes 0.1%/operación):**

| Estrategia | Retorno | Sharpe | vs Buy&Hold |
|---|---|---|---|
| Buy & hold | +82.9% | - | - |
| LONG filtrado (clasificador calibrado, thr 0.5) | +85.2% | 1.60 | +2.3 pp |
| LONG filtrado (clasificador calibrado, thr 0.55) | +2.4% | 0.31 | -80.5 pp |
| LONG filtrado (RF profundo, thr 0.53) | +28.0% | 1.24 | -54.9 pp |
| LONG-SHORT (thr 0.5) | -90.8%* | -5.78 | -174 pp |

*El LONG-SHORT con el clasificador calibrado usaba el retorno desfasado
(ayer->hoy); con el retorno correcto no se reproduce en el notebook 17b.

**Significancia:**
- AUC test (RF profundo) = 0.533; IC95% bootstrap = [0.490, 0.574] (incluye 0.5).
- CV temporal: AUC medio 0.526 +/- 0.043 -> **señal inestable fuera de test**.
- t-test estrategia vs buy&hold: p = 0.079 -> **no significativa al 5%**.

**Conclusión final**: la dirección del oro a 1 día tiene señal **débil y no
explotable** de forma fiable: el AUC calibrado es 0.532 y su intervalo no
confirma ventaja sobre el azar. En backtest no supera de forma robusta a
comprar y mantener. La rentabilidad a largo plazo de
una estrategia de tendencia con estos datos no está demostrada; el valor
real está en la gestión de riesgo (alertas, sizing), no en el timing.

## 9e. Verificación de robustez (fase 17c)

**¿Overfitting?** El modelo original (RF depth=8) sobreajustaba: train AUC
0.907 vs test 0.574 (gap 0.334). El RF profundo alcanza gap 0.467. Con
regularización fuerte (depth=4, leaf=20) el gap baja a **0.124** y el test AUC
es 0.560. La learning curve reduce el gap de 0.301 a 0.187.

**¿Underfitting?** No se observa una red demasiado simple; la señal sigue siendo
débil y dependiente del régimen.

**Walk-forward 2019-2025** (reentrenamiento anual, ventana 5 años):

| Año eval | AUC | Estrategia | Buy&Hold |
|---|---|---|---|
| 2019 | 0.527 | -4.0% | +18.3% |
| 2020 | 0.526 | +13.0% | +25.1% |
| 2021 | 0.482 | -10.6% | -5.1% |
| 2022 | 0.488 | -7.2% | +1.3% |
| 2023 | 0.514 | +3.6% | +13.2% |
| 2024 | 0.533 | +8.2% | +27.1% |
| 2025 | 0.613 | +32.8% | +27.1% |

- AUC medio **0.526 +/- 0.043**; solo 2025 supera 0.55.
- La estrategia gana a buy&hold solo en 2025 (mercado con tendencia fuerte).
- Por régimen: alcista AUC=0.538, bajista 0.448, lateral 0.474 -> la señal
de momentum es condicional al régimen.

**Conclusión de robustez**: el modelo regularizado es **lo mejor posible con
estos datos** (sin memorizar, sin ser trivial): señal real pero débil
(AUC walk-forward ~0.526). Su uso práctico honesto es como **indicador de riesgo**
(reducir exposición cuando P<0.5, alertas), no como fuente de rentabilidad
superior a comprar y mantener.

## 9f. Predicción de volatilidad y gestión de riesgo (fase 17d)

La volatilidad es un proceso con **autocorrelación muy alta** (lag-1 = 0.985,
*volatility clustering*), lo que la hace más predecible que la dirección.

**Modelo de volatilidad realizada a 5 días (RF, mismas features):**

| Modelo | MAE (vol anualizada) | R^2 |
|---|---|---|
| Naive (vol actual) | 0.0553 | -0.212 |
| **RF volatilidad** | **0.0463** | **+0.241** |

El modelo supera al naive (+16.2% MAE) y captura los picos de 2024-25.

**Position sizing por volatilidad** (exposición = min(1, vol_objetivo/vol_prevista),
con vol_objetivo = 12% anualizado):

| Estrategia | Retorno | Sharpe | Max DD | Exposición |
|---|---|---|---|---|
| Buy & hold | +86.4% | 1.61 | -11.3% | 100% |
| **Sizing vol** | +73.2% | **1.83** | **-10.9%** | 81.6% |

La gestión de riesgo basada en volatilidad **mejora el Sharpe (1.83 vs 1.61)
y reduce el drawdown** con menos exposición: es la extensión con mayor valor
práctico del proyecto, coherente con la literatura cuantitativa
(volatility targeting).

## 9g. Auditoría de datos temporales (fase 18b)

**Frecuencias de actualización** (días entre cambios de valor, ventana 2000+):

| Frecuencia | Variables |
|---|---|
| Diaria (1 día) | 26 (tipos, FX, materias primas, índices) |
| Semanal (5 días) | 1 (us_financial_stress_index) |
| Mensual (23-31 días) | 13 (CPI, paro, M2, retail, sentimiento, fed_funds...) |
| Trimestral (92 días) | PIB (us_gdp) |

**Lookahead bias**: las macro se actualizan el **día 1 del mes** en el dataset
(P10=P90=1), pero su publicación real es 5-30 días después (BLS: CPI día 10-15;
BEA: PIB ~30 días). El ffill rellena huecos pero no corrige el adelanto.
Impacto medido: AUC +0.000 en el conjunto actual porque las macro de baja cobertura
se excluyen antes del modelado; el riesgo de publicación anticipada queda documentado.

**Multicolinealidad (VIF)**: 37/83 features con VIF>10 (commodities_bloomberg 206.7,
commodities_crb 177.8, gold_spot_lag21 176.9). Es esperable por los lags del
mismo activo; Ridge (L2) y los árboles la toleran.

**Rango de fechas**: ventana 2000-2025 con cobertura 100% tras warm-up;
antes de 2000 las series no existen. El rango es óptimo y está justificado.

## 9h. Comparación con la versión previa (referencia TFG)

Se evaluaron experimentalmente las técnicas distintivas de la versión
anterior (rama `main`, trabajo de fin de grado) para determinar si aportaban
valor sobre el pipeline actual:

| Técnica de la versión previa | Resultado en v2 | Decisión |
|---|---|---|
| Voting/Ensemble (RF+XGB+LR) | Experimento previo no comparable tras la auditoria de cobertura | **No se incorpora**: la señal es demasiado débil para justificar complejidad |
| Mutual Information (selección) | Top features MI coinciden con importancia RF (retornos FX y momentum) | **Sin features ocultas**: el pipeline actual ya las captura |
| Features técnicas (RSI, MA50/200, vol21) | AUC 0.557 vs 0.562 (actuales); importancia baja (rank 17-53) | **No mejoran**: el momentum ya está cubierto por retornos/rolling |
| Interpolación lineal para imputar | Riesgo de leakage (usa valores futuros) | **Rechazada**: v2 usa solo ffill (correcto) |
| LONG-SHORT sin costes (+99.3% reportado) | Con costes realistas y walk-forward la señal colapsa a AUC ~0.54 | **No replicable**: resultado optimista por ausencia de costes |
| Spearman (correlación no paramétrica) | Complementa a Pearson en el EDA | **Incorporada** (notebook 04) |
| PACF (autocorrelación parcial) | Complementa a ACF | **Incorporada** (notebook 04) |

**Conclusión**: la versión actual (v2) ya incorpora el rigor metodológico
(walk-forward, costes de transacción, significancia estadística, detección de
leakage) que la versión previa no tenía. Las técnicas evaluadas de la versión
anterior no mejoran el rendimiento de forma significativa; se incorporaron
solo las de valor estadístico complementario (Spearman, PACF).

## 10. Error analysis y explicabilidad (fase 18)

- **Errores crecientes en el tiempo:** 2023: MAE 41.4 -> 2024: 99.7 -> 2025: 193.4.
- Peores errores: abril-mayo 2025 (oro 2,750->3,430; el modelo predecía ~2,800) - rallies extremos no capturados.
- SHAP (Ridge): `gold_spot_lag21` domina (338.4), luego `gold_ret_roll21` (26.7), `silver_spot` (14.1), `commodities_crb` (3.3) y `quarter` (3.0).
- Incertidumbre: bootstrap (100 fits) -> intervalo P5-P95 de 26.4 USD de ancho y cobertura ~5% en test -> **mal calibrada**; la volatilidad real excede la paramétrica. Para producción: usar cuantiles/volatilidad GARCH o conformal prediction.

## 11. Robustez (fase 19)

- KS-test train+val vs test: **drift significativo en todas las features clave** (DXY, tipos, plata, USD/JPY... p<0.01).
- Degradación temporal documentada (sección 10).
- Pruebas de entrada: la API valida esquema/tipos/NaN, rangos, fechas y readiness (422/503); 37 tests y validación estructural de notebooks.

## 12. Ética, privacidad y seguridad (fase 20)

- Sin datos personales; series de mercado públicas.
- Uso previsto: analítico; **no** asesoramiento financiero automatizado ni trading sin supervisión.
- Sin secretos en repo; dependencias declaradas con cotas mínimas; la reproducción exacta requiere un lockfile; validación de inputs en API.
- Riesgo de uso indebido: predicciones como "certezas" - se documenta la incertidumbre real.

## 13. Despliegue (fase 21)

- API REST FastAPI (`src/api/main.py`): `GET /health`, `POST /predict` con esquema Pydantic.
- Usa los artefactos entrenados (mismo preprocesador - cero divergencia train/serve).
- CLI: `scripts/predict.py`. Monitorización: `scripts/monitor_drift.py`.

## 14. Riesgos y decisión final

- **Riesgo principal:** drift no estacionario -> reentrenar con cadencia (p. ej. mensual/trimestral) + alerta si MAE rodante > 1.5x MAE test.
- **Decisión:** Ridge es el modelo final por calidad/estabilidad/coste/interpretabilidad (SHAP lineal).
- Alternativas futuras: modelos sobre retornos + reintegración de niveles (evita el problema de extrapolación), LSTM/Transformer con ventana fija, conformal prediction para intervalos calibrados, features de volatilidad (GVZ) y posicionamiento (CFTC).

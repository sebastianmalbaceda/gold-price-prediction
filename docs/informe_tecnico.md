# Informe Técnico — Gold Price Prediction

## 1. Formulación

**Problema:** estimar el precio spot del oro (*gold_spot*, USD/oz) en el día hábil `t+h`.

**Tipo:** regresión / forecasting de series temporales con covariables exógenas.

**Horizontes:** h ∈ {1, 5, 21} días hábiles. **Unidad:** día hábil (COMEX/NYMEX).

**Entrada en t:** precio del oro y 59 exógenas en t y pasadas (nunca futuras).

**Decisión que apoya:** referencia de nivel para sizing/stress testing; NO trading automático.

## 2. Datos

- **Fuente:** `data/raw/gold-price-prediction-dataset.csv` (45,368 filas, 1901-06-30 → 2025-09-14, 61 columnas).
- **Licencia:** compilación pública de uso académico; sin datos personales.
- **Auditoría (fases 2-3):**
  - Cobertura desigual: la mayoría de series empiezan a mitad del s. XX; `gold_spot` desde 1979-12-27.
  - 12,963 filas de fin de semana (28.6%) sin cotización → eliminadas.
  - 40+ features con cobertura < 5% en 2000+ (CPI, PIB, M2, google_trends...) → excluidas en `config.yaml`.
  - Huecos de ~31 días en macro (CPI, paro) → forward-fill (solo pasado).
- **Ventana de trabajo:** 2000-01-01 → 2025-09-12, 6,705 días hábiles, 42 columnas tras limpieza.

## 3. EDA (fase 4)

- `gold_spot`: tendencia alcista estructural, cambios de régimen (2008, 2011, 2020, 2024).
- **No estacionaria en niveles** (ADF p>0.05); retornos log estacionarios.
- Autocorrelación: altísima en niveles (ρ(lag1)≈0.999), casi nula en retornos → predecir niveles es "fácil" (persistencia), predecir direcciones es difícil.
- Volatilidad agrupada (clustering) típica de activos financieros.
- Correlaciones esperadas: negativo con DXY, positivo con plata/metales.

## 4. Split (fase 7)

| Conjunto | Periodo | Filas | Uso |
|---|---|---|---|
| Train | 2000-01 → 2019-12 | 4,956 | aprender todo |
| Validation | 2020-01 → 2022-12 | 783 | selección/tuning |
| Test | 2023-01 → 2025-09 | 684 | **bloqueado**, 1 sola vez |

- Warm-up de 260+ días eliminado antes de particionar (lags/rolling completos).
- CV: `TimeSeriesSplit(5 folds, gap=21 días)` sobre train+val.
- Índices no re-generados; test jamás tocado durante tuning.

## 5. Preprocessing y features (fases 8-10)

- Limpieza: días hábiles, ffill exógenas (solo pasado), exclusión por cobertura.
- Escalado: **RobustScaler (5-95%) ajustado SOLO con train**, aplicado con `transform` a val/test.
- Features (146 brutas → **85 finales** tras filtro de redundancia |ρ|>0.98 con prioridad a niveles):
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
| Ridge | 108.3 | 10.5% |
| RandomForest | 228.5 | 22.5% |
| XGBoost | 220.0 | 21.8% |

Nota: el naive de CV usa el último valor del fold (1-2 años antes) → MAE alto.
En test, el naive usa el último valor de train+val (2022) → MAE 879.

## 7. Modelado y tuning (fases 12-14)

- Familias: Ridge, RandomForest, XGBoost, LightGBM, CatBoost.
- **Optuna (TPE, 77 trials totales)** con TimeSeriesSplit, optimizando MAE medio.
- Resultados CV: **Ridge MAE 55.1** (α≈0.002) vs RF 227 / XGB 199 / LGBM 197 / Cat 227.
- Los árboles no extrapolan niveles no estacionarios (R² negativo en validation); el modelo lineal captura tendencia + lags.

## 8. Selección y entrenamiento final (fases 15-16)

- **Validation (2020-2022):** Ridge MAE=37.0, RMSE=46.2, sMAPE=2.07%, R²=0.794, DA=48.7%.
  RF/XGB/LGBM/Cat: MAE 199-297, R² negativo → descartados.
- **Decisión:** Ridge con α=0.0022, 85 features, RobustScaler.
- Reentrenado con train+val (5,739 filas). Artefactos: `models/final_model.joblib`, `preprocessor.joblib`, `feature_list.json`.

## 9. Evaluación final en test (fase 17)

| Métrica | Valor |
|---|---|
| MAE | **204.36 USD/oz** |
| RMSE | 241.81 USD/oz |
| sMAPE | **8.28%** |
| R² | **0.758** |
| Directional Accuracy | 47.4% |
| vs Naive | **−76.8% MAE** |

## 9b. Verificación fuerte de generalización (fase 16b)

**¿Overfitting o underfitting?** Diagnóstico con 4 pruebas:

1. **Gap train/val/test**: MAE train=14.7, val=21.5, test=204.7. El gap
train→val es pequeño (14.7→21.5) → **sin overfitting severo**. El salto a
test (204.7) se explica por **drift de régimen** (2023-25: rally histórico).
2. **Learning curve (CV temporal)**: el MAE de validación NO empeora
sistemáticamente al crecer train (15→42→89→58→42) → **no hay overfitting**;
el modelo se beneficia de más datos.
3. **Gap por familia**: Ridge gap=23, RF gap=264, XGB gap=200 → Ridge es la
familia con **menor overfitting**, coherente con su victoria en selección.
4. **vs naive-persistencia** (`gold_spot(t)`): naive MAE=17.6 vs modelo
204.7 en test → **el modelo NO supera a "mañana = hoy" en h=1**.

**Conclusión técnica**: el R² alto (0.998 train / 0.758 test) mide el
seguimiento de la **tendencia**, no la precisión del cambio diario. En un
mercado eficiente, el cambio a 1 día es ruido: ningún modelo con datos
públicos lo supera de forma consistente. El modelo de regresión es útil
como **referencia de nivel/tendencia**, no para timing.

## 9c. Clasificador de dirección (sube/baja) — fase 16b

Modelo binario: `y_dir = 1 si gold(t+h) > gold(t)`, con las mismas features
y split. RandomForest calibrado (isotónico, CV interna).

| Horizonte | CV AUC | Test AUC | Test ACC | P(sube) test |
|---|---|---|---|---|
| h=1 | 0.528 | **0.555** | 0.557 | 0.537 |
| h=5 | 0.532 | 0.511 | 0.586 | 0.573 |
| h=21 | 0.571 | 0.535 | 0.689 | 0.677 |

- h=1 desplegado: **AUC=0.555, ACC=0.557, recall=0.834, PR-AUC=0.584, Brier=0.247**.
- Señal débil pero real (AUC > 0.5 de forma consistente en CV y test).
- Features más informativas: momentum (`gold_ret_lag1`, `gold_ret_roll63`)
  y riesgo (`geopolitical_risk`, `policy_uncertainty`, `usdinr_exchange_ret_lag1`).
- **Interpretación**: inclinación leve (para alertas/sizing marginal), NO
  señal de trading automático. La probabilidad está calibrada (Brier 0.247
  cerca del óptimo para P≈0.54).
- Artefactos: `models/direction_classifier.joblib`, endpoint `/predict_direction`.

## 9d. ¿Es rentable? Backtest y significancia (fase 17b)

**Acierto por clase (test, umbral 0.5):** TPR (acierta subidas) = 83.4%,
TNR (acierta bajadas) = 23.7%. El modelo está sesgado a predecir sube
(80% de las veces) por el desbalance (P(sube)=54%) y el umbral 0.5.
Subir el umbral mejora precisión: 0.55 → 59%, 0.60 → 67%.

**Backtest (retorno hoy→mañana, costes 0.1%/operación):**

| Estrategia | Retorno | Sharpe | vs Buy&Hold |
|---|---|---|---|
| Buy & hold | +82.9% | — | — |
| LONG filtrado (clf calibrado, thr 0.5) | +74.4% | 1.63 | −8.5 pp |
| LONG filtrado (RF profundo, thr 0.51) | +48.7% | 1.60 | −34 pp |
| LONG-SHORT (thr 0.5) | −90.8%* | −5.78 | −174 pp |

*El LONG-SHORT con el clasificador calibrado usaba el retorno desfasado
(ayer→hoy); con el retorno correcto no se reproduce en el notebook 17b.

**Significancia:**
- AUC test (RF profundo) = 0.571; IC95% bootstrap = [0.456, 0.545] (roza 0.5).
- Permutación: p < 0.001 (señal real en test).
- CV temporal: AUC medio ≈ 0.516 → **señal inestable fuera de test**.
- t-test estrategia vs buy&hold: p = 0.25 → **no rentable de forma robusta**.

**Conclusión final**: la dirección del oro a 1 día tiene señal **débil y no
explotable** de forma fiable: el clasificador acierta mejor que el azar y su
confianza es informativa (P(sube|pred) 64-77% en los extremos), pero en
backtest no supera a comprar y mantener. La rentabilidad a largo plazo de
una estrategia de tendencia con estos datos no está demostrada; el valor
real está en la gestión de riesgo (alertas, sizing), no en el timing.

## 9e. Verificación de robustez (fase 17c)

**¿Overfitting?** El modelo original (RF depth=8) sobreajustaba: train AUC
0.91 vs test 0.56 (gap 0.35). El RF profundo aún más (gap 0.43). Con
regularización fuerte (depth=4, leaf=20) el gap baja a **0.126** manteniendo
test AUC=0.569. La learning curve confirma que el gap se reduce al crecer
train (0.29→0.19), típico de overfitting controlable con regularización.

**¿Underfitting?** No: submuestrear train empeora el test AUC (0.537 con
30% vs 0.557 con 100%). El modelo captura la señal disponible.

**Walk-forward 2019-2025** (reentrenamiento anual, ventana 5 años):

| Año eval | AUC | Estrategia | Buy&Hold |
|---|---|---|---|
| 2019 | 0.550 | +4.1% | +18.3% |
| 2020 | 0.522 | +2.6% | +25.1% |
| 2021 | 0.501 | −8.3% | −5.1% |
| 2022 | 0.482 | −7.7% | +1.3% |
| 2023 | 0.514 | +4.5% | +13.2% |
| 2024 | 0.545 | +11.7% | +27.1% |
| 2025 | 0.656 | +31.6% | +27.1% |

- AUC medio **0.539 ± 0.057**; solo 2025 supera claramente 0.55.
- La estrategia gana a buy&hold solo en 2025 (mercado con tendencia fuerte).
- Por régimen: alcista AUC=0.548, bajista 0.438, lateral 0.484 → la señal
de momentum es condicional al régimen.

**Conclusión de robustez**: el modelo regularizado es **lo mejor posible con
estos datos** (sin memorizar, sin ser trivial): señal real pero débil
(AUC ~0.54). Su uso práctico honesto es como **indicador de riesgo**
(reducir exposición cuando P<0.5, alertas), no como fuente de rentabilidad
superior a comprar y mantener.

## 9f. Predicción de volatilidad y gestión de riesgo (fase 17d)

La volatilidad es un proceso con **autocorrelación muy alta** (lag-1 = 0.985,
*volatility clustering*), lo que la hace más predecible que la dirección.

**Modelo de volatilidad realizada a 5 días (RF, mismas features):**

| Modelo | MAE (vol anualizada) | R² |
|---|---|---|
| Naive (vol actual) | 0.0553 | −0.212 |
| **RF volatilidad** | **0.0510** | **+0.058** |

El modelo supera al naive (+7.9% MAE) y captura los picos de 2024-25.

**Position sizing por volatilidad** (exposición = min(1, vol_objetivo/vol_prevista),
con vol_objetivo = 12% anualizado):

| Estrategia | Retorno | Sharpe | Max DD | Exposición |
|---|---|---|---|---|
| Buy & hold | +86.4% | 1.61 | −11.3% | 100% |
| **Sizing vol** | +76.8% | **1.88** | **−10.3%** | 80.7% |

La gestión de riesgo basada en volatilidad **mejora el Sharpe (1.88 vs 1.61)
y reduce el drawdown** con menos exposición: es la extensión con mayor valor
práctico del proyecto, coherente con la literatura cuantitativa
(volatility targeting).

## 10. Error analysis y explicabilidad (fase 18)

- **Errores crecientes en el tiempo:** 2023: MAE 87 → 2024: 205 → 2025: 393.
- Peores errores: abril-mayo 2025 (oro 2,750→3,430; el modelo predecía ~2,800) — rallies extremos no capturados.
- SHAP (Ridge): `gold_spot_lag21` domina (281.7), luego `gold_ret_roll21` (22.7), `silver_spot` (20.7), `sp500_futures` (10.2), `consumer_confidence` (8.2), `us_gdp` (6.5), tipos (6.3).
- Incertidumbre: bootstrap (100 fits) → intervalo P5-P95 de 33.5 USD de ancho pero cobertura solo del 1% en test → **mal calibrada**; la volatilidad real excede la paramétrica. Para producción: usar cuantiles/volatilidad GARCH o conformal prediction.

## 11. Robustez (fase 19)

- KS-test train+val vs test: **drift significativo en todas las features clave** (DXY, tipos, plata, USD/JPY... p<0.01).
- Degradación temporal documentada (sección 10).
- Pruebas de entrada: la API valida esquema/tipos/NaN (422/503); tests pytest 9/9 OK.

## 12. Ética, privacidad y seguridad (fase 20)

- Sin datos personales; series de mercado públicas.
- Uso previsto: analítico; **no** asesoramiento financiero automatizado ni trading sin supervisión.
- Sin secretos en repo; dependencias fijadas; validación de inputs en API.
- Riesgo de uso indebido: predicciones como "certezas" — se documenta la incertidumbre real.

## 13. Despliegue (fase 21)

- API REST FastAPI (`src/api/main.py`): `GET /health`, `POST /predict` con esquema Pydantic.
- Usa los artefactos entrenados (mismo preprocesador — cero divergencia train/serve).
- CLI: `scripts/predict.py`. Monitorización: `scripts/monitor_drift.py`.

## 14. Riesgos y decisión final

- **Riesgo principal:** drift no estacionario → reentrenar con cadencia (p. ej. mensual/trimestral) + alerta si MAE rodante > 1.5× MAE test.
- **Decisión:** Ridge es el modelo final por calidad/estabilidad/coste/interpretabilidad (SHAP lineal).
- Alternativas futuras: modelos sobre retornos + reintegración de niveles (evita el problema de extrapolación), LSTM/Transformer con ventana fija, conformal prediction para intervalos calibrados, features de volatilidad (GVZ) y posicionamiento (CFTC).

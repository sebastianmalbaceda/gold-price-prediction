# Model Card - Gold Price Prediction (Ridge + Clasificador de Dirección)

## Resumen
Dos modelos:
1. **Regresión (Ridge)** para el **nivel** del precio spot del oro (USD/oz) a 1 día hábil.
2. **Clasificación (RandomForest calibrado)** para la **dirección** (sube/baja) a 1 día hábil.

Entrenados con datos 2000-2022, evaluados en test 2023-2025.

## Uso previsto
- Referencia cuantitativa de nivel para analistas (sizing, stress testing, alertas).
- Dirección como **inclinación muy leve** (AUC calibrado 0.532; walk-forward 0.526): alertas tempranas, no trading automático.
- Investigación/educación en forecasting financiero.
- Predicción batch/diaria vía API o CLI.

## Uso NO previsto
- **Trading automático** o decisiones de compra/venta sin supervisión humana.
- Horizontes > 21 días.
- Predicción en regímenes sin precedentes (el modelo extrapola mal).

## Datos
- 45,368 filas crudas (1901-2025) -> ventana 2000-2025 -> 6,705 días hábiles.
- 59 variables exógenas en bruto; 25 sobreviven a la limpieza y 83 features derivadas se seleccionan para el modelo.
- Fuente: compilación pública; sin datos personales.

## Métricas - Regresión (test bloqueado 2023-2025)

| Métrica | Valor |
|---|---|
| MAE | 99.75 USD/oz |
| RMSE | 125.08 USD/oz |
| sMAPE | 3.94% |
| R^2 | 0.9352 |
| DA | 46.1% |
| vs naive congelado | -88.7% MAE |

**Limitación clave**: no supera al naive-persistencia diario (MAE 17.6); su
valor está en el seguimiento de tendencia, no en el cambio diario.

## Métricas - Clasificador de dirección (test bloqueado 2023-2025, h=1)

| Métrica | Valor |
|---|---|
| AUC-ROC | **0.532** |
| PR-AUC | 0.576 |
| Accuracy | 0.539 |
| Recall (sube) | 0.984 |
| Precision (sube) | 0.539 |
| F1 | 0.696 |
| Brier | 0.247 |

**Trazabilidad de métricas**: estas cifras del artefacto calibrado provienen de
`models/direction_metrics.json`, que no está versionado; no son auditables sin
reejecutar el pipeline.

Señal **débil e inestable**; el intervalo del diagnóstico incluye 0.5. No apta para trading automático.

## Análisis de rentabilidad (fase 17b)

**Acierto por clase (test, umbral 0.5):**
- Acierta cuando sube (TPR): **98.4%**  |  Acierta cuando baja (TNR): **2.5%**
- Predice sube el 98% de las veces; el umbral 0.55 reduce operaciones y logra 68.8% de precision condicional, pero solo +2.4% de retorno.

**Backtest (retorno hoy->mañana, costes 0.1%/op):**
- LONG filtrado (clasificador calibrado, thr 0.5): +85.2% (Sharpe 1.60) vs Buy&Hold +82.9%.
- RF profundo (thr 0.53): +28.0% (Sharpe 1.24).
- **Ninguna estrategia supera a comprar y mantener**.

**Significancia:** RF profundo AUC test 0.533 (IC95% [0.490, 0.574]);
CV AUC 0.526 +/- 0.043; t-test de retornos p=0.079 (no significativo al 5%).

**Conclusión**: señal débil y no explotable de forma fiable. El modelo sirve
como indicador de riesgo/inclinación, no como estrategia de trading.

## Robustez (fase 17c)

- **Overfitting**: el RF original (depth=8) sobreajustaba (train AUC 0.91 vs
test 0.574). Con regularización (depth=4, leaf=20) el gap baja a **0.124**
con test AUC 0.560.
- **Underfitting**: no (más datos mejoran; el modelo captura la señal).
- **Walk-forward 2019-2025**: AUC medio 0.526 +/- 0.043; la estrategia gana a
buy&hold solo en 1/7 años (2025).
- **Por régimen**: alcista AUC 0.538  |  bajista 0.448  |  lateral 0.474.
- **Conclusión**: con estos datos la señal es débil e inestable (AUC walk-forward ~0.526).
Uso práctico: indicador de riesgo (reducir exposición si P<0.5, alertas).

## Volatilidad y gestión de riesgo (fase 17d)

- Autocorrelación de la volatilidad: 0.985 (lag-1) -> mucho más predecible.
- RF volatilidad (h=5): R^2=+0.241 vs naive -0.212 (+16.2% MAE).
- Position sizing por volatilidad: Sharpe 1.83 vs 1.61 (buy&hold), maxDD
  -10.9% vs -11.3%, exposición 81.6%.
- **Conclusión**: la gestión de riesgo basada en volatilidad añade valor
  práctico real (volatility targeting), a diferencia de la señal de dirección.

## Subgrupos / segmentos
| Segmento | MAE regresión |
|---|---|
| 2023 | 41.4 |
| 2024 | 99.7 |
| 2025 | 193.4 |

Degradación creciente: los segmentos recientes (rally 2024-25) están fuera de
la distribución de entrenamiento (drift).

## Limitaciones
1. Cambio diario difícil de predecir: clasificador calibrado AUC 0.532 y walk-forward 0.526.
2. Drift de mercado: requiere reentrenamiento periódico.
3. Intervalos de incertidumbre mal calibrados (cobertura P5-P95 ~ 5%).
4. Features macro publicadas con retraso (ffill).

## Consideraciones éticas
- Sin datos personales ni variables sensibles.
- El modelo no debe presentarse como asesor financiero; el error medio de
  ~100 USD/oz (y hasta ~193 en 2025) debe comunicarse a los usuarios.

## Mantenimiento
- Reentrenar cuando el MAE rodante (60d) supere 1.5x el MAE de test o cuando
  el KS-drift sea persistente (script `scripts/monitor_drift.py`).
- Versionado de artefactos (`models/`), rollback conservando el modelo anterior.
- Propietario: equipo de datos (proyecto educativo).

## Versiones
- v1.1.0 (2025): Ridge alpha=0.0022 con 83 features, clasificador de dirección, MLP/GRU y endpoint `/predict_direction`.

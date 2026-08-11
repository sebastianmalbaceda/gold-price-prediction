# Model Card — Gold Price Prediction (Ridge + Clasificador de Dirección)

## Resumen
Dos modelos:
1. **Regresión (Ridge)** para el **nivel** del precio spot del oro (USD/oz) a 1 día hábil.
2. **Clasificación (RandomForest calibrado)** para la **dirección** (sube/baja) a 1 día hábil.

Entrenados con datos 2000-2022, evaluados en test 2023-2025.

## Uso previsto
- Referencia cuantitativa de nivel para analistas (sizing, stress testing, alertas).
- Dirección como **inclinación leve** (AUC ≈ 0.55): alertas tempranas, no trading automático.
- Investigación/educación en forecasting financiero.
- Predicción batch/diaria vía API o CLI.

## Uso NO previsto
- **Trading automático** o decisiones de compra/venta sin supervisión humana.
- Horizontes > 21 días.
- Predicción en regímenes sin precedentes (el modelo extrapola mal).

## Datos
- 45,368 filas crudas (1901-2025) → ventana 2000-2025 → 6,705 días hábiles.
- 60 features exógenas (tipos, FX, materias primas, índices, macro).
- Fuente: compilación pública; sin datos personales.

## Métricas — Regresión (test bloqueado 2023-2025)

| Métrica | Valor |
|---|---|
| MAE | 204.36 USD/oz |
| RMSE | 241.81 USD/oz |
| sMAPE | 8.28% |
| R² | 0.758 |
| DA | 47.4% |
| vs naive | −76.8% MAE |

**Limitación clave**: no supera al naive-persistencia diario (MAE 17.6); su
valor está en el seguimiento de tendencia, no en el cambio diario.

## Métricas — Clasificador de dirección (test bloqueado 2023-2025, h=1)

| Métrica | Valor |
|---|---|
| AUC-ROC | **0.555** |
| PR-AUC | 0.584 |
| Accuracy | 0.557 |
| Recall (sube) | 0.834 |
| Precision (sube) | 0.558 |
| F1 | 0.669 |
| Brier | 0.247 |

Señal **débil pero real** (AUC > 0.5). No apta para trading automático.

## Análisis de rentabilidad (fase 17b)

**Acierto por clase (test, umbral 0.5):**
- Acierta cuando sube (TPR): **83.4%** · Acierta cuando baja (TNR): **23.7%**
- Sesgo a predecir sube (80% de las veces) por desbalance + umbral 0.5.
- Con umbral 0.55-0.60 la precisión sube a 59-67% (operando menos).

**Backtest (retorno hoy→mañana, costes 0.1%/op):**
- LONG filtrado (clf, thr 0.5): +74.4% (Sharpe 1.63) vs Buy&Hold +82.9%.
- RF profundo (thr 0.51): +48.7% (Sharpe 1.60).
- **Ninguna estrategia supera a comprar y mantener**.

**Significancia:** AUC test 0.571 (IC95% [0.456, 0.545]); permutación p<0.001;
CV AUC ≈ 0.516 (inestable); t-test retornos p=0.25 (no significativo).

**Conclusión**: señal débil y no explotable de forma fiable. El modelo sirve
como indicador de riesgo/inclinación, no como estrategia de trading.

## Subgrupos / segmentos
| Segmento | MAE regresión |
|---|---|
| 2023 | 86.6 |
| 2024 | 204.8 |
| 2025 | 392.6 |

Degradación creciente: los segmentos recientes (rally 2024-25) están fuera de
la distribución de entrenamiento (drift).

## Limitaciones
1. Cambio diario impredecible (mercado eficiente): dirección ≈ AUC 0.55.
2. Drift de mercado: requiere reentrenamiento periódico.
3. Intervalos de incertidumbre mal calibrados (cobertura P5-P95 ≈ 1%).
4. Features macro publicadas con retraso (ffill).

## Consideraciones éticas
- Sin datos personales ni variables sensibles.
- El modelo no debe presentarse como asesor financiero; el error medio de
  ~204 USD/oz (y hasta 600 en 2025) debe comunicarse a los usuarios.

## Mantenimiento
- Reentrenar cuando el MAE rodante (60d) supere 1.5× el MAE de test o cuando
  el KS-drift sea persistente (script `scripts/monitor_drift.py`).
- Versionado de artefactos (`models/`), rollback conservando el modelo anterior.
- Propietario: equipo de datos (proyecto educativo).

## Versiones
- v1.0.0 (2025): Ridge α=0.0022, 110 features, RobustScaler, h=1.
- v1.1.0 (2025): + clasificador de dirección (RandomForest calibrado), endpoint `/predict_direction`.

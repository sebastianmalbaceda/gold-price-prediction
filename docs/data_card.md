# Data Card — Gold Price Prediction

## Origen y licencia
- **Fuente:** `data/raw/gold-price-prediction-dataset.csv` (compilación pública de
  series de mercado; uso académico/educativo).
- **Licencia:** sin restricciones conocidas para uso académico; no redistribuir
  con fines comerciales sin verificar. El dataset **no contiene datos personales**.
- **Frecuencia:** diaria (incluye fines de semana sin cotización).
- **Periodo:** 1901-06-30 → 2025-09-14 (45,368 filas × 61 columnas).

## Diccionario de datos (variables principales)

### Target
| Nombre | Descripción | Tipo | Unidad | Nulos (2000+) | Rango | Disponibilidad | Leakage | Tratamiento |
|---|---|---|---|---|---|---|---|---|
| `gold_spot` | Precio spot del oro | float | USD/oz | 0.2% | 263–3,430 | diaria (días hábiles) | — | target + lags |

### Exógenas usadas (35 en modelo)
| Nombre | Descripción | Tipo | Nulos | Rango | Leakage | Tratamiento |
|---|---|---|---|---|---|---|
| `us10y_yield`, `us2y_yield` | Tipos del Tesoro USA 10y/2y | float % | <5% | 0.1–5.4 | no | lag1 + retorno |
| `dxy_index`, `dxy_future` | Índice dólar (DXY) | float | <5% | 71–120 | no | lag1 + retorno |
| `usdjpy_exchange`, `eurusd_exchange`, `usdcny_exchange`, `usdinr_exchange` | Pares FX | float | <5% | varios | no | lag1 + retorno |
| `silver_spot`, `platinum_spot`, `palladium_spot`, `copper_futures` | Metales | float USD | <5% | varios | no | lag1 + retorno |
| `wti_spot`, `brent_spot`, `wti_futures`, `brent_futures` | Petróleo | float USD | <5% | 10–130 | no | lag1 + retorno |
| `vix_index`, `vix_futures`, `move_index`, `ovx_index`* | Volatilidad | float | <10% | 9–80 | no | lag1 + retorno |
| `sp500_futures`, `sp500_index`* | Renta variable | float | <30% | 700–6,000 | no | lag1 + retorno |
| `gold_futures`, `comex_micro_gold`* | Futuros oro | float USD | <25% | 260–3,400 | no | lag1 + retorno |
| `policy_uncertainty`, `geopolitical_risk` | Riesgo político/geopolítico | float | <5% | 0–700 | no | lag1 + retorno |
| `credit_spread`, `us_financial_stress_index`* | Estrés crediticio | float | <15% | 0.2–6 | no | lag1 + retorno |
| `commodities_bloomberg`, `commodities_crb` | Índices materias primas | float | <5% | 180–600 | no | lag1 + retorno |
| `bitcoin_price` | Precio BTC | float USD | <5% | 0–100k | no | lag1 + retorno |
| `etf_gold_flows`, `gdx_index` | Flujos ETF oro / mineras | float | <10% | varios | no | lag1 + retorno |
| `us_cpi`, `us_unemployment`, `fed_funds`* | Macro USA | float | **3%** | varios | publicación retrasada | lag1 (ffill) |

\* = disponible en el dataset pero **excluida del modelo** por cobertura <50% (ver `config.yaml → excluded_features`).

### Features excluidas (26) por cobertura insuficiente (<50% en 2000+)
`us_fiscal_deficit`, `us_gdp`, `google_trends_gold_element/word`, `copper_spot`,
`fx_reserves_china`, `us_m2`, `us_industrial_production`, `us_retail_sales`,
`us_consumer_sentiment`, `us_unemployment` (excluida en favor de CPI), `export_price_index`,
`us_cpi` (duplicada por correlación), `fed_funds`, `consumer_confidence`,
`us_personal_saving_rate`, `us10y_real`, `cftc_gold_positions`, `fed_balance_walcl`,
`us_financial_stress_index`, `ovx_index`, `gold_volatility_gvz`, `palladium_futures`,
`sp500_index`, `comex_micro_gold`.

> Nota: algunas quedan fuera por cobertura, otras por redundancia (|ρ|>0.98) tras el filtro de la fase 10.

## Variables derivadas (feature engineering, fase 9)
| Feature | Definición | Leakage |
|---|---|---|
| `gold_spot_lag{k}` | precio del oro hace k días (k∈{1,2,3,5,10,21}) | no (pasado) |
| `gold_ret_lag{k}` | retorno log a k días | no |
| `gold_ret_roll{w}` | retorno log de ventana w∈{5,21,63,126} | no |
| `gold_rollmean_{w}`, `gold_rollstd_{w}` | media/desv. móvil | no |
| `{exog}_lag1`, `{exog}_ret_lag1` | valor/retorno de la exógena en t-1 | no |
| `{exog}_missing` | 1 si el dato original faltaba en t | no |
| `year, month, dayofweek, quarter, dayofyear` | calendario | no |
| `target_{h}` | gold_spot en t+h (h∈{1,5,21}) | **sí por diseño** (etiqueta) |

## Calidad y gobierno
- **Inmutabilidad:** `data/raw/` nunca se modifica; los derivados van a `interim/` y `processed/`.
- **Reproducibilidad:** el pipeline completo se regenera con `python -m src.data.load_data` y `python -m src.features.build_features`.
- **Retención:** no aplica (sin datos personales). **Accesos:** repositorio interno.

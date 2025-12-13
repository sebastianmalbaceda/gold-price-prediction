# Características del proyecto: Gold Price Prediction

Este documento lista todas las características (features) recopiladas, procesadas o derivadas para el proyecto **Gold Price Prediction (TFG)**. Incluye su frecuencia, fuente y estado. En total, se recopilaron **62 columnas** (60 features + fecha + target `gold_spot`).

| #  | Nombre del archivo / Feature       | Descripción                                 | Frecuencia | Fuente / Enlace principal                                                                | Estado |
| -- | ---------------------------------- | ------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------- | ------ |
| 1  | **date**                           | Fecha de registro                           | Diario     | —                                                                                        | ✅      |
| 2  | **gold_spot.csv**                  | Precio spot del oro (XAU/USD) *(target)*    | Diario     | [Investing](https://www.investing.com/currencies/xau-usd)                                | ✅      |
| 3  | **gold_futures.csv**               | Precio futuro del oro (GCZ5)                | Diario     | [Investing](https://www.investing.com/commodities/gold)                                  | ✅      |
| 4  | **us10y_yield.csv**                | Rendimiento nominal 10-años (DGS10)         | Diario     | [FRED](https://fred.stlouisfed.org/series/DGS10)                                         | ✅      |
| 5  | **us10y_breakeven.csv**            | Breakeven / expectativas inflación (T10YIE) | Diario     | [FRED](https://fred.stlouisfed.org/series/T10YIE)                                        | ✅      |
| 6  | **fed_funds.csv**                  | Tasa efectiva fondos FED (FEDFUNDS / EFFR)  | Mensual    | [FRED](https://fred.stlouisfed.org/series/FEDFUNDS)                                      | ✅      |
| 7  | **us2y_yield.csv**                 | Rendimiento 2-años (DGS2)                   | Diario     | [FRED](https://fred.stlouisfed.org/series/DGS2)                                          | ✅      |
| 8  | **dxy_index.csv**                  | Índice Dólar (DXY)                          | Diario     | [Investing](https://www.investing.com/indices/usdollar)                                  | ✅      |
| 9  | **dxy_future.csv**                 | Futuros índice Dólar (DXU5)                 | Diario     | [Investing](https://www.investing.com/currencies/us-dollar-index)                        | ✅      |
| 10 | **sp500_index.csv**                | S&P 500                                     | Diario     | [Investing](https://www.investing.com/indices/us-spx-500)                                | ✅      |
| 11 | **sp500_futures.csv**              | Futuros S&P 500                             | Diario     | [Investing](https://www.investing.com/indices/us-spx-500-futures)                        | ✅      |
| 12 | **vix_index.csv**                  | Índice VIX (volatilidad S&P 500)            | Diario     | [FRED](https://fred.stlouisfed.org/series/VIXCLS)                                        | ✅      |
| 13 | **vix_futures.csv**                | Futuros del VIX                             | Diario     | [CBOE](https://www.cboe.com/tradable_products/vix/vix_historical_data/)                  | ✅      |
| 14 | **wti_spot.csv**                   | Petróleo WTI (spot)                         | Diario     | [FRED](https://fred.stlouisfed.org/series/DCOILWTICO)                                    | ✅      |
| 15 | **wti_futures.csv**                | Futuros WTI                                 | Diario     | [Investing](https://www.investing.com/commodities/crude-oil)                             | ✅      |
| 16 | **brent_spot.csv**                 | Petróleo Brent (spot)                       | Diario     | [FRED](https://fred.stlouisfed.org/series/DCOILBRENTEU)                                  | ✅      |
| 17 | **brent_futures.csv**              | Futuros Brent                               | Diario     | [Investing](https://www.investing.com/commodities/brent-oil)                             | ✅      |
| 18 | **silver_spot.csv**                | Plata spot (XAG/USD)                        | Diario     | [Investing](https://www.investing.com/currencies/xag-usd)                                | ✅      |
| 19 | **silver_futures.csv**             | Futuros de la plata                         | Diario     | [Investing](https://www.investing.com/commodities/silver)                                | ✅      |
| 20 | **copper_spot.csv**                | Cobre spot                                  | Mensual    | [FRED](https://fred.stlouisfed.org/series/PCOPPUSDM)                                     | ✅      |
| 21 | **copper_futures.csv**             | Futuros cobre (HGZ5)                        | Diario     | [Investing](https://www.investing.com/commodities/copper)                                | ✅      |
| 22 | **commodities_crb.csv**            | Índice CRB                                  | Diario     | [Investing](https://www.investing.com/indices/thomson-reuters---jefferies-crb)           | ✅      |
| 23 | **commodities_bloomberg.csv**      | Bloomberg Commodity Index                   | Diario     | [Investing](https://www.investing.com/indices/bloomberg-commodity)                       | ✅      |
| 24 | **cftc_gold_positions.csv**        | Posiciones netas (CFTC Managed Money)       | Diario     | [CFTC](https://www.cftc.gov/)                                                            | ✅      |
| 25 | **etf_gold_flows.csv**             | Holdings ETFs físicos (GLD/IAU)             | Diario     | [SPDR Gold Shares](https://www.spdrgoldshares.com/usa/historical-data/)                  | ✅      |
| 26 | **us_cpi.csv**                     | Índice de inflación (CPI EEUU)              | Mensual    | [FRED](https://fred.stlouisfed.org/series/CPIAUCSL)                                      | ✅      |
| 27 | **us_m2.csv**                      | Oferta monetaria M2                         | Mensual    | [FRED](https://fred.stlouisfed.org/series/M2SL)                                          | ✅      |
| 28 | **us_industrial_production.csv**   | Producción industrial                       | Mensual    | [FRED](https://fred.stlouisfed.org/series/INDPRO)                                        | ✅      |
| 29 | **us_retail_sales.csv**            | Ventas minoristas                           | Mensual    | [FRED](https://fred.stlouisfed.org/series/RSXFSN)                                        | ✅      |
| 30 | **us_unemployment.csv**            | Tasa de desempleo                           | Mensual    | [FRED](https://fred.stlouisfed.org/series/UNRATE)                                        | ✅      |
| 31 | **us_consumer_sentiment.csv**      | Sentimiento consumidor (UMich)              | Mensual    | [FRED](https://fred.stlouisfed.org/series/UMCSENT)                                       | ✅      |
| 32 | **policy_uncertainty.csv**         | Economic Policy Uncertainty Index           | Diario     | [PolicyUncertainty](https://www.policyuncertainty.com/us_monthly.html)                   | ✅      |
| 33 | **fx_reserves_china.csv**          | Reservas FX (China)                         | Mensual    | [FRED](https://fred.stlouisfed.org/series/TRESEGCNM052N)                                 | ✅      |
| 34 | **usdcny_exchange.csv**            | Tipo cambio USD/CNY                         | Diario     | [FRED](https://fred.stlouisfed.org/series/DEXCHUS)                                       | ✅      |
| 35 | **us10y_tips.csv**                 | Tasa real 10-años (TIPS)                    | Diario     | [FRED](https://fred.stlouisfed.org/series/DFII10)                                        | ✅      |
| 36 | **credit_spread.csv**              | Spread crédito (Baa-AAA)                    | Diario     | [FRED](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)                                  | ✅      |
| 37 | **gold_volatility_gvz.csv**        | Volatilidad oro (GVZ)                       | Diario     | [FRED](https://fred.stlouisfed.org/series/GVZCLS)                                        | ✅      |
| 38 | **ovx_index.csv**                  | Volatilidad petróleo (OVX)                  | Diario     | [Investing](https://www.investing.com/indices/cboe-crude-oil-volatility)                 | ✅      |
| 39 | **gdx_index.csv**                  | Market Vectors Gold Miners ETF              | Diario     | [Investing](https://www.investing.com/etfs/market-vectors-gold-miners)                   | ✅      |
| 40 | **comex_micro_gold.csv**           | Micro Futuros COMEX Oro                     | Diario     | [Investing](https://www.investing.com/commodities/comex-micro-gold-c1-futures)           | ✅      |
| 41 | **consumer_confidence.csv**        | Consumer Confidence Index (CCI)             | Mensual    | [OECD](https://www.oecd.org/en/data/indicators/consumer-confidence-index-cci.html)       | ✅      |
| 42 | **geopolitical_risk.csv**          | Geopolitical Risk Index (GPR)               | Diario     | [Iacoviello](https://www.matteoiacoviello.com/gpr.htm)                                   | ✅      |
| 43 | **us_gdp.csv**                     | PIB de EE.UU.                               | Trimestral | [FRED](https://fred.stlouisfed.org/series/GDP)                                           | ✅      |
| 44 | **us_fiscal_deficit.csv**          | Federal Surplus/Deficit                     | Anual      | [FRED](https://fred.stlouisfed.org/series/FYFSD)                                         | ✅      |
| 45 | **platinum_spot.csv**              | Platino spot                                | Diario     | [Investing](https://www.investing.com/currencies/xpt-usd)                                | ✅      |
| 46 | **platinum_futures.csv**           | Futuros platino                             | Diario     | [Investing](https://www.investing.com/commodities/platinum)                              | ✅      |
| 47 | **palladium_spot.csv**             | Paladio spot                                | Diario     | [Investing](https://www.investing.com/currencies/xpd-usd)                                | ✅      |
| 48 | **palladium_futures.csv**          | Futuros paladio                             | Diario     | [Investing](https://www.investing.com/commodities/palladium)                             | ✅      |
| 49 | **bitcoin_price.csv**              | Bitcoin (BTC/USD)                           | Diario     | [Investing](https://www.investing.com/crypto/bitcoin)                                    | ✅      |
| 50 | **google_trends_gold_element.csv** | Google Trends: "Gold (element)"             | Mensual    | [Google Trends](https://trends.google.es/trends/explore?date=all&q=%2Fm%2F025rs2z&hl=es) | ✅      |
| 51 | **google_trends_gold_word.csv**    | Google Trends: "gold"                       | Mensual    | [Google Trends](https://trends.google.es/trends/explore?date=all&geo=ES&q=gold&hl=es)    | ✅      |
| 52 | **comex_inventories.csv**          | Inventarios COMEX                           | Diario     | [TrendForce](https://datatrack.trendforce.com/Chart/content/1544/comex-inventory-gold)   | ✅      |
| 53 | **eurusd_exchange.csv**            | Tipo cambio EUR/USD                         | Diario     | [Investing](https://investing.com/currencies/eur-usd)                                    | ✅      |
| 54 | **usdjpy_exchange.csv**            | Tipo cambio USD/JPY                         | Diario     | [Investing](https://investing.com/currencies/usd-jpy)                                    | ✅      |
| 55 | **usdinr_exchange.csv**            | Tipo cambio USD/INR                         | Diario     | [Investing](https://investing.com/currencies/usd-inr)                                    | ✅      |
| 56 | **export_price_index.csv**         | Export Price Index (Nonmonetary Gold)       | Mensual    | [FRED](https://fred.stlouisfed.org/series/IQ12260)                                       | ✅      |
| 57 | **fed_balance_walcl.csv**          | Balance Sheet FED                           | Semanal    | [FRED](https://fred.stlouisfed.org/series/WALCL)                                         | ✅      |
| 58 | **us_financial_stress_index.csv**  | Financial Stress Index                      | Semanal    | [FRED](https://fred.stlouisfed.org/series/STLFSI4)                                       | ✅      |
| 59 | **us_personal_saving_rate.csv**    | Personal Saving Rate                        | Mensual    | [FRED](https://fred.stlouisfed.org/series/PSAVERT)                                       | ✅      |
| 60 | **us10y_real.csv**                 | Real Interest Rate (10Y)                    | Anual      | [FRED](https://fred.stlouisfed.org/series/REAINTRATREARAT10Y)                            | ✅      |
| 61 | **move_index.csv**                 | MOVE Index (Volatilidad bonos)              | Diario     | [Investing](https://www.investing.com/indices/ice-bofaml-move)                           | ✅      |

### Features derivadas

Además de las features originales, se planea generar un conjunto de features derivadas para capturar información histórica y dinámica del mercado (hasta 3 años atrás, periodo relevante por el aumento drástico del precio del oro). Estas se basan en funciones implementadas en el código Python del proyecto y en los notebooks, incluyendo:

* Lags de distintas longitudes
* Medias móviles (MA) de distintos periodos
* Volatilidad histórica sobre distintas ventanas
* Momentum (Rate of Change) y log-returns
* Diferencias simples y cumulative returns
* Opcionales: RSI, MACD, Bollinger Bands
* Features temporales derivadas de la fecha (`month`, `day_of_week`)

### Features no conseguidas (por disponibilidad o complejidad)

* Flujos netos a ETFs (diarios/semanales)
* Reservas oficiales de oro / compras de bancos centrales
* Producción minera de oro
* Demanda industrial o joyería (WGC)
* Open Interest en futuros de oro (CME)

### Nota sobre el uso de las features

El dataset presentado contiene todas las características recolectadas y sirve como **punto de partida** para el análisis. No todas las features se utilizarán directamente en los modelos: algunas pueden no tener lógica clara, otras pueden estar altamente correlacionadas o ser multicolineales, y algunas (como el precio futuro del oro) pueden contener información especulativa que requiere un uso cuidadoso.

Durante el proceso de análisis exploratorio (EDA), se evaluará la relevancia, consistencia y potencial predictivo de cada feature, descartando aquellas que no aporten valor o que puedan introducir sesgos. Esto garantiza que el modelo final se construya únicamente con variables útiles y justificadas.

Para más información sobre cada feature, incluyendo descripción, impacto esperado y motivos de inclusión, consulte el documento [Features Details](features_details.md).

---

**Total:** 61 columnas (59 features efectivas + `date` + `gold_spot` target)

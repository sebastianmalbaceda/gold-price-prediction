# Detalle de Features

Este documento amplía la información de cada feature incluida en el dataset inicial, explicando su significado, fuente, posible impacto en el precio del oro y motivos por los que fue considerada para el análisis.

---

## Target y columna de fecha

* **date** → Fecha del registro. Permite indexar los datos y generar features temporales (mes, día de la semana, etc.).
* **gold_spot** → Precio spot del oro (XAU). Es el target que queremos predecir.

---

## Precios y futuros del oro

* **gold_futures** → Precio futuro del oro (GCZ5). Refleja la expectativa de precio en el mercado; se incluyó por su relación potencial con el spot, aunque puede contener información especulativa.

---

## Tipos de interés y curvas de rendimiento

* **us10y_yield** → Rendimiento nominal a 10 años de EEUU. Indicador macroeconómico clave que influye en la inversión en oro como activo refugio.
* **us2y_yield** → Rendimiento a 2 años. Se utiliza junto con el 10Y para calcular la pendiente de la curva (`yield_curve_slope`), que puede indicar expectativas económicas.
* **us10y_breakeven** → Breakeven o expectativa de inflación a 10 años. Influye en el oro como cobertura frente a inflación.
* **fed_funds** → Tasa efectiva de los fondos de la FED. Refleja política monetaria y afecta la demanda de oro.

---

## Índices y mercados financieros

* **dxy_index** → Índice dólar (DXY). Un dólar fuerte suele presionar a la baja los precios del oro.
* **dxy_future** → Futuro del índice dólar. Captura expectativas sobre el dólar a futuro.
* **sp500_index** → S&P 500. Representa la confianza en los mercados de renta variable, inversamente correlacionado con oro en muchos casos.
* **sp500_futures** → Futuros del S&P 500. Similar a SP500_index pero con componente de expectativa.
* **vix_index** → Índice de volatilidad (VIX). A mayor volatilidad, suele aumentar la demanda de oro como activo refugio.
* **vix_futures** → Futuros VIX. Permite anticipar movimientos del mercado de volatilidad.

---

## Commodities

* **wti_spot / brent_spot** → Precio spot del petróleo WTI/Brent. Los precios de energía afectan la inflación y, por tanto, al oro.
* **wti_futures / brent_futures** → Futuros del petróleo. Información sobre expectativas del mercado energético.
* **silver_spot / silver_futures** → Precio y futuros de la plata. Metal correlacionado con oro; se incluye para capturar relaciones entre metales preciosos.
* **copper_spot / copper_futures** → Precio y futuros del cobre. Indicador económico industrial; puede reflejar demanda global.
* **commodities_crb / commodities_bloomberg** → Índices de commodities. Reflejan el comportamiento general del mercado de materias primas.
* **platinum_spot / platinum_futures** → Platino. Metal precioso alternativo; puede reflejar tendencias del mercado de metales.
* **palladium_spot / palladium_futures** → Paladio. Similar al platino, útil para captar tendencias correlacionadas.

---

## ETFs y posiciones de mercado

* **cftc_gold_positions** → Posiciones netas CFTC. Permite ver cómo los grandes traders especulan sobre el oro.
* **etf_gold_flows** → Holdings de ETFs físicos (GLD, IAU). Indica demanda institucional de oro físico.

---

## Macro y economía

* **us_cpi** → Índice de precios al consumidor (CPI). Medida directa de inflación, afectando demanda de oro como cobertura.
* **us_m2** → Oferta monetaria M2. Mayor masa monetaria puede incrementar precio del oro.
* **us_industrial_production** → Producción industrial. Relaciona la actividad económica con la demanda de metales.
* **us_retail_sales** → Ventas minoristas. Indicador de consumo y economía interna.
* **us_unemployment** → Tasa de desempleo. Indicador macroeconómico clave.
* **us_consumer_sentiment** → Sentimiento del consumidor. Refleja confianza económica, indirectamente puede afectar oro.
* **policy_uncertainty** → Índice de incertidumbre política. Mayor incertidumbre aumenta demanda de oro.
* **fx_reserves_china** → Reservas en divisas de China. Puede reflejar compras de oro por bancos centrales.
* **usdcny_exchange / usdjpy / eurusd / usdinr** → Tipos de cambio. Dólar fuerte/debil puede afectar precio oro.

---

## Otros indicadores financieros

* **us10y_tips** → Rendimiento real de bonos TIPS a 10 años.
* **credit_spread** → Diferencial Baa-AAA. Riesgo de crédito, afecta percepción de refugio seguro.
* **gold_volatility_gvz / ovx_index** → Volatilidad histórica del oro y petróleo.
* **gdx_index** → ETF de mineras de oro.
* **comex_micro_gold** → Futuros micro oro en COMEX.
* **consumer_confidence** → Confianza del consumidor (OECD/FRED).
* **geopolitical_risk** → Riesgo geopolítico. Incrementa demanda de oro como activo seguro.
* **us_gdp / us_fiscal_deficit / us10y_real** → PIB, déficit fiscal, tasa real. Indicadores macro que pueden influir en el oro.
* **fed_balance_walcl / us_financial_stress_index / us_personal_saving_rate / move_index** → Otros indicadores financieros y de liquidez del mercado, volatilidad, manufactura y ahorro.

---

## Google Trends y criptomonedas

* **google_trends_gold_element / google_trends_gold_word** → Tendencias de búsqueda sobre oro. Puede reflejar interés público y volatilidad.
* **bitcoin_price** → Precio de Bitcoin. Se analiza como activo alternativo que puede competir o correlacionar con oro.

---

## Inventarios y exportaciones

* **comex_inventories** → Inventarios físicos en COMEX.
* **export_price_index** → Precio de exportación del oro no monetario.

---

## Nota final sobre el uso del dataset

El dataset contiene todas las características recolectadas y sirve como **punto de partida**. No todas las features se usarán automáticamente; algunas pueden ser descartadas por alta correlación, multicolinealidad o información especulativa (como futuros). Cada feature será evaluada durante el EDA para determinar su relevancia y utilidad, asegurando que el modelo final incluya únicamente variables justificadas.

Cabe destacar que estas features han sido unidas y hay registros que datan incluso desde 1901. Sin embargo, debido a que muchas características tienen frecuencias de actualización muy altas o variables (diaria, semanal, mensual, trimestral, anual), existen más valores faltantes que registros válidos en los periodos más antiguos. Por este motivo, se considera ideal tomar como referencia el periodo desde **1980 hasta el 14 de septiembre**, fecha hasta la cual se recolectó toda la información.  

Dado que las features tienen frecuencias variadas, se requiere una **imputación correcta y diferenciada según la frecuencia de cada variable**, lo cual se explicará con detalle en el notebook.

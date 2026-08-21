# Data Card - Gold Price Prediction

## Origen y alcance

- **Fuente:** `data/raw/gold-price-prediction-dataset.csv`, compilacion publica
  de series de mercado para uso academico.
- **Periodo crudo:** 1901-06-30 -> 2025-09-14.
- **Tamano crudo:** 45,368 filas y 61 columnas: fecha, `gold_spot` y 59
  variables exogenas.
- **Privacidad:** no contiene datos personales conocidos.
- **Frecuencia:** diaria en el fichero crudo; incluye fines de semana sin
  cotizacion de oro.

## Transformacion reproducible

1. Se valida que `date` sea valida, ordenada y unica.
2. Se recorta a 2000-01-01 -> 2025-09-12 y se eliminan fines de semana.
3. Se excluyen 24 variables por cobertura o riesgo documentado en
   `configs/config.yaml`.
4. Se aplica forward-fill causal a exogenas. `gold_spot` solo se rellena hasta
   tres dias habiles y las ausencias largas no se convierten en precios
   ficticios.
5. Las exogenas con cobertura residual superior al 10% se eliminan. El
   resultado actual tiene 6,705 filas y 27 columnas.
6. Se generan lags, retornos, rolling, calendario e indicadores de ausencia.
   El parquet de features tiene 133 columnas y 83 features seleccionadas para
   el modelo final despues del filtro de redundancia.

Los derivados se regeneran con:

```bash
python -m src.data.load_data
python -m src.features.build_features
```

## Variables principales

| Grupo | Ejemplos | Tratamiento | Leakage |
|---|---|---|---|
| Target | `gold_spot` | nivel y targets futuros | target por diseño |
| Mercado | DXY, FX, plata, platino, petroleo, futuros | nivel, lag 1 y retorno cuando es positivo | no |
| Riesgo | `policy_uncertainty`, `geopolitical_risk`, VIX | nivel, lag y retorno | no |
| Calendario | `year`, `month`, `dayofweek`, `quarter`, `dayofyear` | calculado desde `date` | no |
| Ausencia | `{feature}_missing` | indicador del valor crudo ausente | no |

Las variables macro de baja cobertura (PIB, CPI, empleo, M2, ventas, etc.) se
mantienen documentadas en la auditoria, pero se excluyen del conjunto actual
por `excluded_features` o por cobertura residual. Las publicaciones tempranas
en el fichero crudo pueden introducir lookahead; se cuantifica el riesgo y se
propone aplicar `PUBLICATION_LAG` antes de un uso productivo.

## Targets y variables derivadas

| Nombre | Definicion | Leakage |
|---|---|---|
| `gold_spot_lag{k}` | precio del oro hace k observaciones | no |
| `gold_ret_lag{k}` | retorno logaritmico a k observaciones | no |
| `gold_ret_roll{w}` | retorno logaritmico de ventana w | no |
| `gold_rollmean_{w}`, `gold_rollstd_{w}` | estadisticos moviles | no |
| `{exog}_lag1`, `{exog}_ret_lag1` | valor/retorno de la exogena en t-1 | no |
| `target_{h}` | `gold_spot` en t+h, h en {1,5,21} | **si, etiqueta** |

Todas las ventanas se construyen hacia atras. Los targets futuros se excluyen
de las matrices mediante `get_feature_columns`, que elimina cualquier columna
con prefijo `target_` aunque el llamador solicite un unico horizonte.

## Calidad y gobierno

- `data/raw/` es inmutable y esta excluido de Git.
- `data/interim/` y `data/processed/` son derivados regenerables.
- El split es temporal: train 2000-2019, validation 2020-2022, test 2023-2025.
- El preprocesador se ajusta solo con train.
- La licencia y las condiciones de redistribucion de la compilacion deben
  verificarse antes de un uso comercial.

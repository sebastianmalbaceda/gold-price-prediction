# Auditoría técnica integral — rama `v2`

**Proyecto:** predicción del precio del oro  
**Fecha de auditoría:** 23 de agosto de 2026  
**Alcance de código:** exclusivamente la rama de trabajo `v2`; la rama `main` no forma parte del alcance y se mantiene intacta.  
**Commit auditado:** `bcdb3fa87cfb0171809afacd64633147b9db743c` — *Auditoria integral: robustez, seguridad y coherencia de resultados*.

---

## 1. Resumen ejecutivo

Esta auditoría revisó la coherencia metodológica, la integridad temporal, la evaluación, el código fuente, la superficie de seguridad, la infraestructura, los notebooks y la documentación del proyecto. Se contrastaron los tres informes de hallazgos originales, los dos registros de correcciones, el diff real de Git, el estado del árbol de trabajo y la nueva batería de regresión.

### Resultado central

El repositorio contiene correcciones sustantivas y bien orientadas para defectos que podían sesgar la evaluación, inducir fuga temporal, publicar métricas no reproducibles o exponer innecesariamente la API. Sin embargo, **no puede afirmarse todavía que las métricas publicadas describan el pipeline corregido**: los datos y los artefactos de modelo necesarios para ejecutar los notebooks no están disponibles en el checkout, porque están excluidos por `.gitignore`. Por ello, las cifras afectadas deben considerarse **históricas, no vigentes**, hasta reejecutar el pipeline de extremo a extremo y versionar sus nuevos artefactos.

### Inventario y priorización de hallazgos

Los informes de origen contienen **40 registros de hallazgos**. Tras iniciarse esta redacción se detectó una regresión adicional de ejecución en un notebook y se documentaron dos correcciones de empaquetado y tooling; el inventario final contiene por tanto **43 registros**. Para priorizar su impacto en decisiones de modelado se ha elevado el baseline naive de TEST a crítico: aunque estaba anotado como alto en el informe de notebooks B, invalida una comparación de referencia usada para sostener una mejora material. La fuga de etiquetas aparece en dos informes originales por abarcar dos grupos de notebooks, pero describe la misma causa raíz y se consolida en el apartado C-01.

| Severidad final | Registros documentados | Interpretación |
|---|---:|---|
| Crítica | 4 | Tres causas raíz consolidadas; C-01 está registrada dos veces en los informes de origen. |
| Alta | 16 | Errores que comprometen comparaciones, generalización, monitorización, seguridad o trazabilidad. |
| Media | 17 | Debilidades metodológicas, de coherencia, documentación y gobierno de resultados. |
| Baja | 6 | Problemas de mantenibilidad, precisión editorial, empaquetado o robustez operacional. |
| **Total** | **43** | Recuento de registros de auditoría; no equivale necesariamente a 43 causas raíz independientes. |

### Estado de remediación

| Estado | Alcance real |
|---|---|
| **Corregido y cubierto por regresión** | Las 43 pruebas nuevas cubren, entre otros, purga de etiquetas, cobertura previa al forward-fill, métricas direccionales, calibración temporal, escritura atómica, API, Benjamini–Hochberg y validaciones del predictor batch. |
| **Corregido en código, pendiente de verificación empírica** | Los cambios que alteran datos, particiones, features, targets, selección de modelos o backtests requieren ejecutar notebooks con los datos y modelos reales. |
| **Documentado o mitigado, pendiente de operación** | Dependencias, fijación de imágenes/acciones, cobertura de CI, gobierno de artefactos y endurecimiento de despliegue requieren decisiones de entrega y ejecución continuada. |

---

## 2. Metodología y limitaciones

### 2.1 Metodología aplicada

La revisión siguió cinco líneas de evidencia:

1. **Lectura completa de los hallazgos previos:** `audit_notebooks_A.md`, `audit_notebooks_B.md` y `audit_notebooks_C.md`, que cubren notebooks 01–23, el despliegue y la documentación.
2. **Lectura completa de los registros de corrección:** `fixes_docs.md` y `fixes_notebooks.md`, incluyendo el antes/después de 143 intervenciones registradas en notebooks.
3. **Inspección del cambio efectivo:** `git diff --stat`, `git diff -- src scripts pytest.ini Dockerfile .github configs`, `git status --short`, `git diff --check` y el contenido completo de `tests/test_audit_fixes.py`.
4. **Revisión estática de código y configuración:** funciones de carga, split, features, métricas, clasificación, API, escritura de artefactos, monitorización, predictor batch, Docker y CI.
5. **Conciliación de resultados:** cada hallazgo se clasifica como corregido por código, verificable por prueba de regresión, pendiente de reejecución empírica o pendiente de operación/documentación.

La revisión incluyó una comprobación estática AST notebook a notebook. Su valor quedó demostrado por una regresión introducida durante la propia corrección de `18b_auditoria_datos.ipynb`: al retirar un experimento de publication lag no concluyente se eliminó también el contexto compartido que requerían celdas posteriores. El AST detectó la situación de nombres usados antes de definirse; la corrección restauró el contexto y la comprobación final confirma que ninguno de los 21 notebooks presenta ya ese patrón. Es un control necesario porque una corrección metodológica también puede introducir una regresión de ejecución.

Se verificó además que el diff no contiene errores de espacios en blanco según `git diff --check`.

### 2.2 Qué sí pudo verificarse

- El commit, el alcance de los cambios y el estado de Git.
- La implementación de las correcciones en `src/`, `scripts/`, configuración, infraestructura y CI.
- La existencia de **43** funciones de prueba en `tests/test_audit_fixes.py`.
- La coherencia estática de los cambios respecto de los defectos descritos: por ejemplo, que `purge_label_overlap` existe y se aplica en el flujo; que la métrica direccional excluye empates; que se incorpora corrección FDR; y que la API incluye controles de autenticación, tamaño y concurrencia.
- El inventario de cambios rastreados: el `git diff --stat` inspeccionado registra **49 ficheros cambiados, 1.632 inserciones y 829 eliminaciones**.

### 2.3 Límites materiales de la auditoría

1. **No fue posible ejecutar los notebooks.** El checkout no contiene los datos de `data/` ni los artefactos `models/*.joblib`, ambos excluidos por `.gitignore`. Por tanto, no se recalcularon métricas, curvas, umbrales, backtests ni informes generados.
2. **Las comprobaciones de calidad estática sí se ejecutaron en este entorno**, sobre un entorno virtual dedicado con Python 3.12 creado durante la auditoría. La suite completa se lanzó repetidamente después de cada bloque de correcciones, con **80 pruebas superadas** en la última ejecución. Lo que no pudo ejecutarse es el pipeline de datos, por el motivo del punto 1: los tests cubren lógica de biblioteca con datos sintéticos, no el entrenamiento real.
3. **`pip-audit` no pudo ejecutarse** por falta de espacio en disco del entorno. Esta limitación queda abierta; no debe interpretarse como una auditoría de vulnerabilidades de dependencias satisfactoria.
4. Las correcciones de notebooks pueden ser estructuralmente correctas y aun así producir resultados distintos al reentrenar. En consecuencia, no se ha dado por válida ninguna cifra afectada sólo porque su código haya sido modificado.
5. Algunas observaciones sólo pueden remediarse en código, no editando resultados históricos: una fuga en la partición, un backtest con información contemporánea o un umbral escogido sobre TEST obligan a **regenerar** los artefactos derivados.

---

## 3. Hallazgos críticos

### C-01. Fuga de etiquetas entre particiones temporales

**Área:** datos, particionado y evaluación.  
**Registros de origen:** A-01 y B-01; dos registros para la misma causa raíz.  
**Estado:** corregido en código y cubierto por pruebas; resultados históricos pendientes de reejecución.

| Aspecto | Evaluación |
|---|---|
| Qué era | En una predicción a horizonte `h`, las últimas `h` filas de train poseen etiquetas cuyo precio futuro ya cae en validación. El `gap` de `TimeSeriesSplit` protege los pliegues internos de validación cruzada, pero no elimina ese solapamiento entre las particiones consecutivas del split temporal principal. |
| Por qué es grave | El entrenamiento puede recibir información de etiquetas pertenecientes al período de validación. Esto sesga selección de hiperparámetros, evaluación y cualquier conclusión de generalización. |
| Corrección aplicada | `src/data/split.py` incorpora `purge_label_overlap`; los notebooks actualizados deben aplicarlo inmediatamente después de `temporal_split`. La verificación estática registrada identifica 14 usos de `temporal_split` acompañados por 14 usos de `purge_label_overlap`. |
| Cifras invalidadas | Todas las métricas obtenidas con el split no purgado: MAE, RMSE, R², sMAPE, MAPE, métricas de clasificación/dirección, resultados de tuning, selección de modelo, backtests y análisis de robustez derivados. |

La corrección evita reproducir la fuga en ejecuciones futuras; no sanea resultados ya calculados. Deben reejecutarse los notebooks desde el particionado y preservarse un manifiesto de filas por partición, horizonte y `gap` aplicado.

### C-02. Umbral de Sharpe optimizado directamente sobre TEST en `17b`

**Área:** evaluación financiera y backtesting.  
**Registro de origen:** B-02.  
**Estado:** documentado/corregido en el notebook; requiere reejecución completa.

| Aspecto | Evaluación |
|---|---|
| Qué era | El notebook `17b_acierto_y_backtest.ipynb` exploraba 16 umbrales, entre 0,50 y 0,65, seleccionándolos sobre el conjunto TEST para maximizar Sharpe. |
| Por qué es grave | TEST deja de ser una muestra de evaluación final independiente. La selección se adapta al ruido del propio TEST y sobreestima el rendimiento de la estrategia. |
| Corrección aplicada | Verificada por lectura directa de la celda 14 del notebook: la rejilla de 16 umbrales se recorre ahora sobre VALIDACIÓN (`p_val`, `ret_val`), se selecciona `thr_opt` por Sharpe y se imprime explícitamente «Umbral seleccionado en VALIDATION»; solo después se reentrena sobre train+val y se evalúa **una única vez** sobre TEST con ese umbral congelado. Se añadieron además el desplazamiento `shift(1)` de la señal (la posición se toma con la probabilidad del día anterior, no la del mismo día) y `prepend=0.0` en el cálculo de costes, para no omitir el coste de la primera operación. |
| Cifras invalidadas | **Sharpe 1,236** y **retorno acumulado 28,0 %** publicados por ese procedimiento. También quedan invalidadas las comparaciones de umbral y cualquier conclusión de superioridad basada en ellas. |

No es suficiente con cambiar texto o fijar retrospectivamente el umbral. Hay que rerunear la selección en validación, congelar el umbral elegido y entonces evaluar una sola vez sobre TEST.

### C-03. Baseline naive de TEST construido con una constante de train

**Área:** evaluación de regresión y comparabilidad de baselines.  
**Registro de origen:** B-09, elevado de alto a crítico por impacto directo en la afirmación de mejora.  
**Estado:** corregido/documentado en notebooks; requiere reejecución.

| Aspecto | Evaluación |
|---|---|
| Qué era | El baseline naive de TEST usaba una constante proveniente de train, en vez de un naive temporal que utilice el último valor disponible en cada instante de predicción. |
| Por qué es grave | El baseline queda artificialmente débil y la comparación con el modelo no responde a la pregunta operativa correcta: si éste supera a la persistencia temporal disponible en producción. |
| Corrección aplicada | Los notebooks separan explícitamente el baseline temporal y revisan la narrativa de comparación. La corrección sólo será concluyente cuando se recalcule con las series reales. |
| Cifras invalidadas | **MAE naive 879,667** y la afirmación de **“mejora” de 88,661 %**. |

La invalidez de la referencia afecta la interpretación de la calidad relativa del modelo, incluso si algunas métricas absolutas de regresión permanecieran numéricamente cercanas tras la reejecución.

---

## 4. Hallazgos de severidad alta

Los 16 registros altos se presentan por área. Cuando dos informes registran la misma deficiencia direccional, se preserva el recuento de registros pero se evita duplicar el diagnóstico técnico.

### 4.1 Datos, features y particionado

| ID | Hallazgo y riesgo | Corrección / estado | Impacto en resultados |
|---|---|---|---|
| H-01 | La cobertura de exógenas se medía después de `forward-fill`; una serie con una sola observación podía aparentar cobertura completa. | `src/data/load_data.py` calcula cobertura real pre-ffill y hueco interno máximo; incorpora `min_coverage_before_ffill` y `max_ffill_gap_days`, con valores por defecto que preservan el comportamiento histórico. Cubierto por regresión; requiere datos para validar qué series quedan excluidas. | Cambia potencialmente el universo de exógenas, matrices de features y métricas posteriores. |
| H-02 | Las variables exógenas se definían con posible look-ahead y el logaritmo no se recomputaba correctamente en cada lag. | `src/features/build_features.py` decide el conjunto en entrenamiento y recompone la transformación por lag. Corregido estáticamente; pendiente de rerun. | Puede alterar features seleccionadas, importancias y desempeño. |
| H-03 | No se imponía el retraso de publicación de las variables exógenas; el diagnóstico publicado indicaba `impacto_auc=0.0` y AUC `0.560439749`. | Los notebooks de auditoría documentan el problema y la necesidad de lag operativo. Pendiente de aplicar y validar con calendario de publicación por fuente. | Esa AUC y la conclusión de impacto nulo no son utilizables operacionalmente. |

### 4.2 Modelado, clasificación y validación

| ID | Hallazgo y riesgo | Corrección / estado | Impacto en resultados |
|---|---|---|---|
| H-04 | `CalibratedClassifierCV` empleaba K-fold aleatorio en una serie temporal. | `src/models/classifier.py` usa `TimeSeriesSplit` con `gap` y horizonte; `make_direction_targets` ahora rechaza NaN en lugar de convertirlos silenciosamente en clase 0. Cubierto por regresión. | Calibración, probabilidades, AUC y umbrales previos requieren recálculo. |
| H-05 | El conjunto TEST se consultaba antes del cierre final de la selección de modelo. | Los notebooks separan el uso de validación y TEST; requiere disciplina de ejecución y rerun. | Las comparaciones y decisiones basadas en TEST no son una validación final limpia. |
| H-06 | La exactitud direccional se calculaba con una definición incorrecta; los empates podían contar como acierto. El mismo defecto aparece en dos registros de origen (A-03 y B-05). | `src/evaluation/metrics.py` excluye empates y añade `forecast_directional_accuracy(y_true, y_pred, y_current)`; `regression_metrics` guarda `directional_accuracy_definition`. Cubierto por regresión. | La exactitud direccional publicada, incluido **46,1201 %**, debe regenerarse con la definición persistida. |
| H-07 | El “gap” de generalización comparaba validación in-sample de un modelo entrenado con train+validación; se comunicó **24,9264** como si fuera evaluación honesta. | Se corrige el flujo de evaluación y se documenta la distinción. Pendiente de rerun. | El valor **24,9264** es inválido como gap de generalización; el MAE de validación previamente calculado de **60,4433** y el MAE TEST de **99,7458** son referencias históricas, no sustitutos de una nueva evaluación purgada. |
| H-08 | Diagnósticos de regularización consultaban validación/TEST y daban, entre otros síntomas, `val_auc=1`. | Corrección en los notebooks; pendiente de ejecución. | Las conclusiones sobre regularización y selección quedan condicionadas por fuga de evaluación. |

### 4.3 Evaluación financiera y monitorización

| ID | Hallazgo y riesgo | Corrección / estado | Impacto en resultados |
|---|---|---|---|
| H-09 | La señal del backtest se aplicaba sobre el mismo día, introduciendo look-ahead de ejecución. | Se documenta el desplazamiento temporal necesario de la señal; requiere rerun. | Rentabilidades, Sharpe, operaciones y gráficos del backtest deben recalcularse. |
| H-10 | `scripts/monitor_drift.py` realizaba 83 contrastes KS a \(\alpha=0,01\) sin corrección múltiple; la probabilidad calculada de al menos un falso positivo era **56,58 %**. | Implementa FDR Benjamini–Hochberg, persiste método/rangos/alertas sin corregir y fue validado contra `statsmodels.stats.multitest.multipletests` en 200 casos aleatorios con coincidencia exacta. | Las 34 alertas históricas y sus conclusiones deben interpretarse de nuevo con FDR. |
| H-11 | La ventana reciente podía solaparse con la referencia train+validación, comparando parte de una muestra consigo misma. | El script recorta la referencia antes del inicio reciente y aborta si quedan menos de 30 filas. Cubierto por regresión; pendiente de corrida con datos. | Las pruebas KS y alertas de drift emitidas sobre ventanas solapadas son inválidas. |
| H-12 | El umbral rolling de MAE utilizaba futuro del propio TEST. | Se corrige la separación temporal en notebooks; pendiente de rerun. | Alertas/umbrales rolling de error y cualquier evaluación asociada deben regenerarse. |

### 4.4 API, infraestructura y documentación operativa

| ID | Hallazgo y riesgo | Corrección / estado | Impacto en resultados |
|---|---|---|---|
| H-13 | El diagrama de despliegue mostraba datos crudos como entrada directa de la API, cuando el contrato real es un vector de features preparado. | Documentación ajustada. | Reduce el riesgo de integrar una API con un contrato de entrada erróneo. |
| H-14 | `docker-compose.yml` exponía la API en todas las interfaces sin autenticación, límite de tamaño ni control de abuso. | La publicación se limita a loopback; la API añade autenticación opcional y límite de cuerpo. Véase sección 7. | Mitiga exposición local, pero el despliegue externo requiere controles de red y una clave configurada. |
| H-15 | El README afirmaba 26 variables diarias y otras frecuencias que no coincidían con el artefacto: se identificaron 25 variables, todas diarias. | Documentación corregida. | Se elimina una discrepancia material de inventario; no altera por sí sola datos ni modelo. |

### 4.5 Notebooks y regresiones de corrección

| ID | Hallazgo y riesgo | Corrección / estado | Impacto en resultados |
|---|---|---|---|
| H-16 | En `notebooks/18b_auditoria_datos.ipynb`, celda 10, la reescritura del experimento de publication lag eliminó también el contexto compartido que definía `parts`, `sel_cols`, `drop_warmup` y `json`. Las celdas 12 (VIF), 14 (cobertura) y 16 (guardado de `reports/data_audit.json`) quedaban expuestas a `NameError`, de modo que el notebook no era ejecutable de principio a fin. | Se restauró carga de features, `temporal_split`, `purge_label_overlap`, lectura de `models/feature_list.json` e importación de `json`, manteniendo la conclusión honesta: sin fechas de publicación/vintages el experimento es no concluyente y `impacto_auc` se persiste como `null`, no `0.0`. La comprobación AST final confirma ausencia de nombres usados antes de definirse en los 21 notebooks. | Sin la corrección no se podían generar VIF, cobertura ni `reports/data_audit.json`; una ejecución posterior habría fallado pese a que la narrativa de publication lag estuviera metodológicamente corregida. |

---

## 5. Hallazgos de severidad media

### 5.1 Datos, features y configuración

| ID | Hallazgo | Corrección / situación actual | Consecuencia |
|---|---|---|---|
| M-01 | La EDA afirmaba analizar retornos exógenos, pero utilizaba niveles. | Notebook/documentación corregidos; requiere ejecución para actualizar gráficos. | Interpretaciones de estacionariedad, correlación y relación con el target deben revalidarse. |
| M-02 | Se anunciaban cuatro baselines sin implementar todos los prometidos. | La documentación y el flujo se ajustaron. | Las comparaciones históricas no justifican la amplitud de baseline anunciada. |
| M-03 | El tuning ignoraba parámetros presupuestarios como `n_trials`/timeout. | Notebooks corregidos para respetar configuración. | La reproducibilidad de la búsqueda previa era incompleta. |
| M-04 | `params.yaml` discrepaba del JSON generado por tuning y podía dejar parámetros mutados en memoria. | Se corrige el flujo y se documenta la fuente de verdad. | Los hiperparámetros históricos pueden no ser los que efectivamente produjeron un resultado. |
| M-05 | Se usaba el valor absoluto del sesgo con etiqueta MAE. | Etiqueta/cálculo corregidos en notebook. | La métrica rotulada como MAE no era la definición estándar. |
| M-06 | El CPI se trataba como constante diaria sin explicitar el tratamiento del dato mensual. | Se documenta la transformación y queda pendiente validación con calendario de publicación. | Riesgo de interpretación temporal errónea de una exógena macroeconómica. |

### 5.2 Inferencia, evaluación estadística y riesgo

| ID | Hallazgo | Corrección / situación actual | Consecuencia |
|---|---|---|---|
| M-07 | Bootstrap iid y t-test se aplicaban a una serie temporal; se publicaron IC AUC **[0,490; 0,574]** y `p=0,0786`. | Se corrige/documenta el enfoque de bloques temporales; requiere rerun. | Esos intervalos y p-valor no sustentan inferencia temporal válida. |
| M-08 | Comparaciones de volatilidad usaban muestras distintas y comunicaban mejora **16,249 %**. | Se armonizan períodos/muestras en notebook. | La magnitud comparativa previa debe regenerarse. |
| M-09 | El coste de entrada no estaba incluido en el backtest de transacciones. | Corrección metodológica documentada; requiere parámetros de costes y rerun. | La rentabilidad neta y Sharpe histórico pueden estar sobreestimados. |
| M-10 | En `17d`, el texto describía una mejora cercana a 8 %, mientras la salida mostraba **16,248657… %**. | Texto corregido. | Era una inconsistencia editorial de una cifra de riesgo. |
| M-11 | `18_errores_explicabilidad` comunicaba cobertura bootstrap **4,97076 %** (4,97 %) y ancho 26,42 como si fueran intervalos adecuados. | Diagnóstico y etiquetas revisados; requiere nueva ejecución. | La cobertura no justifica el uso de esos intervalos como incertidumbre calibrada. |

### 5.3 Monitorización, documentación y entrega

| ID | Hallazgo | Corrección / situación actual | Consecuencia |
|---|---|---|---|
| M-12 | El notebook KS no aplicaba corrección múltiple y redondeaba p-valores de forma poco informativa. | Alineado con FDR Benjamini–Hochberg en el script; pendiente de corrida real. | Los umbrales/alertas anteriores no son comparables al nuevo método. |
| M-13 | Se monitorizaban diez variables, pero la política consideraba alerta cualquiera sin una taxonomía de severidad. | Documentado; política operativa aún pendiente. | Riesgo de ruido operacional y de falta de priorización. |
| M-14 | La documentación verificaba sobre todo tamaño de archivos, no contenido, esquema ni procedencia. | Se ampliaron guías, pero hace falta validación de contratos automatizada. | Continúa el riesgo de artefactos presentes pero incorrectos. |
| M-15 | Métricas de clasificación calibrada (por ejemplo AUC alrededor de **0,532**) no estaban versionadas en el informe final. | Se añade aviso/documentación; persiste la necesidad de versionar `direction_metrics.json`. | No hay trazabilidad suficiente de resultados de clasificación publicados. |
| M-16 | Se declaraban dependencias “fijadas” aunque el conjunto analizado contenía 30 requisitos con operadores `>=`. | La documentación corrige la afirmación, no la reproducibilidad. | El entorno puede variar entre instalaciones y cambiar resultados o exposición a CVE. |
| M-17 | Uvicorn se planteaba como servidor de producción sin la arquitectura de proceso/proxy correspondiente. | Documentación revisada. | Requiere un despliegue de producción explícito, con terminación TLS, workers y controles perimetrales. |

---

## 6. Hallazgos de severidad baja

| ID | Área | Hallazgo | Estado |
|---|---|---|---|
| L-01 | Utilidades/notebooks | El descubrimiento del directorio raíz podía fallar silenciosamente y producir comportamientos dependientes del directorio de ejecución. | Mejorado/documentado; conviene fallar de forma explícita cuando no se encuentre el root esperado. |
| L-02 | Documentación | Persistían cifras obsoletas, como 146 features frente a 128 y Ridge MAE cercano a 55 frente a 69,98. | Corregido en documentación/notebooks; todas las cifras deben regenerarse tras el rerun. |
| L-03 | Documentación de riesgo | La narración de `17d` no coincidía con la salida numérica de volatilidad. | Corregido editorialmente. |
| L-04 | Datos/documentación | `data/README.md` contenía una inconsistencia de suma/inventario. | Corregido; revisar el inventario contra el dataset real en la siguiente ejecución. |
| L-05 | Empaquetado y tooling | Aunque el código de `src/` contiene anotaciones de tipos, no existía `src/py.typed`. Los consumidores del paquete instalado ignoraban silenciosamente esas anotaciones, por lo que el trabajo de tipado no aportaba comprobación aguas abajo. | Corregido: se añade el marcador vacío PEP 561 y `[tool.setuptools.package-data]` lo declara como `src = ["py.typed"]` para incluirlo en la distribución. Las anotaciones pasan a ser visibles para los consumidores tras construir e instalar el paquete; conviene verificarlo en el artefacto de distribución. |
| L-06 | Empaquetado y tooling | La configuración de formato y estilo estaba duplicada en argumentos de `.pre-commit-config.yaml` y CI. Un desarrollador que ejecutase `black .` o `isort .` localmente podía obtener un resultado distinto al de CI por divergencia de parámetros. | Corregido: `pyproject.toml` centraliza `[tool.black]`, `[tool.isort]`, `[tool.mypy]` y `[tool.ruff]`. Mypy queda deliberadamente no bloqueante (`disallow_untyped_defs = false`): el código no está anotado al 100 % y activar `--strict` produciría cientos de errores inabordables. `pyproject.toml` parsea con `tomllib`; `black --check` e `isort --check-only` siguen limpios. Disminuye la divergencia de tooling y establece una base gradual para el tipado, pero no equivale aún a verificación estricta de tipos. |

---

## 7. Seguridad

### 7.1 Exposición de la API

El cambio de `docker-compose.yml` limita la publicación de la API a loopback. Esta medida reduce la exposición accidental del entorno de desarrollo, pero no sustituye una política de red, proxy inverso, TLS, observabilidad y autenticación obligatoria cuando se despliegue fuera de la máquina local.

### 7.2 Autenticación y tamaño de petición

`src/api/main.py` incorpora `GOLD_API_KEY` y valida la cabecera `X-API-Key` mediante comparación segura. La autenticación es **opcional por diseño** para preservar el comportamiento histórico: si la variable no se configura, la API sigue permitiendo el acceso. En producción debe tratarse como obligatoria y gestionarse como secreto.

También se añade `API_MAX_BODY_BYTES`, con valor por defecto de **128 KiB**, respuesta HTTP 413 y un límite Pydantic de **256 features**. El control basado en `Content-Length` es una mejora concreta frente a la ausencia de límite, pero deben probarse las rutas de transferencia sin longitud declarada y añadirse defensa perimetral para solicitudes grandes o maliciosas.

### 7.3 Concurrencia, metadatos y estabilidad

La carga perezosa de artefactos estaba expuesta a una condición de carrera: un hilo podía observar el modelo ya asignado con el preprocesador aún nulo. Se añadieron cerrojos de carga, caché de metadatos, lectura de YAML no repetida en cada petición, uso coherente del horizonte y manejo seguro de `_model_version` incompleto. Estas correcciones están cubiertas por las nuevas pruebas de regresión, aunque el rendimiento y comportamiento concurrente en un servidor real deben ensayarse bajo carga.

### 7.4 Deserialización y dependencias

Los modelos se cargan desde `joblib`; ese formato no es seguro frente a archivos no confiables. El control adecuado es de cadena de suministro: sólo cargar artefactos provenientes de un pipeline autenticado, con checksum/firma, permisos restrictivos y procedencia versionada. No debe aceptarse nunca una ruta o un artefacto proporcionado por usuarios.

La auditoría detectó que los requisitos usan `>=` y no un lockfile reproducible. Además, `pip-audit` no pudo ejecutarse por falta de espacio en disco, por lo que no existe una conclusión vigente sobre vulnerabilidades transitivas. El CI añade un paso de `pip-audit` con continuidad ante error; debe evolucionar a una política que haga visible y gestione el resultado de forma gobernada.

### 7.5 Herramientas de seguridad

La evidencia de verificación registrada indica que `bandit -ll -r src scripts` no produjo hallazgos. Este resultado es útil, pero no cubre dependencias, secretos, configuración de red, deserialización de artefactos confiables ni vulnerabilidades de despliegue.

---

## 8. Correcciones que exigen reejecutar el pipeline

Esta sección prevalece sobre cualquier lectura de métricas históricas. Una corrección de código no actualiza automáticamente los resultados ya grabados en notebooks, JSON, gráficos o documentación.

| Cifra o artefacto publicado afectado | Motivo de invalidez | Notebook(s) a reejecutar | Salida que debe regenerarse |
|---|---|---|---|
| Todas las métricas de train/validación/TEST obtenidas con splits no purgados | C-01: etiquetas futuras de train solapan validación para horizonte `h`. | `07_particion`, `08_10_preprocessing_features`, `11_13_baselines_modelado`, `14_tuning`, `15_16_seleccion_entrenamiento`, `17_test_final` y dependientes. | Particiones, matrices, modelos, métricas y artefactos de evaluación. |
| **MAE naive 879,667** y **mejora 88,661 %** | C-03: el baseline de TEST no es una persistencia temporal válida. | `11_13_baselines_modelado`, `17_test_final`. | Tabla de baselines y narrativa comparativa. |
| **Sharpe 1,236** y **retorno acumulado 28,0 %** | C-02: selección de 16 umbrales sobre TEST; además H-09 y M-09 afectan ejecución y costes. | `16_direccion_clasificacion`, `17b_acierto_y_backtest`. | Umbral congelado desde validación, curva de capital, operaciones, costes, retorno y Sharpe netos. |
| Exactitud direccional, incluido **46,1201 %** | H-06: definición equivocada/empates y falta de referencia `y_current`; también dependen de splits. | `05_target_y_metricas`, `15_16_seleccion_entrenamiento`, `16_direccion_clasificacion`, `17_test_final`, `17c_robustez_direccion`. | `direction_metrics.json`, tablas de clasificación, gráficos y texto. |
| “Gap” **24,9264**, validación **60,4433** y TEST **99,7458** | H-07: evaluación in-sample para el gap; C-01 cambia el split. | `15_16_seleccion_entrenamiento`, `17_test_final`. | Evaluación honesta de validación/TEST y narrativa de generalización. |
| Métricas de regresión históricas: MAE **99,745767**, RMSE **125,076380**, R² **0,935166**, sMAPE **3,939475**, MAPE **3,835497**, `n=684` | Coincidían internamente con artefactos previos, pero proceden del pipeline anterior a la purga y otras correcciones. | Desde `07_particion` hasta `17_test_final`. | Informe final de regresión y artefactos versionados. |
| AUC **0,560439749** e `impacto_auc=0.0` de retraso de publicación | H-03: no se aplicaba un lag de publicación operativo. | `18b_auditoria_datos` y notebooks de features/clasificación aguas abajo. | Auditoría de disponibilidad, matriz de features y AUC recalculada. |
| IC AUC **[0,490; 0,574]**, `p=0,0786`, cobertura **4,97076 %**, ancho **26,42** | M-07 y M-11: inferencia iid/intervalos no calibrados. | `17b_acierto_y_backtest`, `18_errores_explicabilidad`. | Inferencia por bloques, intervalos y diagnóstico de calibración. |
| Mejora de volatilidad **16,249 %** / texto cercano a 8 % | M-08 y M-10: muestras diferentes e inconsistencia narrativa. | `17d_volatilidad_riesgo`. | Comparación sobre idéntica muestra, cifra y texto. |
| 34 alertas KS históricas y conclusiones de drift | H-10/H-11/M-12: sin FDR y con posible solapamiento referencia–reciente. | `23_monitorizacion` y `scripts/monitor_drift.py`. | Informe de drift con método BH, rangos temporales y alertas crudas/corregidas. |
| Métricas calibradas no versionadas, por ejemplo AUC alrededor de **0,532** | M-15: falta de artefacto versionado y cambian calibración, targets y split. | `16_direccion_clasificacion`, `17c_robustez_direccion`, `22_documentacion`. | `direction_metrics.json` versionado, model card y documentación. |

### Orden recomendado de reejecución

1. Restaurar una instantánea identificable de datos y artefactos de entrada, con hashes y fechas de disponibilidad.
2. Ejecutar desde `01_problema_y_diseno` hasta `07_particion`, validando el corte temporal y la purga.
3. Regenerar features y baselines antes de cualquier tuning o selección.
4. Reentrenar, seleccionar exclusivamente con validación y congelar configuración, umbrales y calendario de publicación.
5. Evaluar TEST una sola vez, ejecutar backtests con señal desplazada y costes explícitos.
6. Regenerar robustez, explicabilidad, monitorización, model card, README, reportes y artefactos JSON versionados.
7. Adjuntar a la entrega el log de ejecución, versiones de dependencias, hashes de datos/modelos y resultados de pruebas.

---

## 9. Verificación y pruebas

| Comprobación | Resultado declarado o observado | Alcance / límite |
|---|---|---|
| Suite de tests | **37** pruebas baseline superadas; tras las correcciones, **80** superadas: 37 originales intactas + **43** nuevas en `tests/test_audit_fixes.py`. Las 80 siguieron pasando tras los cambios de empaquetado/tooling. | Ejecutada en este entorno con `python -m pytest tests/` sobre un venv Python 3.12; se relanzó tras cada bloque de correcciones. Cubre lógica de biblioteca con datos sintéticos, no el entrenamiento real. |
| Pruebas de regresión | El fichero nuevo contiene 43 funciones `test_`. | Inspección estática completa realizada. |
| Bandit | `bandit -ll -r src scripts`: sin hallazgos. | Resultado registrado; no sustituye revisión de dependencias o despliegue. |
| Estilo | `flake8`, `black --check` e `isort --check-only` con line-length 100: limpios; tras centralizar configuración, `black --check` e `isort --check-only` continuaron limpios. | Resultado registrado; `pyproject.toml` también parsea con `tomllib`. |
| Notebooks | 21 notebooks validados con `nbformat.validate`. | Valida estructura, no ejecución ni exactitud de métricas. |
| Análisis AST | Ningún nombre usado antes de definirse en los 21 notebooks. El control detectó y permitió corregir la regresión de contexto compartido en `18b`, que habría provocado `NameError` en las celdas de VIF, cobertura y guardado. | Comprobación estática, no semántica ni de datos. |
| Particionado | 14 usos de `temporal_split` acompañados por 14 usos de `purge_label_overlap`. | Confirma inserción estructural de la purga; requiere datos para verificar filas finales. |
| Benjamini–Hochberg | Validado contra `statsmodels.stats.multitest.multipletests` en **200** casos aleatorios, con coincidencia exacta. | Cubre equivalencia de implementación; no valida alertas reales sin ejecutar datos. |
| `pip-audit` | **No ejecutado** por falta de espacio en disco. | Hallazgo abierto de seguridad/dependencias. |
| Ejecución de notebooks | **No ejecutada**. | Faltan `data/` y `models/*.joblib`, excluidos por `.gitignore`. |
| Integridad del diff | `git diff --check` sin salida. | No se detectaron errores de espacios en blanco en los cambios inspeccionados. |

---

## 10. Recomendaciones pendientes

Las siguientes acciones no deben confundirse con correcciones ya concluidas:

1. **Crear y exigir un lockfile de dependencias.** Conservar `requirements` de alto nivel si se desea, pero resolver a versiones exactas, con hashes cuando sea viable, y actualizarlo mediante una política controlada.
2. **Fijar el digest real de la imagen base.** El Dockerfile conserva `python:3.11-slim` como etiqueta; el comentario sobre digest no equivale a una referencia `@sha256:`. Debe fijarse y revisarse periódicamente.
3. **Fijar acciones de GitHub por SHA.** Las referencias de CI deben pasar de tags mutables a commits SHA con una estrategia de actualización y revisión.
4. **Añadir cobertura a CI.** Establecer medición, umbral gradual y reporte de cobertura, priorizando split temporal, carga de datos, API, monitorización y scripts operativos.
5. **Versionar `direction_metrics.json`.** Debe incluir definición de métrica direccional, horizonte, particiones, hash de datos, configuración, fecha y versión del modelo. Evitar que la documentación dependa de salidas efímeras de notebook.
6. **Reejecutar el pipeline completo.** Es la condición indispensable para sustituir cifras invalidadas por resultados del código corregido.
7. **Endurecer la política de API en producción.** Configurar `GOLD_API_KEY`, secretos, TLS/proxy, límites perimetrales, registro de intentos fallidos y pruebas de carga/concurrencia.
8. **Establecer gobierno de datos y artefactos.** Versionar o registrar hashes, procedencia, calendario de publicación y retención de modelos, sin introducir datos sensibles en Git.
9. **Convertir `pip-audit` en un control efectivo.** Resolver la limitación de espacio, ejecutar el análisis y definir severidades/bloqueos y proceso de excepción.

---

## 11. Anexo — inventario de ficheros modificados

El `git diff --stat` inspeccionado registra **49 ficheros rastreados**, con **1.632 inserciones** y **829 eliminaciones**. Las siguientes tablas reproducen el inventario y el recuento por fichero del diff real; los valores se expresan como `+líneas / −líneas`.

### 11.1 Infraestructura, configuración y documentación (15 ficheros)

| Fichero | + / − |
|---|---:|
| `.env.example` | 6 / 1 |
| `.github/workflows/ci.yml` | 29 / 2 |
| `CHANGELOG.md` | 3 / 1 |
| `Dockerfile` | 9 / 1 |
| `README.md` | 6 / 2 |
| `SECURITY.md` | 10 / 2 |
| `configs/config.yaml` | 9 / 1 |
| `data/README.md` | 1 / 1 |
| `docker-compose.yml` | 5 / 1 |
| `docs/api_manual.md` | 24 / 3 |
| `docs/audit_report.md` | 5 / 2 |
| `docs/informe_tecnico.md` | 2 / 1 |
| `docs/methodology-guide.md` | 4 / 5 |
| `docs/model_card.md` | 4 / 0 |
| `pyproject.toml` | 33 / 0 |

### 11.2 Notebooks (21 ficheros)

| Fichero | + / − |
|---|---:|
| `notebooks/01_problema_y_diseno.ipynb` | 14 / 9 |
| `notebooks/02_datos_y_auditoria.ipynb` | 14 / 9 |
| `notebooks/04_eda.ipynb` | 17 / 13 |
| `notebooks/05_target_y_metricas.ipynb` | 14 / 9 |
| `notebooks/07_particion.ipynb` | 16 / 10 |
| `notebooks/08_10_preprocessing_features.ipynb` | 17 / 11 |
| `notebooks/11_13_baselines_modelado.ipynb` | 33 / 23 |
| `notebooks/14_tuning.ipynb` | 40 / 18 |
| `notebooks/15_16_seleccion_entrenamiento.ipynb` | 23 / 23 |
| `notebooks/16_direccion_clasificacion.ipynb` | 71 / 92 |
| `notebooks/16c_deep_learning_clasificacion.ipynb` | 16 / 10 |
| `notebooks/17_test_final.ipynb` | 23 / 19 |
| `notebooks/17b_acierto_y_backtest.ipynb` | 93 / 67 |
| `notebooks/17c_robustez_direccion.ipynb` | 51 / 49 |
| `notebooks/17d_volatilidad_riesgo.ipynb` | 37 / 27 |
| `notebooks/18_errores_explicabilidad.ipynb` | 49 / 42 |
| `notebooks/18b_auditoria_datos.ipynb` | 64 / 100 |
| `notebooks/19_20_robustez_etica.ipynb` | 42 / 28 |
| `notebooks/21_despliegue.ipynb` | 18 / 15 |
| `notebooks/22_documentacion.ipynb` | 53 / 27 |
| `notebooks/23_monitorizacion.ipynb` | 58 / 35 |

### 11.3 Código fuente, scripts y pytest (13 ficheros)

| Fichero | + / − |
|---|---:|
| `pytest.ini` | 17 / 1 |
| `scripts/monitor_drift.py` | 83 / 23 |
| `scripts/predict.py` | 59 / 8 |
| `src/api/main.py` | 130 / 40 |
| `src/config.py` | 35 / 10 |
| `src/data/load_data.py` | 70 / 3 |
| `src/data/split.py` | 73 / 1 |
| `src/evaluation/metrics.py` | 77 / 6 |
| `src/features/build_features.py` | 61 / 8 |
| `src/models/classifier.py` | 57 / 22 |
| `src/models/deep_learning.py` | 12 / 22 |
| `src/models/train_model.py` | 4 / 16 |
| `src/utils.py` | 41 / 10 |

### 11.4 Ficheros no rastreados observados en `git status --short`

Estos ficheros no forman parte de `git diff --stat` mientras permanezcan sin añadir a Git, pero son relevantes para la revisión:

| Fichero | Observación |
|---|---|
| `tests/test_audit_fixes.py` | Nueva batería de 43 pruebas de regresión, inspeccionada íntegramente. |
| `src/py.typed` | Marcador PEP 561 nuevo; se declara también como package data en `pyproject.toml` para incluirlo en la distribución. |

---

## Conclusión

La rama `v2` corrige causas reales de sesgo temporal, definición de métricas, robustez de monitorización, exposición operativa, empaquetado y tooling. El avance es significativo, especialmente porque las correcciones se acompañan de pruebas de regresión, comprobación AST de notebooks y mejoras de CI. El criterio de cierre no puede ser “el código cambió”, sino “el pipeline corregido se ejecutó de forma reproducible y sustituyó cada artefacto afectado”. Hasta completar ese ciclo, los resultados históricos señalados en la sección 8 no deben utilizarse para justificar capacidad predictiva, rendimiento financiero ni preparación para producción.

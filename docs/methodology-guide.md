# Guía Metodológica - Proyectos de Machine Learning

Documento de referencia que define el **proceso completo y ordenado** para el
desarrollo de proyectos de machine learning aplicado. No todas las fases
tendrán la misma profundidad en cada caso, pero deben considerarse siempre;
las partes entre paréntesis dependen del tipo de problema: tabular, visión,
NLP, forecasting, recomendación, etc.

**Regla central:** *definir -> auditar -> dividir -> aprender solo con train ->
seleccionar con validación/CV -> comprobar una vez con test -> empaquetar ->
monitorizar*. El test nunca debe influir en el preprocesamiento, la selección
de variables, los hiperparámetros o la elección de modelo.

> Esta guía es la base metodológica del presente proyecto. Cada fase del
> repositorio (`notebooks/`) se corresponde con una o más secciones de este
> documento, garantizando trazabilidad completa entre metodología y ejecución.

# Índice completo

```text
0. Gobierno, contexto y definición del problema
1. Diseño técnico, entorno y reproducibilidad
2. Obtención, licencia y gobierno de datos
3. Ingesta, auditoría y validación de datos
4. EDA y entendimiento del dominio
5. Definición de target, features y disponibilidad temporal
6. Diseño de evaluación, métricas y criterios de éxito
7. Estrategia de partición y protocolo experimental
8. Preprocessing y limpieza segura
9. Feature engineering y representaciones
10. Feature selection y reducción dimensional
11. Baselines
12. Modelado inicial
13. Validación y cross-validation apropiada
14. Hyperparameter tuning y experimentación
15. Model selection y decisión técnica
16. Entrenamiento final y versionado de artefactos
17. Evaluación final bloqueada en test
18. Análisis de errores, explicabilidad e incertidumbre
19. Robustez, generalización y pruebas adversas
20. Ética, equidad, privacidad, seguridad y cumplimiento
21. Empaquetado, inferencia y despliegue
22. Documentación, comunicación y entrega
23. Monitorización, mantenimiento y reentrenamiento
```
* Pueden estar agrupadas en notebooks pero deben ralizarse todas y cada una de ellas e incluso más si es conveniente.


# 0\. Problema y contexto

## 0.1 Contexto

* Problema de negocio, investigación o producto
* Usuario final o equipo que utilizará el sistema
* Proceso actual sin automatización
* Decisión que apoyará el modelo
* Valor esperado, coste y restricciones
* Alternativas no basadas en ML: reglas, SQL, heurísticas o proceso manual



## 0.2 Formulación técnica

* Clasificación binaria, multiclase o multietiqueta
* Regresión
* Ranking
* Recomendación
* Clustering
* Reducción dimensional
* Detección de anomalías
* Forecasting
* Visión: clasificación, detección, segmentación, OCR
* NLP: clasificación, NER, QA, resumen, generación
* RAG y agentes
* Aprendizaje por refuerzo
* Robótica, planificación o control



## 0.3 Definición operativa

* Unidad de predicción: persona, transacción, imagen, vivienda, texto, usuario, ítem, episodio, etc.
* Entrada disponible en el momento real de predecir
* Salida esperada: clase, probabilidad, valor, ranking, acción o respuesta
* Horizonte de predicción
* Frecuencia de inferencia y actualización
* Coste de falsos positivos, falsos negativos y abstención
* Límites de latencia, memoria, coste, privacidad e interpretabilidad

\---

# 1\. Diseño y reproducibilidad

## 1.1 Estructura del proyecto

* Repositorio Git
* README
* Entorno y dependencias fijadas
* Configuraciones separadas del código
* Carpetas de datos crudos, procesados, modelos, notebooks, código fuente, informes y pruebas
* `.gitignore`
* Gestión segura de secretos y claves



## 1.2 Reproducibilidad

* Semillas aleatorias
* Versiones de Python, librerías y frameworks
* Configuración de CPU/GPU/CUDA cuando aplique
* Fecha, fuente y versión del dataset
* Registro de experimentos
* Versionado de código, datos, modelos y prompts si hay LLMs
* Scripts ejecutables de principio a fin

\---

# 2\. Gobierno de datos

## 2.1 Obtención

* Fuente de datos
* Método de extracción: archivos, API, base de datos, sensores, scraping autorizado
* Periodo temporal y frecuencia
* Integridad de archivos
* Tamaño y coste de almacenamiento/procesamiento



## 2.2 Legalidad y uso

* Licencia
* Atribución
* Restricciones de redistribución
* Uso académico o comercial
* Copyright
* Datos personales, sensibles o confidenciales
* Consentimiento, minimización y retención



## 2.3 Etiquetas y calidad

* Origen del target o etiquetas
* Criterios de anotación
* Ambigüedad de clases
* Ruido de etiqueta
* Sesgo de los anotadores
* Acuerdo entre anotadores, si existe
* Retraso con el que se obtiene la etiqueta real

\---

# 3\. Ingesta y auditoría

## 3.1 Validación de esquema

* Número de filas y columnas
* Nombres y tipos de variables
* Unidades
* Valores permitidos
* Rango esperado
* Cardinalidad
* Fechas, zonas horarias e identificadores
* Rutas de archivos en visión/audio
* Esquema de entrada esperado para inferencia futura



## 3.2 Calidad básica

* Valores faltantes
* Duplicados exactos y semánticos
* Valores imposibles
* Errores de formato
* Columnas constantes o casi constantes
* Categorías inconsistentes
* Imágenes corruptas
* Textos vacíos, dañados o duplicados
* Archivos inexistentes
* Outliers sospechosos



## 3.3 Riesgos iniciales

* IDs que codifican información indebida
* Variables postevento
* Variables derivadas del target
* Muestras repetidas entre train y test
* Datos de futuro
* Contaminación de benchmark
* Información no disponible en producción
* Desbalance extremo
* Representación desigual de grupos

\---

# 4\. EDA y dominio

## 4.1 EDA del target

* Distribución de clases o valores
* Desbalance
* Valores extremos
* Asimetría
* Tendencia temporal
* Frecuencia por grupo, zona o fuente
* Calidad visual/manual de ejemplos



## 4.2 EDA de features

* Estadísticos descriptivos
* Histogramas y boxplots
* Correlaciones
* Relaciones con target
* Relaciones no lineales
* Cardinalidad de categóricas
* Categorías raras
* Nulos y patrones de ausencia
* Redundancia entre variables



## 4.3 EDA específico

* **Visión:** ejemplos por clase, resolución, canales, fondos espurios, duplicados, oclusión
* **NLP:** idioma, longitud, tokens, duplicados, HTML, URLs, PII, contaminación de etiquetas
* **Series temporales:** tendencia, estacionalidad, huecos, autocorrelación, cambios de régimen
* **Recomendación:** dispersión de interacciones, *cold start*, popularidad, recencia
* **Grafos:** nodos, aristas, componentes, grado, atributos y conectividad
* **RL:** observaciones, acciones, rango de recompensas, duración de episodios y seguridad del entorno
* **RAG:** calidad documental, cobertura, duplicados, formatos, metadatos y permisos



## 4.4 Hipótesis

* Qué variables parecen útiles
* Qué transformaciones podrían ayudar
* Qué sesgos o fugas son posibles
* Qué baseline tiene sentido
* Qué tipo de split corresponde
* Qué hipótesis se probarán después

\---

# 5\. Target y variables

## 5.1 Definir el target

* Nombre y significado
* Tipo de tarea
* Método de etiquetado
* Horizonte temporal
* Posible retraso de etiquetas
* Criterio de positividad/negatividad
* Riesgo de *label leakage*



## 5.2 Clasificar variables

* Target
* Features válidas
* Identificadores
* Variables sensibles
* Variables proxy de atributos sensibles
* Variables postevento
* Variables con riesgo de fuga
* Fechas
* Texto, imagen, audio, vídeo, grafos o sensores
* Metadatos
* Variables que no estarán disponibles en producción



## 5.3 Diccionario de datos

Cada variable debe incluir:

* Nombre
* Descripción
* Tipo
* Unidad
* Nulos
* Rango/categorías
* Momento de disponibilidad
* Riesgo de leakage
* Tratamiento previsto: usar, transformar, auditar o eliminar

\---

# 6\. Métricas y éxito

La **selección de métricas debe realizarse antes del entrenamiento**. No se escoge la métrica que hace que el modelo parezca mejor; se escoge la que representa la decisión y el coste real. [^3](https://docs.cloud.google.com/architecture/guidelines-for-developing-high-quality-ml-solutions)

## 6.1 Métrica primaria

* Clasificación balanceada: accuracy, F1
* Clasificación desbalanceada: PR-AUC, recall, precision, F-beta, coste
* Fraude: recall bajo presupuesto de alertas, PR-AUC, coste de errores
* Regresión: MAE, RMSE, R² complementario
* Forecasting: MAE, RMSE, sMAPE por horizonte
* Ranking/recomendación: Recall@k, Precision@k, NDCG@k, MRR
* Detección: mAP e IoU
* Segmentación: IoU, Dice
* Anomalías: PR-AUC, AUROC, Recall@k
* RAG: Recall@k, precisión del contexto, fidelidad, corrección y abstención
* RL: recompensa media, tasa de éxito, estabilidad y violaciones de restricciones



## 6.2 Métricas secundarias

* Matriz de confusión
* Precision, recall y F1 por clase
* Calibración y Brier score
* Latencia
* Memoria
* Coste de inferencia
* Tamaño del modelo
* Robustez
* Equidad por subgrupos
* Consumo de recursos



## 6.3 Umbral de decisión

* Umbral por defecto
* Umbral ajustado por coste
* Umbral ajustado por capacidad de revisión humana
* Curvas PR, ROC y coste
* Calibración antes de interpretar probabilidades
* Política de abstención cuando hay baja confianza



## 6.4 Criterios de aceptación

* Ganancia mínima sobre baseline
* Rendimiento mínimo en métrica primaria
* Límites para métricas secundarias
* Latencia máxima
* Coste máximo
* Nivel mínimo de robustez
* Ausencia de degradación grave en subgrupos
* Reproducibilidad del resultado

\---

# 7\. Partición y protocolo

Esta fase ocurre **antes de cualquier operación que haga `fit`**, como imputar, escalar, hacer PCA, seleccionar features, calcular embeddings o aplicar target encoding. [^1](https://scikit-learn.org/stable/common_pitfalls.html)

## 7.1 Conjuntos

* **Train:** aprende preprocessing y parámetros del modelo
* **Validation:** elige arquitectura, hiperparámetros, umbrales, épocas y modelo
* **Test:** evaluación final bloqueada



## 7.2 Estrategias de split

* División aleatoria simple
* División estratificada
* K-Fold
* Stratified K-Fold
* GroupKFold
* Leave-One-Group-Out
* División temporal
* TimeSeriesSplit
* División geográfica/espacial
* División por usuario, paciente, escena, documento, producto o vídeo
* Nested Cross-Validation, si necesitas una evaluación experimental muy rigurosa



## 7.3 Según el tipo

* **Clasificación desbalanceada:** estratificación
* **Usuarios/pacientes repetidos:** grupos
* **Forecasting:** orden temporal estricto
* **Visión con imágenes de la misma escena/objeto:** separar por escena u objeto
* **Recomendadores:** split temporal o por interacción, evitando futuro
* **NLP:** evitar documentos duplicados o del mismo origen en train y test
* **RL:** semillas separadas, episodios de evaluación y entornos no vistos si aplica



## 7.4 Control experimental

* Semilla guardada
* Índices de particiones guardados
* Test bloqueado
* Protocolo documentado
* No alterar el split por conveniencia tras mirar métricas

\---

# 8\. Preprocessing seguro

Todo preprocessing que aprende estadísticas debe ajustarse **solo con train** y aplicarse a validation/test mediante `transform`, nunca mediante un nuevo `fit`. [^1](https://scikit-learn.org/stable/common_pitfalls.html)

## 8.1 Tabular

* Imputación
* Indicadores de nulos
* Limpieza de formatos
* Escalado
* Normalización
* Transformación logarítmica
* Tratamiento de outliers
* One-Hot Encoding
* Ordinal Encoding con orden real
* Agrupación de categorías raras
* Hashing o encoding de alta cardinalidad
* Target encoding dentro de folds de CV



## 8.2 Visión

* Carga y comprobación de imágenes
* Redimensionado
* Conversión de canales
* Normalización
* Conversión a tensor
* **Data augmentation solo en train:** *flip*, *crop*, rotación, ruido, color jitter, MixUp, CutMix, etc.
* Transformaciones deterministas en validation/test



## 8.3 NLP

* Limpieza mínima trazable
* Tokenización
* Truncado y padding
* Normalización lingüística cuando proceda
* TF-IDF para baselines
* Tokenizador del Transformer
* Eliminación o anonimización de PII, si aplica



## 8.4 Series temporales

* Ordenar por tiempo
* Identificar huecos
* Resampling
* Imputación temporal sin usar futuro
* Escalado basado solo en pasado/train
* Variables de calendario
* Lags
* Ventanas móviles históricas
* Variables exógenas disponibles antes del pronóstico



## 8.5 RAG

* Parsing de PDFs, HTML, DOCX u otras fuentes
* Limpieza de texto
* Extracción de metadatos
* Segmentación o *chunking*
* Deduplicación
* Control de versión documental
* Indexación léxica/vectorial
* Separación entre corpus, conjunto de evaluación y preguntas de prueba

\---

# 9\. Feature engineering

Las ideas nacen durante EDA, pero su implementación debe ser segura respecto al split y al tiempo.

## 9.1 Tabular

* Ratios, diferencias y sumas
* Interacciones
* Bins
* Logaritmos
* Indicadores booleanos
* Frecuencias
* Variables agregadas históricas
* Variables geográficas
* Componentes temporales
* Conocimiento de dominio



## 9.2 Texto

* Longitud
* n-grams
* TF-IDF
* Embeddings
* Entidades
* Metadatos
* Similitud semántica



## 9.3 Imágenes

* Embeddings de modelos preentrenados
* Representaciones latentes
* Características clásicas, cuando sean pertinentes



## 9.4 Series temporales

* Lags
* Rolling mean y rolling std
* Tendencia
* Estacionalidad
* Variables de eventos
* Calendario
* Variables exógenas
* Ventanas de observación



## 9.5 Recomendación

* Recencia
* Frecuencia
* Popularidad
* Historial
* Similitud de ítems
* Contexto temporal
* Embeddings de usuario/ítem



## 9.6 Reglas contra leakage

* No incluir futuro
* No calcular agregados globales con test
* No incluir la observación actual en sus propios agregados
* No calcular estadísticas de target fuera de folds
* No generar lags con orden temporal incorrecto
* No usar información de respuesta posterior a la predicción

\---

# 10\. Feature selection

Se realiza tras preprocessing básico y debe ocurrir **dentro del pipeline y de los folds**, especialmente si emplea la variable objetivo.

## 10.1 Selección por reglas

* Eliminar target
* Eliminar posteventos
* Eliminar IDs
* Eliminar variables no disponibles en producción
* Eliminar variables prohibidas o injustificadas
* Eliminar fugas evidentes



## 10.2 Métodos filtro

* Varianza baja
* Correlación
* Información mutua
* Chi-cuadrado
* ANOVA
* Redundancia
* Frecuencias mínimas



## 10.3 Métodos wrapper

* RFE
* Forward selection
* Backward selection
* Sequential feature selection



## 10.4 Métodos embedded

* Lasso
* Elastic Net
* Árboles
* Boosting
* Regularización
* Importancia de variables



## 10.5 Reducción dimensional

* PCA
* SVD
* ICA
* UMAP/t-SNE, principalmente para exploración
* Autoencoders
* Embeddings

\---

# 11\. Baselines

Siempre deben existir antes de utilizar modelos complejos.

## 11.1 Baselines simples

* Clase mayoritaria
* Media/mediana
* Último valor
* Pronóstico estacional ingenuo
* Regla de negocio
* Popularidad en recomendación
* BM25 en recuperación
* Agente aleatorio o heurístico en RL



## 11.2 Baselines de ML

* Logistic Regression
* Linear/Ridge/Lasso Regression
* Naive Bayes
* Árbol poco profundo
* k-NN
* TF-IDF + Logistic Regression
* CNN pequeña
* Factorización matricial simple
* Q-learning básico

\---

# 12\. Modelado inicial

## 12.1 Secuencia de complejidad

```text
Baseline
-> modelo simple y explicable
-> modelo no lineal
-> ensemble/boosting
-> deep learning
-> transferencia/fine-tuning
```



## 12.2 Familias

* Regresión lineal, logística, Ridge, Lasso
* k-NN, Naive Bayes, SVM
* Árboles, Random Forest, Extra Trees
* XGBoost, LightGBM, CatBoost
* MLP
* CNN, ResNet, EfficientNet, ViT
* RNN, LSTM, GRU, Transformer
* GNN
* Autoencoders, GAN, difusión
* Isolation Forest, One-Class SVM
* ARIMA/SARIMA, ETS, Prophet, modelos con *lags*
* Filtrado colaborativo y factorización
* DQN, PPO, SAC y otros métodos de RL
* RAG: BM25, recuperación densa, híbrida, reranking y LLM



## 12.3 Disciplina experimental

* Una hipótesis por experimento
* Cambios registrados
* Configuración guardada
* Métricas de train y validation
* Tiempo y recursos consumidos
* Semilla y versión de datos
* Notas de interpretación

\---

# 13\. Validación y cross-validation

## 13.1 Validación

Se usa para:

* Elegir modelo
* Elegir arquitectura
* Ajustar learning rate
* Aplicar *early stopping*
* Elegir número de épocas
* Seleccionar transformaciones
* Elegir umbral de clasificación
* Elegir estrategia de *augmentation*
* Seleccionar el sistema RAG o *retriever*



## 13.2 Tipos de CV

* K-Fold
* StratifiedKFold
* GroupKFold
* TimeSeriesSplit
* Nested CV
* Repeated CV
* Validación por *holdout*



## 13.3 Pipeline dentro de CV

En cada fold deben ajustarse únicamente con el subconjunto de train del fold:

* Imputer
* Scaler
* Encoder
* PCA
* Feature selection
* Target encoding
* Modelo

Esto evita que validation contamine el aprendizaje. [^1](https://scikit-learn.org/stable/common_pitfalls.html)

\---

# 14\. Hiperparámetros

## 14.1 Qué ajustar

* Regularización
* Número de árboles
* Profundidad
* Learning rate
* Batch size
* Arquitectura
* Dropout
* Weight decay
* Scheduler
* Número de epochs
* Umbral
* Parámetros de *augmentation*
* Parámetros de recuperación RAG: chunk size, overlap, top-k, modelo de embeddings, reranker
* Parámetros de RL: tasa de exploración, descuento, política, pasos de entrenamiento



## 14.2 Métodos

* Búsqueda manual razonada
* Grid Search
* Random Search
* Bayesian Optimization
* Optuna
* Hyperband
* Early stopping



## 14.3 Restricciones

* Solo validation/CV
* Presupuesto de cómputo
* Registro de cada experimento
* No usar test
* No optimizar una métrica desconectada del objetivo real

\---

# 15\. Selección de modelo

La elección final se basa en validation/CV, no en test.

## 15.1 Criterios

* Métrica primaria
* Métricas secundarias
* Media y desviación entre folds
* Robustez
* Calibración
* Equidad
* Interpretabilidad
* Latencia
* Coste
* Tamaño
* Mantenibilidad
* Facilidad de despliegue
* Riesgo de seguridad



## 15.2 Decisión

No elijas automáticamente el modelo con una décima más de métrica. Elige el mejor equilibrio entre:

```text
Calidad + estabilidad + coste + rapidez
+ explicabilidad + seguridad + mantenibilidad
```



\---

# 16\. Entrenamiento final

## 16.1 Congelar decisiones

* Features
* Preprocessing
* Selección de variables
* Modelo
* Arquitectura
* Hiperparámetros
* Umbral
* Política de abstención
* Configuración de inferencia



## 16.2 Reentrenar

* Con train completo
* O con train + validation, si la fase de selección ya terminó
* Guardando todo el pipeline, no solo los pesos del modelo



## 16.3 Artefactos

* Modelo
* Preprocessor
* Encoders
* Tokenizador
* Lista de features
* Umbral
* Configuración
* Dataset/versionado
* Checkpoint
* Métricas de validación
* Esquema de entrada

\---

# 17\. Test final bloqueado

## 17.1 Protocolo

```text
Decisiones congeladas
-> cargar test intacto
-> aplicar transformaciones ya aprendidas
-> inferir
-> calcular métricas finales
-> documentar
```



## 17.2 Presentación

* Tabla baseline vs. candidatos vs. final
* Métrica primaria
* Métricas secundarias
* Curvas ROC/PR/calibración, si corresponde
* Matriz de confusión
* Residuos, si es regresión
* Curvas de entrenamiento, si es DL
* Métricas por segmento
* Latencia, coste y recursos
* Intervalos de confianza o estabilidad entre semillas, si es relevante



## 17.3 Regla

Si cambias decisiones porque el test dio mal resultado, el test se convierte en validation. Debes crear un nuevo test bloqueado para volver a evaluar honestamente.

\---

# 18\. Análisis final

## 18.1 Análisis de errores

* Falsos positivos y falsos negativos
* Sobreestimaciones e infraestimaciones
* Casos de alta confianza erróneos
* Clases confundidas
* Errores por segmento
* Casos raros
* Ejemplos fuera de distribución
* Etiquetas dudosas
* Fallos de datos frente a fallos del modelo



## 18.2 Explicabilidad

* Coeficientes
* Feature importance
* Permutation importance
* SHAP
* PDP e ICE
* Grad-CAM en visión, con cautela
* Ejemplos similares
* Explicaciones locales



## 18.3 Incertidumbre

* Calibración
* Brier score
* Intervalos de predicción
* Conformal prediction
* Ensambles
* Confianza
* Abstención

\---

# 19\. Robustez y generalización

## 19.1 Pruebas de entrada

* Nulos
* Tipos erróneos
* Categorías desconocidas
* Valores extremos
* Texto vacío
* Imagen corrupta
* Archivos excesivos
* Datos fuera de rango



## 19.2 Pruebas de cambio

* Dataset shift
* Covariate shift
* Concept drift
* Cambios temporales
* Cambios geográficos
* Ruido
* Iluminación, oclusión o compresión en visión
* Variación lingüística en NLP
* Datos fuera de distribución



## 19.3 Pruebas técnicas

* Diferentes semillas
* Submuestras
* Estrés de latencia
* Carga concurrente
* Memoria
* Fallo de dependencias
* Reproducibilidad de entrenamiento
* Compatibilidad entre entrenamiento y servicio

\---

# 20\. Ética, privacidad y seguridad

Estas cuestiones se revisan desde el inicio, no solamente al final.

## 20.1 Ética y equidad

* Grupos afectados
* Representatividad
* Sesgo histórico
* Métricas por subgrupos
* Variables proxy
* Riesgo de discriminación
* Impacto de errores
* Supervisión humana
* Uso indebido
* Limitaciones de automatización



## 20.2 Privacidad

* Datos personales y sensibles
* Minimización
* Anonimización
* Retención
* Consentimiento
* Accesos
* Reidentificación
* Logs sin exposición de datos



## 20.3 Seguridad

* Validación de inputs
* Autenticación y autorización
* Secretos
* Auditoría
* Dependencias
* Data poisoning
* Ataques adversariales
* Model extraction
* Membership inference
* Prompt injection y fuga de contexto en LLM/RAG
* Seguridad de herramientas y APIs usadas por agentes

\---

# 21\. Empaquetado y despliegue

## 21.1 Pipeline de inferencia

```text
Input crudo
-> validación de esquema
-> preprocessing entrenado
-> feature engineering
-> feature selection
-> modelo
-> umbral/postprocesamiento
-> respuesta
```

El sistema de inferencia debe usar exactamente el preprocessing entrenado; recrearlo manualmente de forma “similar” causa divergencias entre entrenamiento y producción. [^4](https://developers.google.com/machine-learning/guides/rules-of-ml)

## 21.2 Formas de entrega

* Script de línea de comandos
* Notebook reproducible
* API REST
* FastAPI
* Streamlit o Gradio
* Batch inference
* Docker
* Cloud
* Aplicación web
* Integración con base de datos o cola de mensajes



## 21.3 Validación de entradas

* Esquema
* Tipos
* Rangos
* Campos obligatorios
* Categorías
* Tamaños de archivo
* Mensajes de error
* Manejo de datos desconocidos
* Política de abstención o fallback

\---

# 22\. Documentación y comunicación

## 22.1 README

* Problema
* Dataset y licencia
* Instalación
* Ejecución
* Estructura del proyecto
* Resultados
* Métricas
* Limitaciones
* Demo
* Referencias



## 22.2 Informe técnico

* Formulación
* Datos
* EDA
* Split
* Preprocessing
* Feature engineering
* Baselines
* Modelos
* Validación
* Métricas
* Error analysis
* Riesgos
* Decisión final



## 22.3 Model card / data card

* Uso previsto
* Uso no previsto
* Datos
* Métricas
* Subgrupos
* Limitaciones
* Riesgos
* Consideraciones éticas
* Mantenimiento
* Versiones

\---

# 23\. Monitorización y mantenimiento

## 23.1 Monitorización técnica

* Disponibilidad
* Latencia
* Throughput
* Errores
* CPU/GPU
* Memoria
* Coste
* Colas



## 23.2 Monitorización de datos

* Nulos
* Categorías nuevas
* Rangos anómalos
* Drift de features
* Drift de embeddings
* Drift de predicciones
* Inputs fuera de distribución



## 23.3 Monitorización de rendimiento

Cuando lleguen etiquetas reales:

* Accuracy, F1, PR-AUC
* MAE/RMSE
* Recall de fraude
* NDCG/Recall@k
* Calibración
* Métricas por grupo
* Degradación temporal
* Tasa de abstención



## 23.4 Reentrenamiento

* Criterios para reentrenar
* Nuevos datos mínimos
* Nueva validación
* Comparación contra modelo actual
* Aprobación
* Versionado
* Rollback
* Registro de cambios



# Adaptaciones por tipo

|Tipo de proyecto|Etapas que requieren especial atención|
|-|-|
|Supervisado tabular|Pipeline, feature engineering/selection, CV, calibración, explicabilidad|
|No supervisado|Definición del objetivo exploratorio, escalado, elección de distancia, validación cualitativa, interpretación de clusters|
|Anomalías|Definición de normalidad, desbalance, umbral, coste de alertas, drift|
|Visión|Auditoría de imágenes, split por escena/objeto, augmentation solo train, transferencia, análisis visual|
|NLP|Duplicados, idioma, tokenización, longitud, contaminación entre train/test, evaluación humana cuando sea necesaria|
|Series temporales|Horizonte, orden temporal, lags sin futuro, validación temporal, baseline *naive*, drift|
|Recomendación|Split temporal, *cold start*, ranking, sesgo de popularidad, diversidad y métricas @k|
|RAG/LLM|Calidad documental, chunking, retrieval, reranking, citación, fidelidad, seguridad y evaluación separada de recuperación/generación|
|RL|Entorno, estado, acción, recompensa, exploración, semillas, seguridad, evaluación por episodios|
|Robótica|Sensores, simulación, control, seguridad, *sim-to-real*, restricciones físicas|

# Orden mínimo que debes recordar

```text
1. Problema
2. Datos y gobierno
3. Auditoría
4. EDA
5. Target/features
6. Métricas
7. Split
8. Preprocessing
9. Feature engineering/selection
10. Baseline
11. Modelado
12. CV + tuning
13. Selección
14. Entrenamiento final
15. Test final
16. Error analysis
17. Robustez, ética y seguridad
18. Despliegue
19. Documentación
20. Monitorización
```

La idea definitiva es: **el EDA genera hipótesis; train aprende; validation y CV eligen; test verifica; análisis explica; despliegue operacionaliza; monitorización mantiene la validez**.

<div align="center">⁂</div>


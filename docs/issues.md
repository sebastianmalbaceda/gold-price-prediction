# Guía de Solución de Problemas y Reporte de Errores

Este documento tiene como objetivo centralizar la solución a los problemas técnicos más comunes encontrados durante la ejecución del proyecto y establecer el protocolo para reportar nuevos errores en el repositorio.

---

## 1. Protocolo para Abrir una Incidencia (Issue)

Antes de abrir una nueva incidencia, revise la lista de errores comunes en la sección 2. Si el problema persiste, debe abrir un "Issue" proporcionando obligatoriamente la siguiente información:

### Plantilla de Reporte

**Descripción del Error**  
Describa de forma técnica y concisa qué está fallando.

**Pasos para Reproducir**  
Enumere los pasos exactos para replicar el fallo:
- Script o Notebook ejecutado.
- Configuración utilizada (ej. uso de GPU o CPU).
- Momento exacto del fallo (carga de datos, entrenamiento, visualización).

**Traceback / Logs**  
Copie y pegue la salida completa del error en la consola. No adjunte capturas de pantalla de código, use texto.

**Entorno de Ejecución**
- Sistema Operativo:
- Versión de Python:
- Versión de las librerías principales (pandas, catboost):

---

## 2. Solución de Problemas y Preguntas Frecuentes (FAQ)

A continuación se detallan las soluciones a errores técnicos recurrentes, divididos por categorías.

---

### A. Problemas con Fuentes de Datos (FRED y Investing.com)

**P:** Error "ValueError: Ticker not found" o fallo al descargar un activo específico de FRED.  
**R:** La Reserva Federal (FRED) ocasionalmente discontinúa series temporales antiguas, cambia sus códigos de identificación (Tickers) o deja de actualizarlos. Si el código solicita un ID que ya no existe, la API devolverá un error.  
**Solución:** Verifique manualmente en la web fred.stlouisfed.org si el código del activo sigue vigente. Si ha cambiado, actualice el diccionario de configuración en el notebook. Si ha sido eliminado, deberá excluir esa variable del análisis.

**P:** Error 403 Forbidden o 429 Too Many Requests al usar la API.  
**R:** Esto ocurre cuando se excede el límite de peticiones permitidas por la API Key gratuita o si la clave no está configurada correctamente.  
**Solución:** El sistema está diseñado para fallar de manera segura ("fail-safe"). Si la API bloquea la conexión, asegúrese de que los archivos .csv en la carpeta data/ están presentes. El código cargará automáticamente estos archivos locales si la conexión remota falla.

**P:** Discrepancias en los datos de Investing.com.  
**R:** Los datos provenientes de Investing.com a menudo se descargan manualmente y pueden contener caracteres especiales en los nombres de las columnas o formatos de fecha no estándar dependiendo de la configuración regional del navegador al momento de la descarga.  
**Solución:** El script de limpieza contiene funciones específicas para normalizar estos formatos. Si descarga nuevos datos, asegúrese de mantener el formato MM/DD/YYYY o DD/MM/YYYY consistente.

**P:** Valores vacíos (NaN) tras la carga de datos.  
**R:** Es un comportamiento esperado. Diferentes mercados (Forex, Commodities, Acciones) tienen calendarios de festivos distintos. Al alinear las fechas en un solo DataFrame, se generan huecos.  
**Solución:** No intente eliminar estas filas manualmente. El pipeline utiliza el método Forward Fill (rellenar con el último valor conocido) para mantener la continuidad de la serie temporal sin introducir sesgos de "mirar al futuro" (look-ahead bias).

---

### B. Problemas de Instalación y Librerías

**P:** Error "ModuleNotFoundError: No module named..."  
**R:** Indica que el entorno virtual no tiene instaladas todas las dependencias.  
**Solución:** Ejecute `pip install -r requirements.txt` en la raíz del proyecto.

**P:** Conflictos de versión con CatBoost o XGBoost en macOS (Apple Silicon M1/M2/M3).  
**R:** Las arquitecturas ARM pueden presentar problemas con versiones antiguas de bibliotecas de Gradient Boosting compiladas para x64.  
**Solución:** Instale cmake (`brew install cmake`) y fuerce la reinstalación de la librería compilando desde el código fuente o actualice a la última versión disponible: `pip install catboost --upgrade`.

**P:** Error al visualizar árboles de decisión (Graphviz Executable Not Found).  
**R:** La librería de Python graphviz es solo una interfaz. Necesita tener instalado el software binario Graphviz en su sistema operativo y agregado al PATH del sistema.

---

### C. Problemas de Hardware y Rendimiento

**P:** Error "CUDA error: no kernel image is available" o fallos de GPU.  
**R:** Ocurre cuando se intenta entrenar con aceleración GPU pero los drivers de NVIDIA son incompatibles con la versión de CUDA que utiliza CatBoost/XGBoost, o la tarjeta gráfica no tiene capacidad de cómputo suficiente.  
**Solución:** Cambie el parámetro task_type de "GPU" a "CPU" en la configuración del modelo dentro del notebook. El entrenamiento será más lento pero funcionará en cualquier máquina.

**P:** El notebook consume toda la memoria RAM y se cierra (Kernel Crash).  
**R:** El dataset consolidado con todas las variables generadas (lags, medias móviles) puede aumentar significativamente el uso de memoria.  
**Solución:** Si dispone de menos de 8GB de RAM, reduzca el horizonte temporal de entrenamiento (ej. inicie en 2010 en lugar de 2000) o elimine las variables con menor importancia (Feature Importance) en la etapa de preprocesamiento.

---

### D. Dudas Metodológicas y Resultados

**P:** ¿Por qué cambian ligeramente los resultados de precisión (Accuracy) entre ejecuciones?  
**R:** Aunque se ha fijado una semilla aleatoria (random_state) para la reproducibilidad, ciertas operaciones de optimización en GPU y paralelización en CPU no son deterministas al 100%. Las variaciones deberían ser marginales (< 0.5%).

**P:** El Feature Importance muestra variables distintas a las documentadas.  
**R:** La importancia de las variables (SHAP values) es dinámica y depende del periodo de entrenamiento. Si modifica el rango de fechas o el número de iteraciones del modelo, el peso relativo de las variables macroeconómicas puede cambiar.

**P:** El modelo predice siempre la misma clase (ej. siempre "Subida").  
**R:** Esto suele indicar un problema de desbalanceo de clases severo o un sobreajuste extremo. Verifique que no ha eliminado el parámetro class_weights o auto_class_weights en la configuración del modelo CatBoost.

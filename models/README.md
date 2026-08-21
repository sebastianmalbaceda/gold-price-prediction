# Artefactos de modelos

Los artefactos serializados se generan localmente desde los notebooks de
entrenamiento y no se versionan en Git por su tamaño y por la necesidad de
reproducirlos con una version concreta de los datos y dependencias.

Para preparar el flujo completo:

```bash
python -m src.data.load_data
python -m src.features.build_features
bash scripts/run_notebooks.sh 08_10_preprocessing_features
bash scripts/run_notebooks.sh 14_tuning
bash scripts/run_notebooks.sh 15_16_seleccion_entrenamiento
bash scripts/run_notebooks.sh 16_direccion_clasificacion
bash scripts/run_notebooks.sh 16c_deep_learning_clasificacion
```

La API necesita `final_model.joblib`, `preprocessor.joblib` y sus listas de
features. El endpoint `/ready` devuelve `503` hasta que los artefactos de
regresion y clasificacion estan disponibles. Los pesos neuronales `.pt` se
guardan en CPU para poder reutilizarlos en equipos sin CUDA.

Advertencia: los ficheros joblib y PyTorch son serializaciones de Python y solo
deben cargarse si proceden de una fuente confiable.

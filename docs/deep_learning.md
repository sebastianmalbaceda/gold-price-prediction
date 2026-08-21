# Extension de deep learning

## Alcance

La extension de deep learning amplia la fase de clasificacion de direccion
sin sustituir los modelos clasicos ni modificar el test bloqueado. El notebook
principal es `notebooks/16c_deep_learning_clasificacion.ipynb`.

Se comparan:

- MLP tabular sobre las 83 features del modelo estable.
- GRU causal sobre ventanas de 21 dias.

La etiqueta es `dir_1(t) = 1` si `gold_spot(t+1) > gold_spot(t)`.

## Hardware y fallback

El modulo `src/models/deep_learning.py` expone `get_device()` y selecciona:

1. CUDA si `torch.cuda.is_available()` es verdadero.
2. CPU en caso contrario.

La ejecucion queda registrada mediante `device_report()`. En la ejecucion de
referencia se utilizo una NVIDIA GeForce RTX 3050 Ti Laptop GPU con CUDA 12.8.

Comprobacion rapida:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Instalacion general del proyecto:

```bash
pip install -r requirements.txt
```

Para una build CUDA especifica de PyTorch, consulte el indice oficial de
PyTorch y seleccione la version compatible con su controlador. La deteccion
de hardware se realiza en tiempo de ejecucion y el sistema usa CPU si CUDA
no esta disponible.

## Control de leakage y sobreajuste

- El `RobustScaler` se ajusta solo con train.
- Las ventanas GRU terminan en t y nunca incluyen t+1.
- El early stopping utiliza exclusivamente validation.
- Se aplican dropout, weight decay y clipping de gradiente.
- El test se consulta despues de congelar arquitectura e hiperparametros.
- Se guardan semilla, dispositivo, epoch optimo y metadatos.

## Resultados de referencia

| Modelo | Train AUC | Validation AUC | Test AUC | Dispositivo |
|---|---:|---:|---:|---|
| MLP | 0.649 | 0.550 | 0.534 | CUDA |
| GRU | 0.569 | 0.543 | 0.510 | CUDA |

El MLP es el mejor modelo DL por validation, pero no supera al RandomForest
regularizado de la fase 17c. Por ello se conserva como experimento reproducible
y no se reemplaza el modelo operativo.

## Regeneracion

```bash
make train-neural
# Alternativa sin make:
bash scripts/run_notebooks.sh 16c_deep_learning_clasificacion
```

Los pesos `.pt` y los metadatos generados se mantienen fuera de Git mediante
`.gitignore`. Se pueden regenerar siempre desde el notebook y el dataset.

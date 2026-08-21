# Informe de auditoria tecnica y de seguridad

## Alcance

Se revisaron el codigo fuente, configuracion, dependencias, scripts, API,
Docker, CI, tests, notebooks, datos derivados, reports y documentacion de la
rama `v2`. La auditoria se realizo despues de regenerar el pipeline completo
con el dataset disponible y de ejecutar los notebooks desde procesos limpios.

## Hallazgos corregidos

### Datos, features y leakage

1. **Exclusiones de cobertura ignoradas.** La lista `excluded_features` estaba
   bajo `features` en YAML, pero el cargador solo la buscaba bajo `data`.
   Ahora se combinan ambas ubicaciones por compatibilidad. La limpieza pasa de
   42 a 27 columnas y el conjunto actual conserva 25 exogenas despues del
   filtro de cobertura.
2. **Forward-fill ilimitado del target.** Un `ffill()` posterior anulaba el
   limite de tres dias para `gold_spot`. Ahora las exogenas se rellenan por
   separado y el target conserva el limite; una ausencia larga no se convierte
   en precio artificial.
3. **Warm-up incorrecto con columnas completamente nulas.** `idxmax()` no
   detectaba columnas all-null. `drop_warmup` ahora rechaza esas columnas,
   valida el warm-up y calcula posiciones validas de forma segura.
4. **Posible leakage accidental de targets.** `get_feature_columns` ahora
   excluye cualquier columna con prefijo `target_`, incluso si el llamador
   solicita un unico horizonte.
5. **Validaciones temporales.** Se validan fechas, orden, duplicados, rangos
   no solapados y splits no vacios. Las ventanas GRU rechazan fechas ausentes,
   historia insuficiente, NaN e infinitos en vez de descartar observaciones en
   silencio.
6. **Valores no validos para logaritmos.** Feature engineering rechaza precios
   no positivos o no finitos y solo calcula retornos log de exogenas positivas.

### Inferencia y CLI

1. **Prediccion CLI desplazada.** `predict_for_date` llamaba a `make_targets`,
   que elimina las ultimas `h` filas aun cuando el target futuro no existe en
   inferencia. Ahora construye features hasta la fecha y selecciona exactamente
   esa fecha; tambien rechaza fines de semana y fechas sin cotizacion.
2. **Validacion de metricas.** Se añadieron comprobaciones de longitudes,
   finitud, clases binarias, probabilidades en `[0,1]`, umbrales y horizontes.
   El MAPE de una serie completamente cero devuelve `nan` explicitamente sin
   emitir warnings de NumPy.
3. **Artefactos atomicos.** Los modelos clasicos, clasificadores, modelos
   neuronales y JSON se escriben en temporales y se reemplazan atomicamente.
   Los pesos `.pt` se convierten a CPU al guardarse para que sean portables.
4. **Compatibilidad de artefactos.** La API verifica que modelo,
   preprocesador y lista de features tengan la misma dimensionalidad antes de
   declarar readiness.

### Deep learning integrado

1. PyTorch se incorporo a `requirements.txt` y a las dependencias normales de
   `pyproject.toml`; se eliminaron `requirements-dl*.txt` y el extra `.[dl]`.
2. La MLP y la GRU usan CUDA automaticamente cuando esta disponible y CPU como
   fallback. Se registran dispositivo, version, GPU, semilla y epoch.
3. Se mantienen preprocessing solo con train, ventanas causales, early
   stopping por validation, dropout, AdamW, clipping y weight decay.
4. El notebook ahora toma hiperparametros desde `configs/config.yaml`, valida
   la correspondencia de fechas entre MLP/GRU y usa historia previa suficiente
   para las primeras ventanas sin incluir observaciones futuras.

### Notebooks y automatizacion

1. Se corrigio un import ausente en el notebook de despliegue.
2. Se corrigio la auditoria 18b para pandas moderno (`groupby(...)[cols]`) y se
   eliminaron conclusiones y graficos con metricas hardcodeadas obsoletas.
3. El runner ya no depende de `python -m jupyter nbconvert`, que fallaba en el
   entorno actual; usa `python -m nbconvert`, detecta venv Windows/Unix,
   valida nombres, limpia temporales con `trap` y no deja outputs parciales.
4. Se añadieron tests que validan estructura, compilacion y ausencia de errores
   persistidos en todos los notebooks.

### API, seguridad y despliegue

1. Fechas ISO estrictas, claves top-level desconocidas prohibidas, limite de
   cantidad de features, NaN/Inf y rangos fisicamente absurdos rechazados.
2. Se separaron `/health` (liveness) y `/ready` (readiness). Los errores de
   artefactos no exponen rutas internas ni trazas al cliente.
3. Se documenta que joblib y PyTorch son deserializaciones de Python y solo se
   deben cargar artefactos confiables.
4. La imagen Docker ejecuta como usuario no root; Compose usa filesystem de
   solo lectura, `tmpfs`, `cap_drop: ALL` y `no-new-privileges`.
5. Se añadió `.dockerignore` para no enviar datos, notebooks, tests y entornos
   locales al contexto de build.
6. CI limita permisos a lectura, ejecuta `pip check`, tests, compilacion,
   flake8 y Black. Las restricciones de Starlette/Pydantic/httpx evitan la
   advertencia de compatibilidad de `TestClient`.

## Estado reproducido

- Dataset limpio: 6,705 observaciones y 27 columnas.
- Features procesadas: 133 columnas; 83 seleccionadas.
- Regresion Ridge: validation MAE 60.44; test MAE 99.75, RMSE 125.08,
  R2 0.9352.
- Clasificador calibrado desplegado: test AUC 0.532, PR-AUC 0.576,
  accuracy 0.539, Brier 0.247.
- MLP: train/validation/test AUC 0.649/0.550/0.534.
- GRU: train/validation/test AUC 0.569/0.543/0.510.
- Dispositivo DL observado: NVIDIA GeForce RTX 3050 Ti Laptop GPU, CUDA 12.8.
- Todos los notebooks del runner se ejecutaron sin errores persistidos.
- 37 tests pasan; flake8, Black, compileall y `pip check` pasan.
- La API local responde correctamente en `/health`, `/ready`, `/predict` y
  `/predict_direction` con artefactos compatibles.

## Riesgos residuales aceptados y documentados

- Las fechas de publicacion de algunas series macro del dataset pueden estar
  adelantadas respecto a la disponibilidad historica real. El impacto actual
  es 0.000 AUC porque esas variables se excluyen, pero la correccion formal
  `PUBLICATION_LAG` queda como medida necesaria para incorporar macro en el
  futuro.
- Los artefactos de modelos no se versionan en Git; deben regenerarse con el
  dataset y las dependencias de `requirements.txt`. `/ready` permite detectar
  el caso de forma explicita.
- La API academica no incorpora autenticacion, rate limiting ni TLS. Debe
  situarse detras de un proxy seguro antes de cualquier exposicion externa.
- La imagen general instala PyTorch junto con el resto del proyecto. Para
  produccion conviene generar un lockfile y seleccionar la wheel CUDA/CPU
  compatible con el hardware, sin volver a separar la extension del flujo
  metodologico.

La rama `main` no se modifica durante esta auditoria.

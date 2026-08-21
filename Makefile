# ============================================================
# Makefile - Gold Price Prediction
# ============================================================
ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
PIP ?= .venv/Scripts/pip.exe
else
PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
endif

.PHONY: help setup data features train train-neural train-dl test api notebooks run-notebooks clean lint

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Crea el virtualenv e instala todas las dependencias del proyecto
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data: ## Regenera data/interim y data/processed desde el crudo
	$(PYTHON) -m src.data.load_data
	$(PYTHON) -m src.features.build_features

features: data ## Alias para regenerar datos y features

train: ## Explica el flujo completo de entrenamiento de modelos
	@echo "El flujo usa notebooks/15_16_seleccion_entrenamiento.ipynb, 16_direccion_clasificacion.ipynb y 16c_deep_learning_clasificacion.ipynb"
	@echo "Todas las dependencias, incluido PyTorch, se instalan desde requirements.txt"

train-neural: ## Ejecuta la fase neuronal del flujo integrado
	mkdir -p .nb_tmp
	$(PYTHON) -m nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=1800 \
		--output-dir=.nb_tmp --output=16c_deep_learning_clasificacion_exec.ipynb \
		notebooks/16c_deep_learning_clasificacion.ipynb
	mv .nb_tmp/16c_deep_learning_clasificacion_exec.ipynb notebooks/16c_deep_learning_clasificacion.ipynb

train-dl: train-neural ## Alias de compatibilidad para el objetivo train-neural

api: ## Arranca la API REST en modo desarrollo
	$(PYTHON) -m uvicorn src.api.main:app --reload --host "$${API_HOST:-127.0.0.1}" --port "$${API_PORT:-8000}"

test: ## Ejecuta los tests
	$(PYTHON) -m pytest tests/ -q

notebooks: run-notebooks ## Alias para ejecutar todos los notebooks

run-notebooks: ## Ejecuta todos los notebooks y persiste outputs
	bash scripts/run_notebooks.sh

clean: ## Limpia artefactos generados no versionados
	rm -rf .pytest_cache .nb_tmp .nb_tmp_dl.ipynb catboost_info
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

lint: ## Verifica sintaxis de Python
	$(PYTHON) -m compileall -q src scripts tests
	@echo "Sintaxis OK"

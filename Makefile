# ============================================================
# Makefile - Gold Price Prediction
# ============================================================
PYTHON ?= .venv/Scripts/python.exe
PIP ?= .venv/Scripts/pip.exe

.PHONY: help setup data features train test api notebooks clean lint

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Crea el virtualenv e instala dependencias
	python -m venv .venv
	$(PIP) install -r requirements.txt

data: ## Regenera data/interim y data/processed desde el crudo
	$(PYTHON) -m src.data.load_data
	$(PYTHON) -m src.features.build_features

train: ## Entrena el modelo final (requiere features + feature_list.json)
	$(PYTHON) -X utf8 -c "from src.models.train_model import save_model_artifacts; print('El entrenamiento se ejecuta desde los notebooks 15-16 (ver README)')"

api: ## Arranca la API REST
	$(PYTHON) -m uvicorn src.api.main:app --reload

test: ## Ejecuta los tests
	$(PYTHON) -m pytest tests/ -q

notebooks: ## (deprecado: los notebooks son autocontenidos; ejecutar con run-notebooks)

run-notebooks: ## Ejecuta todos los notebooks (persiste outputs)
	bash scripts/run_notebooks.sh

clean: ## Limpia artefactos generados
	rm -rf .pytest_cache .nb_tmp catboost_info
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

lint: ## Verifica sintaxis de Python
	$(PYTHON) -m compileall -q src scripts tests
	@echo "Sintaxis OK"

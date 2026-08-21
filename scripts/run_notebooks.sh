#!/usr/bin/env bash
# Ejecuta notebooks de forma reproducible y persiste sus outputs.
# Uso: bash scripts/run_notebooks.sh [notebook_sin_extension]
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
    PY=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
else
    PY="python"
fi

if ! command -v "$PY" >/dev/null 2>&1 && [[ ! -x "$PY" ]]; then
    echo "No se encontro el interprete Python: $PY" >&2
    exit 1
fi

TMP=".nb_tmp"
mkdir -p "$TMP"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

run_one() {
    local nb="$1"
    if [[ ! "$nb" =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo "Nombre de notebook invalido: $nb" >&2
        exit 2
    fi
    if [[ ! -f "notebooks/$nb.ipynb" ]]; then
        echo "No existe notebooks/$nb.ipynb" >&2
        exit 2
    fi
    echo "=== $nb ==="
    "$PY" -X utf8 -m nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=1800 \
        --output-dir="$TMP" --output="${nb}_exec.ipynb" \
        "notebooks/$nb.ipynb"
    mv "$TMP/${nb}_exec.ipynb" "notebooks/$nb.ipynb"
    echo "  -> OK (outputs persistidos)"
}

if [ "$#" -ge 1 ]; then
    run_one "$1"
else
    for nb in 01_problema_y_diseno 02_datos_y_auditoria 04_eda \
              05_target_y_metricas 07_particion 08_10_preprocessing_features \
              11_13_baselines_modelado 14_tuning 15_16_seleccion_entrenamiento \
              16_direccion_clasificacion 16c_deep_learning_clasificacion \
              17b_acierto_y_backtest 17c_robustez_direccion \
              17d_volatilidad_riesgo 17_test_final \
              18_errores_explicabilidad 18b_auditoria_datos \
              19_20_robustez_etica 21_despliegue 22_documentacion \
              23_monitorizacion; do
        run_one "$nb"
    done
fi

echo "TODOS LOS NOTEBOOKS EJECUTADOS CON OUTPUTS"

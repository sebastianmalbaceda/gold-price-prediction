#!/usr/bin/env bash
# Ejecuta todos los notebooks de forma reproducible (con outputs persistidos).
# Uso: bash scripts/run_notebooks.sh [notebook_especifico]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
TMP=".nb_tmp"
mkdir -p "$TMP"

run_one() {
    local nb="$1"
    echo "=== $nb ==="
    "$PY" -X utf8 -m jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=1800 \
        "notebooks/$nb.ipynb" --output "../$TMP/${nb}_exec.ipynb"
    mv "$TMP/${nb}_exec.ipynb" "notebooks/$nb.ipynb"
    echo "  -> OK (outputs persistidos)"
}

if [ $# -ge 1 ]; then
    run_one "$1"
else
    for nb in 01_problema_y_diseno 02_datos_y_auditoria 04_eda \
              05_target_y_metricas 07_particion 08_10_preprocessing_features \
              11_13_baselines_modelado 14_tuning 15_16_seleccion_entrenamiento \
              16_direccion_clasificacion 17b_acierto_y_backtest 17_test_final \
              18_errores_explicabilidad \
              19_20_robustez_etica 21_despliegue 22_documentacion \
              23_monitorizacion; do
        run_one "$nb"
    done
fi
rm -rf "$TMP"
echo "TODOS LOS NOTEBOOKS EJECUTADOS CON OUTPUTS"

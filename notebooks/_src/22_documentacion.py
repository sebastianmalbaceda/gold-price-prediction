# %% [markdown]
# # Fase 22: Documentación y comunicación
#
# ## Entregables de documentación
#
# 1. **README.md** → problema, dataset, instalación, ejecución, resultados, métricas, estructura, limitaciones.
# 2. **docs/informe_tecnico.md** → formulación, datos, EDA, split, preprocessing, features, baselines, modelos, validación, error analysis, riesgos, decisión final.
# 3. **docs/model_card.md** → uso previsto/no previsto, datos, métricas, subgrupos, limitaciones, ética, mantenimiento.
# 4. **docs/data_card.md** → diccionario de datos (fase 5.3): nombre, descripción, tipo, unidad, nulos, rango, disponibilidad, riesgo de leakage, tratamiento.
# 5. **docs/api_manual.md** → manual de la API REST.
#
# ---

# %%
"""Configuración del notebook (raíz del proyecto)."""
import sys
from pathlib import Path

def _find_root():
    p = Path.cwd()
    for _ in range(5):
        if (p / "configs" / "config.yaml").exists():
            return p
        p = p.parent
    return Path.cwd()

ROOT = _find_root()
sys.path.insert(0, str(ROOT))

# %%
"""Verificación de la documentación generada.

Comprobamos que existen todos los entregables de documentación y que no
están vacíos. Si falta alguno, hay que crearlo.
"""
docs = ["README.md",
        "docs/informe_tecnico.md",
        "docs/model_card.md",
        "docs/data_card.md",
        "docs/api_manual.md"]

for d in docs:
    p = Path(ROOT) / d
    size = p.stat().st_size if p.exists() else 0
    print(f"{'OK ' if p.exists() and size > 0 else 'FALTA '} {d} ({size} bytes)")

# Artefactos del clasificador de dirección (fase 16b)
for a in ["direction_classifier.joblib", "direction_preprocessor.joblib",
          "direction_feature_list.json", "direction_metrics.json"]:
    p = Path(ROOT) / "models" / a
    size = p.stat().st_size if p.exists() else 0
    print(f"{'OK ' if p.exists() and size > 0 else 'FALTA '} models/{a} ({size} bytes)")

# %%
"""Resumen de resultados finales registrados.

Leemos `reports/test_results.json` (fase 17) para mostrar las métricas
finales que alimentan el README y el informe técnico.
"""
import json

try:
    with open(ROOT / "reports" / "test_results.json") as f:
        tr = json.load(f)
    print("Resultados finales en test:")
    print(json.dumps(tr["final_test"], indent=2))
    print("\nGanancia vs naive: {:.1f}%".format(tr["improvement_vs_naive_pct"]))
except FileNotFoundError:
    print("\n(reports/test_results.json aún no existe: ejecutar fases 1-17)")

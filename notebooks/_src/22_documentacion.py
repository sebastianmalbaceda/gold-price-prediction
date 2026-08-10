# %% [markdown]
# # Fase 22: Documentación y comunicación
#
# ## Entregables de documentación
# 1. **README.md** → problema, dataset, instalación, ejecución, resultados, métricas, estructura, limitaciones.
# 2. **docs/informe_tecnico.md** → formulación, datos, EDA, split, preprocessing, features, baselines, modelos, validación, error analysis, riesgos, decisión final.
# 3. **docs/model_card.md** → uso previsto/no previsto, datos, métricas, subgrupos, limitaciones, ética, mantenimiento.
# 4. **docs/data_card.md** → diccionario de datos (fase 5.3): nombre, descripción, tipo, unidad, nulos, rango, disponibilidad, riesgo de leakage, tratamiento.
#
# ---

# %%
"""Bootstrap del path del proyecto (funciona desde cualquier cwd)."""
import sys
from pathlib import Path

def _find_root():
    p = Path.cwd()
    for _ in range(5):
        if (p / "configs" / "config.yaml").exists():
            return p
        p = p.parent
    return Path.cwd()

sys.path.insert(0, str(_find_root()))

"""Fase 22: verificación de la documentación generada."""
from pathlib import Path

docs = ["README.md",
        "docs/informe_tecnico.md",
        "docs/model_card.md",
        "docs/data_card.md",
        "docs/api_manual.md"]

for d in docs:
    p = Path(d)
    print(f"{'OK ' if p.exists() else 'FALTA '} {d} ({p.stat().st_size if p.exists() else 0} bytes)")

# Resumen de resultados finales registrados
import json
try:
    with open("reports/test_results.json") as f:
        tr = json.load(f)
    print("\nResultados finales en test:")
    print(json.dumps(tr["final_test"], indent=2))
except FileNotFoundError:
    print("\n(reports/test_results.json aún no existe: ejecutar fases 1-17)")

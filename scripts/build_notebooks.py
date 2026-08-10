"""Construye notebooks .ipynb a partir de fuentes .py con marcadores de celda.

Formato de fuente (notebooks/_src/*.py):
    # %% [markdown]     -> inicia celda markdown
    # %%                -> inicia celda de código
    # %% [markdown] skip -> celda markdown que se omite (notas de trabajo)

Uso:  python scripts/build_notebooks.py [--all|nombre_base]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

SRC_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "_src"
OUT_DIR = Path(__file__).resolve().parents[1] / "notebooks"

MD = "# %% [markdown]"
CODE = "# %%"


def parse_source(path: Path) -> list[tuple[str, str]]:
    """Devuelve [(tipo, contenido)] con tipo in {markdown, code}."""
    lines = path.read_text(encoding="utf-8").splitlines()
    cells: list[tuple[str, list[str]]] = []
    current_type, current = None, []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped == MD or stripped.startswith(MD + " "):
            if current_type:
                cells.append((current_type, current))
            skip = stripped.endswith("skip")
            current_type, current = "markdown", []
            continue
        if stripped == CODE:
            if current_type:
                cells.append((current_type, current))
            current_type, current = "code", []
            skip = False
            continue
        if current_type is None:
            continue  # cabecera fuera de celdas
        if not skip:
            # Las celdas de código en la fuente son Python real (los
            # docstrings no van comentados); se copian tal cual.
            current.append(line)
    if current_type:
        cells.append((current_type, current))
    return [(t, "\n".join(c).strip() + "\n") for t, c in cells if c]


def build(path: Path, out: Path) -> None:
    cells = parse_source(path)
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    for kind, content in cells:
        if kind == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        else:
            nb.cells.append(nbf.v4.new_code_cell(content))
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"[nb] {out.name}: {len(cells)} celdas")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("name", nargs="?", default=None)
    args = ap.parse_args()

    if args.name:
        build(SRC_DIR / args.name, OUT_DIR / args.name.replace(".py", ".ipynb"))
    else:
        for src in sorted(SRC_DIR.glob("*.py")):
            build(src, OUT_DIR / src.name.replace(".py", ".ipynb"))


if __name__ == "__main__":
    main()

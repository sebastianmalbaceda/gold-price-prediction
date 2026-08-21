"""Validaciones estructurales de notebooks sin ejecutar el pipeline completo."""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_all_notebooks_are_valid_and_compile():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks, "No se encontraron notebooks"
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(cell.source, f"{path}:{index}", "exec")
            assert not any(
                output.get("output_type") == "error" for output in cell.get("outputs", [])
            ), f"{path} contiene un error persistido en la celda {index}"

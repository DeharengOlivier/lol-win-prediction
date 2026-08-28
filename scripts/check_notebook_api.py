#!/usr/bin/env python
"""Fail if the notebook calls anything the package no longer exposes.

The notebook is a committed deliverable that needs the 42 MB dataset to run, so
CI cannot execute it. What CI can do cheaply is import every name the notebook
imports and confirm it still exists, which catches the failure that actually
happens: a rename in the package leaving the notebook calling a ghost.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "lol-win-prediction.ipynb"


def imported_names(notebook: Path) -> list[tuple[str, str]]:
    """Return every ``(module, name)`` the notebook imports from ``lolwin``."""
    cells = json.loads(notebook.read_text())["cells"]
    sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    found: list[tuple[str, str]] = []
    for source in sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:  # a cell using IPython magics is not our concern
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "lolwin"
            ):
                found.extend((node.module or "", alias.name) for alias in node.names)
    return found


def main() -> int:
    missing = []
    for module_name, name in imported_names(NOTEBOOK):
        module = importlib.import_module(module_name)
        if not hasattr(module, name):
            missing.append(f"{module_name}.{name}")

    if missing:
        print(f"The notebook imports names that no longer exist: {missing}")
        return 1
    print(f"{NOTEBOOK.name}: every imported name still exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

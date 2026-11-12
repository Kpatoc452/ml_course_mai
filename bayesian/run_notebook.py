"""Выполнить лабораторную в новом ядре и сохранить выводы после успешного запуска."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    base = Path(__file__).resolve().parent
    path = base / "lab04_bayesian_network.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(base)}},
    ).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Выполнено и сохранено: {path}")


if __name__ == "__main__":
    main()

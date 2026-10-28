"""Carpetas del proyecto. Las rutas de la configuración se interpretan respecto de ``PROJECT_ROOT``."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
REFERENCES_DIR = PROJECT_ROOT / "references"
RUNS_DIR = PROJECT_ROOT / "runs"


def resolve(path: str | Path) -> Path:
    """Ruta absoluta: las relativas son respecto de la carpeta del proyecto."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

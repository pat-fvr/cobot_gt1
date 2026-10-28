"""Zona de trabajo de la celda (toma, bandeja, volteo)."""

from __future__ import annotations

from dataclasses import dataclass

from gt1.geometry import FloatArray


@dataclass(frozen=True, eq=False)
class Zone:
    """Polígono en el plano XY de la base, sobre una superficie a ``surface_z_mm``."""

    name: str
    polygon_mm: FloatArray  # (N, 2) vértices en orden
    surface_z_mm: float

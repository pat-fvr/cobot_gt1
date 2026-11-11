"""Bandeja detectada por la cámara."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from gt1.geometry import FloatArray


@dataclass(frozen=True, eq=False)
class TrayDetection:
    """Pose de la bandeja sobre la mesa, en la base del robot.

    ``measurements`` es lo que midió el detector para aceptarla, con la unidad
    en el nombre, para el registro (vacío si la bandeja es fija).
    """

    center_mm: FloatArray  # (3,) centro del fondo
    axis_angle_deg: float  # eje mayor respecto del eje X de la base, en [-90, 90)
    corners_mm: FloatArray  # (4, 3) esquinas del borde superior
    measurements: Mapping[str, float]

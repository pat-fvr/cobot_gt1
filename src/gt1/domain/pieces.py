"""Piezas detectadas y la cara que muestran hacia la cámara."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from gt1.geometry import FloatArray


class Face(StrEnum):
    """Cara de la pechuga que queda hacia arriba."""

    SKIN = "piel"
    BONE = "hueso"
    UNCERTAIN = "incierto"  # el clasificador no está seguro: la pieza no se toma


class PieceStatus(StrEnum):
    """Si la pieza se puede tomar tal como se ve."""

    OK = "ok"
    CLUSTER = "grupo"  # área mayor a la de una pieza: varias piezas juntas o superpuestas
    CUT_BY_ZONE_EDGE = "en_borde"  # toca el borde de la zona: puede estar parcialmente fuera


@dataclass(frozen=True, eq=False)
class Piece:
    """Una pieza vista por la cámara.

    El contorno está en píxeles (para dibujar y para la máscara de la imagen);
    el resto, en mm en la base del robot, a la altura del contorno.

    El eje mayor es la línea central que separa los dos lóbulos cuando el
    detector la mide (``lobes_direction_deg`` no es None); si no, el eje del
    rectángulo que encierra el contorno.
    """

    contour_px: NDArray[np.int32]  # (N, 2)
    center_mm: FloatArray  # (3,) centro del contorno en la base
    axis_angle_deg: float  # ángulo del eje mayor respecto del eje X de la base, en [-90, 90)
    length_mm: float  # largo a lo largo del eje mayor
    width_mm: float  # ancho perpendicular al eje mayor
    area_mm2: float
    status: PieceStatus
    # Sentido del eje mayor desde la punta hacia los lóbulos, en [-180, 180); None si el detector no
    # distingue los extremos.
    lobes_direction_deg: float | None


@dataclass(frozen=True)
class FaceResult:
    """Cara de una pieza y qué tan claro fue el resultado.

    ``margin`` está en las unidades de cada clasificador: positivo = se parece
    más a piel, negativo = a hueso, cerca de 0 = dudoso. ``measurements`` es lo
    que midió el clasificador para decidir, con la unidad en el nombre, para el registro.
    """

    face: Face
    margin: float
    measurements: Mapping[str, float]

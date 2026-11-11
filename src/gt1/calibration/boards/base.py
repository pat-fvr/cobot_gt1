"""Tablero de calibración: interfaz común y las esquinas que se tocan con el pincho.

El sistema del tablero tiene X e Y a lo largo de los cuadros y Z = 0 en la
superficie impresa. Cada implementación decide dónde pone el origen; lo único
que se pide es que, dentro de una misma detección, número de esquina, píxel y
posición en el tablero sean consistentes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.domain.frame import Image
from gt1.geometry import FloatArray


@dataclass(frozen=True, eq=False)
class BoardDetection:
    """Esquinas internas del tablero encontradas en una imagen."""

    corner_ids: NDArray[np.int32]  # (N,) número de cada esquina en el tablero
    pixels_px: FloatArray  # (N, 2) posición en la imagen
    board_points_mm: FloatArray  # (N, 3) posición en el sistema del tablero


class CalibrationBoard(ABC):
    @property
    @abstractmethod
    def corner_count(self) -> int:
        """Esquinas internas del tablero completo."""

    @property
    @abstractmethod
    def thickness_mm(self) -> float:
        """De la mesa a la superficie impresa."""

    @abstractmethod
    def detect(self, image: Image) -> BoardDetection | None:
        """Esquinas visibles, o None si no se ve el tablero."""

    @abstractmethod
    def render(self, pixels_per_mm: float) -> Image:
        """Imagen del tablero para imprimir a escala, con un margen blanco."""


def to_gray(image: Image) -> Image:
    """Los detectores trabajan en gris; las cámaras monocromáticas ya lo entregan así."""
    return image if image.ndim == 2 else np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def touch_corner_ids(detection: BoardDetection) -> NDArray[np.int32]:
    """Esquinas que se tocan con el pincho: las cuatro extremas de lo visto y la más central.

    Se eligen sobre la detección (no sobre el tablero completo) para que siempre
    sean esquinas visibles. El orden es estable: origen, fin de la primera fila,
    inicio de la última fila, esquina opuesta al origen y centro.
    """
    x, y = detection.board_points_mm[:, 0], detection.board_points_mm[:, 1]
    center = detection.board_points_mm[:, :2].mean(axis=0)
    indices = [
        np.argmin(x + y),
        np.argmax(x - y),
        np.argmin(x - y),
        np.argmax(x + y),
        np.argmin(np.linalg.norm(detection.board_points_mm[:, :2] - center, axis=1)),
    ]
    return detection.corner_ids[np.asarray(indices)].astype(np.int32)

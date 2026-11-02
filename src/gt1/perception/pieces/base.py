"""Interfaz de los detectores de piezas."""

from __future__ import annotations

from abc import ABC, abstractmethod

from gt1.domain import Frame, Piece, Zone
from gt1.perception.image_ops import Mask


class PieceDetector(ABC):
    @abstractmethod
    def detect(self, frame: Frame, zone: Zone) -> list[Piece]:
        """Piezas dentro de ``zone``, con su posición y orientación en la base del robot."""

    @abstractmethod
    def foreground_mask(self, frame: Frame, zone: Zone) -> Mask:
        """Lo que el detector considera pieza (0/255), para ajustar sus parámetros mirando la imagen."""

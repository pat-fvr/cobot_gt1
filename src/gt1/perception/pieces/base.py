"""Interfaz de los detectores de piezas."""

from __future__ import annotations

from abc import ABC, abstractmethod

from gt1.domain import Frame, Piece, Zone


class PieceDetector(ABC):
    @abstractmethod
    def detect(self, frame: Frame, zone: Zone) -> list[Piece]:
        """Piezas dentro de ``zone``, con su posición y orientación en la base del robot."""

"""Interfaz de los detectores de bandeja.

Primera implementación prevista: bandeja PET transparente sobre el fondo de
alto contraste, por bordes, validada contra las medidas de ``config/trays/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gt1.domain import Frame, TrayDetection, Zone


class TrayDetector(ABC):
    @abstractmethod
    def detect(self, frame: Frame, zone: Zone) -> TrayDetection | None:
        """La bandeja dentro de ``zone``, o None si no se ve."""

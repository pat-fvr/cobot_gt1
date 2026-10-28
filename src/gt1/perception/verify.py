"""Interfaz del verificador de colocación: cómo quedó la pieza en la bandeja.

Se usa después de cada ciclo, con el robot fuera de la vista. La primera
implementación vendrá con ``pick_place_test``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from gt1.domain import Face, Frame, TrayDetection


@dataclass(frozen=True)
class PlacementResult:
    piece_in_tray: bool
    face: Face  # cara que quedó hacia arriba
    clear_of_seal_flange: bool  # nada sobre el borde de sellado

    @property
    def success(self) -> bool:
        return self.piece_in_tray and self.face == Face.SKIN and self.clear_of_seal_flange


class PlacementVerifier(ABC):
    @abstractmethod
    def verify(self, frame: Frame, tray: TrayDetection) -> PlacementResult: ...

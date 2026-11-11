"""Interfaces de las estrategias de toma, volteo y colocación.

Una estrategia es pura y determinista: recibe lo que vio la cámara y devuelve
un ``Plan`` (lista de pasos), sin cámara, robot, archivos ni estado mutable. El
ejecutor es quien lleva a cabo el plan. Así se prueban con pytest y una forma
nueva de tomar, voltear o dejar es solo otra clase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gt1.domain import Piece, Plan, TrayDetection


class PickStrategy(ABC):
    @abstractmethod
    def plan(self, piece: Piece) -> Plan:
        """Desde fuera de la vista hasta la pieza tomada y levantada."""


class FlipStrategy(ABC):
    @abstractmethod
    def plan(self, piece: Piece) -> Plan:
        """Con la pieza tomada (hueso arriba), hasta dejarla con la piel arriba, lista para retomarla."""


class PlaceStrategy(ABC):
    @abstractmethod
    def plan(self, piece: Piece, tray: TrayDetection) -> Plan:
        """Con la pieza tomada, hasta dejarla en la bandeja y subir el gripper sobre ella.

        Salir de la vista de la cámara es otro plan (``clear_view_plan``), que se usa también antes
        de cada captura.
        """

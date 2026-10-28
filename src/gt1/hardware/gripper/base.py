"""Interfaz de los grippers (y de cualquier herramienta montada en la brida)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from gt1.domain import ToolSettings


@dataclass(frozen=True)
class GripperAction:
    name: str
    description: str


class Gripper(ABC):
    """Herramienta con acciones por nombre ('grip', 'release' y las que defina su archivo)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del archivo de configuración (p. ej. 'soft_2dedos')."""

    @property
    @abstractmethod
    def tool_settings(self) -> ToolSettings:
        """TCP y carga que hay que cargar en el robot con esta herramienta montada."""

    @property
    @abstractmethod
    def actions(self) -> tuple[GripperAction, ...]: ...

    @abstractmethod
    def perform(self, action: str) -> None:
        """Ejecuta una acción y espera a que termine. Solo con el robot quieto."""

"""Herramienta montada en la brida del robot y salidas digitales que la accionan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gt1.geometry import FloatArray

from .pose import Pose


class OutputPort(StrEnum):
    """Grupo de salidas digitales del controlador UR."""

    STANDARD = "standard"  # DO0..DO7 del gabinete
    CONFIGURABLE = "configurable"  # CO0..CO7 del gabinete
    TOOL = "tool"  # TO0..TO1 en el conector de la herramienta


@dataclass(frozen=True)
class DigitalOutput:
    port: OutputPort
    number: int

    def describe(self) -> str:
        prefix = {OutputPort.STANDARD: "DO", OutputPort.CONFIGURABLE: "CO", OutputPort.TOOL: "TO"}[self.port]
        return f"{prefix}{self.number}"


@dataclass(frozen=True, eq=False)
class ToolSettings:
    """Lo que el controlador necesita saber de la herramienta: punta (TCP) y carga."""

    tcp: Pose  # punta respecto de la brida
    payload_kg: float
    center_of_gravity_mm: FloatArray  # (3,) respecto de la brida

"""Entradas y salidas digitales del controlador del robot (gripper, tope, sensores)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OutputPort(StrEnum):
    """Grupo de salidas digitales del controlador UR."""

    STANDARD = "standard"  # DO0..DO7 del gabinete
    CONFIGURABLE = "configurable"  # CO0..CO7 del gabinete
    TOOL = "tool"  # TO0..TO1 en el conector de la herramienta


class InputPort(StrEnum):
    """Grupo de entradas digitales del controlador UR."""

    STANDARD = "standard"  # DI0..DI7 del gabinete
    CONFIGURABLE = "configurable"  # CI0..CI7 del gabinete
    TOOL = "tool"  # TI0..TI1 en el conector de la herramienta


@dataclass(frozen=True)
class DigitalOutput:
    port: OutputPort
    number: int

    def describe(self) -> str:
        prefix = {OutputPort.STANDARD: "DO", OutputPort.CONFIGURABLE: "CO", OutputPort.TOOL: "TO"}[self.port]
        return f"{prefix}{self.number}"


@dataclass(frozen=True)
class DigitalInput:
    port: InputPort
    number: int

    def describe(self) -> str:
        prefix = {InputPort.STANDARD: "DI", InputPort.CONFIGURABLE: "CI", InputPort.TOOL: "TI"}[self.port]
        return f"{prefix}{self.number}"

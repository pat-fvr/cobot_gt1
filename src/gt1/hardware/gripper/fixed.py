"""Herramienta sin acciones (p. ej. el pincho de calibración): solo aporta TCP y carga."""

from __future__ import annotations

from gt1.config.schemas import FixedToolConfig
from gt1.domain import ToolSettings

from .base import Gripper, GripperAction
from .tool_settings import tool_settings_from_config


class FixedTool(Gripper):
    def __init__(self, name: str, config: FixedToolConfig) -> None:
        self._name = name
        self._config = config

    @property
    def name(self) -> str:
        return self._name

    @property
    def tool_settings(self) -> ToolSettings:
        return tool_settings_from_config(self._config)

    @property
    def actions(self) -> tuple[GripperAction, ...]:
        return ()

    def perform(self, action: str) -> None:
        raise ValueError(f"La herramienta {self._name} no tiene acciones.")

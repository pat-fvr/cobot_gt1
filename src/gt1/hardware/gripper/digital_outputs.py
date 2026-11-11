"""Gripper accionado por salidas digitales del robot (p. ej. soft gripper neumático).

Cada acción del archivo de configuración es una lista de salidas que se fijan
en orden y una espera para que los dedos terminen de moverse. Si las salidas
ya estaban así, los dedos no se mueven y no se espera: p. ej. abrir antes de
tomar, justo después de soltar en la bandeja.
"""

from __future__ import annotations

import logging
import time

from gt1.config.schemas import DigitalOutputGripperConfig, GripperActionConfig
from gt1.domain import ToolSettings
from gt1.hardware.robot import Robot

from .base import Gripper, GripperAction
from .tool_settings import tool_settings_from_config

logger = logging.getLogger(__name__)


class DigitalOutputGripper(Gripper):
    def __init__(self, name: str, config: DigitalOutputGripperConfig, robot: Robot) -> None:
        self._name = name
        self._config = config
        self._robot = robot

    @property
    def name(self) -> str:
        return self._name

    @property
    def tool_settings(self) -> ToolSettings:
        return tool_settings_from_config(self._config)

    @property
    def actions(self) -> tuple[GripperAction, ...]:
        return tuple(GripperAction(name, action.description) for name, action in self._config.actions.items())

    def perform(self, action: str) -> None:
        if action not in self._config.actions:
            raise ValueError(f"El gripper {self._name} no tiene la acción {action!r}.")
        definition = self._config.actions[action]
        if self._outputs_already_set(definition):
            logger.info("Gripper %s: %s (ya estaba así)", self._name, action)
            return
        logger.info("Gripper %s: %s", self._name, action)
        for setting in definition.outputs:
            self._robot.set_digital_output(setting.digital_output(), setting.value)
        time.sleep(definition.settle_s)

    def _outputs_already_set(self, definition: GripperActionConfig) -> bool:
        return all(
            self._robot.digital_output(setting.digital_output()) == setting.value
            for setting in definition.outputs
        )

"""Robot simulado: llega al instante a cada objetivo y recuerda sus salidas y su herramienta.

Sirve para desarrollar y probar rutinas sin robot. No simula física ni
colisiones; un gemelo digital sería otra implementación de ``Robot``.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from gt1.config.schemas import SimRobotConfig
from gt1.domain import DigitalOutput, Pose, ToolSettings

from .base import Joints, MotionResult, Robot

logger = logging.getLogger(__name__)


class SimRobot(Robot):
    def __init__(self, config: SimRobotConfig) -> None:
        self._tcp = Pose.from_position(config.initial_tcp_mm)
        self._joints: Joints = config.initial_joints_deg
        self._outputs: dict[DigitalOutput, bool] = {}
        self._tool = ToolSettings(
            tcp=Pose.from_position((0.0, 0.0, 0.0)), payload_kg=0.0, center_of_gravity_mm=np.zeros(3)
        )
        self._connected = False
        self.freedrive = False

    @property
    def is_simulated(self) -> bool:
        return True

    def connect(self) -> None:
        self._connected = True
        logger.info("Robot simulado conectado.")

    def disconnect(self) -> None:
        self._connected = False

    def tcp_pose(self) -> Pose:
        self._require_connected()
        return self._tcp

    def joints_deg(self) -> Joints:
        self._require_connected()
        return self._joints

    def move_linear(self, target: Pose, speed_mm_s: float, stop: threading.Event) -> MotionResult:
        self._require_connected()
        if stop.is_set():
            return MotionResult.STOPPED
        self._tcp = target
        return MotionResult.COMPLETED

    def move_joints(self, joints_deg: Joints, speed_deg_s: float, stop: threading.Event) -> MotionResult:
        self._require_connected()
        if stop.is_set():
            return MotionResult.STOPPED
        self._joints = joints_deg
        return MotionResult.COMPLETED

    def set_digital_output(self, output: DigitalOutput, value: bool) -> None:
        self._require_connected()
        self._outputs[output] = value
        logger.info("[sim] %s -> %s", output.describe(), "1" if value else "0")

    def digital_output(self, output: DigitalOutput) -> bool:
        self._require_connected()
        return self._outputs.get(output, False)

    def apply_tool(self, tool: ToolSettings) -> None:
        self._require_connected()
        self._tool = tool

    def tool(self) -> ToolSettings:
        self._require_connected()
        return self._tool

    def set_freedrive(self, enabled: bool) -> None:
        self._require_connected()
        self.freedrive = enabled

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("El robot simulado no está conectado.")

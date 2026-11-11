"""Interfaz de los robots.

Hilos: un robot se usa desde un solo hilo (el hilo de trabajo de la rutina).
Para detener un movimiento desde otro hilo no se llama al robot: se activa la
señal ``stop`` que recibe cada movimiento, y el propio movimiento frena.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from types import TracebackType
from typing import Self

from gt1.domain import DigitalInput, DigitalOutput, MoveLinear, Pose, ToolSettings

Joints = tuple[float, float, float, float, float, float]


class MotionResult(StrEnum):
    COMPLETED = "completado"
    STOPPED = "detenido"  # se activó la señal de parada antes de llegar


class Robot(ABC):
    """Posiciones en mm y grados, en la base del robot; la punta es el TCP de la herramienta cargada."""

    @property
    @abstractmethod
    def is_simulated(self) -> bool: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None:
        """Detiene cualquier movimiento y libera el robot. Se puede llamar aunque no esté conectado."""

    @abstractmethod
    def tcp_pose(self) -> Pose:
        """Pose actual de la punta de la herramienta en la base."""

    @abstractmethod
    def joints_deg(self) -> Joints: ...

    @abstractmethod
    def move_linear(self, target: Pose, speed_mm_s: float, stop: threading.Event) -> MotionResult:
        """Lleva la punta en línea recta a ``target``. Vuelve al llegar o al activarse ``stop``."""

    def move_linear_path(self, moves: Sequence[MoveLinear], stop: threading.Event) -> MotionResult:
        """Movimientos rectos seguidos: pasa en curva por cada punto con su ``blend_radius_mm`` y se detiene
        en el último. Vuelve al llegar o al activarse ``stop``.

        Esta versión los hace de a uno, deteniéndose en cada punto: basta para los robots simulados. Un robot
        real la reemplaza por una trayectoria sin paradas.
        """
        for move in moves:
            if self.move_linear(move.target, move.speed_mm_s, stop) == MotionResult.STOPPED:
                return MotionResult.STOPPED
        return MotionResult.COMPLETED

    @abstractmethod
    def move_joints(self, joints_deg: Joints, speed_deg_s: float, stop: threading.Event) -> MotionResult:
        """Movimiento articular a ``joints_deg``. Vuelve al llegar o al activarse ``stop``."""

    @abstractmethod
    def set_digital_output(self, output: DigitalOutput, value: bool) -> None: ...

    @abstractmethod
    def digital_output(self, output: DigitalOutput) -> bool: ...

    @abstractmethod
    def digital_input(self, sensor: DigitalInput) -> bool: ...

    @abstractmethod
    def apply_tool(self, tool: ToolSettings) -> None:
        """Carga en el controlador el TCP y la carga (payload) de la herramienta montada."""

    @abstractmethod
    def tool(self) -> ToolSettings:
        """TCP y carga que el controlador tiene cargados ahora."""

    @abstractmethod
    def set_freedrive(self, enabled: bool) -> None:
        """Modo de guiado a mano (freedrive) para enseñar posiciones."""

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.disconnect()

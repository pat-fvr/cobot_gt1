"""Ejecutor: lleva a cabo un ``Plan`` con el robot y el gripper, paso por paso.

Cada paso arranca solo si el anterior terminó: los movimientos vuelven al
llegar, así que el gripper toma y suelta siempre con el robot quieto. La
excepción son los movimientos rectos seguidos que pasan en curva por su punto
(``blend_radius_mm``): van juntos, en una sola trayectoria, hasta el primero que
se detiene; una acción del gripper nunca queda dentro de una. Si se
activa la señal de parada o el robot informa un error, el plan se corta ahí,
con el robot detenido y el gripper como estaba (si tenía la pieza, la sigue
teniendo: soltarla a medio camino es peor). El resultado dice hasta dónde
llegó y por qué, para mostrarlo y registrarlo.

Las esperas (``Wait``, ``WaitForInput`` y ``WaitForOutput``) revisan la señal de
parada: la tecla ``x`` corta también la espera a que llegue una bandeja.

``still_since_s`` dice desde cuándo el robot está quieto (después del último
movimiento que hizo el ejecutor): una captura no repite la espera de vibración
si el robot ya lleva ese tiempo detenido.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from gt1.domain import (
    GripperCommand,
    MoveJoints,
    MoveLinear,
    MoveVertical,
    Plan,
    PlanStep,
    Pose,
    SetOutput,
    Wait,
    WaitForInput,
    WaitForOutput,
)
from gt1.hardware.gripper import Gripper
from gt1.hardware.robot import MotionResult, Robot

logger = logging.getLogger(__name__)

# Cada cuánto se lee una entrada o salida mientras se espera: muy por debajo de lo que tarda una bandeja en
# llegar.
IO_POLL_S = 0.02
MOTION_STEPS = (MoveLinear, MoveVertical, MoveJoints)


class PlanOutcome(StrEnum):
    COMPLETED = "completado"
    STOPPED = "detenido"  # el operador detuvo el plan
    FAILED = "fallo"  # el robot o el gripper informaron un error


@dataclass(frozen=True)
class PlanExecution:
    plan: Plan
    outcome: PlanOutcome
    steps_completed: int
    reason: str  # vacío si se completó

    @property
    def completed(self) -> bool:
        return self.outcome == PlanOutcome.COMPLETED

    def describe(self) -> str:
        if self.completed:
            return f"{self.plan.name}: completado"
        step = self.plan.steps[self.steps_completed].description
        return f"{self.plan.name}: {self.outcome.value} en '{step}' ({self.reason})"


class PlanExecutor:
    def __init__(self, robot: Robot, gripper: Gripper) -> None:
        self._robot = robot
        self._gripper = gripper
        self._still_since_s: float | None = None

    @property
    def still_since_s(self) -> float | None:
        """Desde cuándo (``time.monotonic()``) está quieto el robot; None si este ejecutor no lo movió."""
        return self._still_since_s

    def execute(self, plan: Plan, stop: threading.Event) -> PlanExecution:
        self._check_gripper_actions(plan)
        logger.info("Plan '%s': %d pasos.", plan.name, len(plan.steps))
        index = 0
        while index < len(plan.steps):
            if stop.is_set():
                return PlanExecution(plan, PlanOutcome.STOPPED, index, "parada pedida por el operador")
            group = blended_group(plan.steps, index)
            try:
                result = self._run_group(group, stop)
                if any(isinstance(step, MOTION_STEPS) for step in group):
                    self._still_since_s = time.monotonic()
            except RuntimeError as error:
                self._still_since_s = None  # no se sabe cómo quedó
                logger.error("Plan '%s', paso '%s': %s", plan.name, group[0].description, error)
                return PlanExecution(plan, PlanOutcome.FAILED, index, str(error))
            if result == MotionResult.STOPPED:
                return PlanExecution(plan, PlanOutcome.STOPPED, index, "parada pedida por el operador")
            index += len(group)
        return PlanExecution(plan, PlanOutcome.COMPLETED, len(plan.steps), "")

    def _check_gripper_actions(self, plan: Plan) -> None:
        """Antes de mover nada: un plan con una acción que el gripper no tiene no arranca."""
        available = {action.name for action in self._gripper.actions}
        missing = {step.action for step in plan.steps if isinstance(step, GripperCommand)} - available
        if missing:
            raise ValueError(
                f"El plan '{plan.name}' usa acciones que el gripper {self._gripper.name} no tiene: "
                f"{', '.join(sorted(missing))}. Agregarlas a config/grippers/{self._gripper.name}.toml."
            )

    def _run_group(self, group: tuple[PlanStep, ...], stop: threading.Event) -> MotionResult:
        """Un paso solo, o una trayectoria de movimientos rectos que pasan en curva por sus puntos."""
        if len(group) == 1:
            return self._run(group[0], stop)
        moves = [step for step in group if isinstance(step, MoveLinear)]
        logger.info("Trayectoria: %s", " -> ".join(move.description for move in moves))
        return self._robot.move_linear_path(moves, stop)

    def _run(self, step: PlanStep, stop: threading.Event) -> MotionResult:
        logger.info("Paso: %s", step.description)
        match step:
            case MoveLinear() | MoveVertical():
                return self._robot.move_linear(self._linear_target(step), step.speed_mm_s, stop)
            case MoveJoints():
                return self._robot.move_joints(step.joints_deg, step.speed_deg_s, stop)
            case GripperCommand():
                self._gripper.perform(step.action)
                return MotionResult.COMPLETED
            case Wait():
                return MotionResult.STOPPED if stop.wait(step.duration_s) else MotionResult.COMPLETED
            case SetOutput():
                self._robot.set_digital_output(step.output, step.value)
                return MotionResult.COMPLETED
            case WaitForInput() | WaitForOutput():
                return self._wait_until(lambda: self._signal(step) == step.value, stop)

    def _signal(self, step: WaitForInput | WaitForOutput) -> bool:
        """Lo que vale ahora la entrada o la salida que espera el paso."""
        if isinstance(step, WaitForInput):
            return self._robot.digital_input(step.sensor)
        return self._robot.digital_output(step.output)

    def _linear_target(self, step: MoveLinear | MoveVertical) -> Pose:
        """El destino del movimiento recto; el de uno vertical sale de donde está el robot ahora."""
        if isinstance(step, MoveLinear):
            return step.target
        return self._robot.tcp_pose().at_height(step.z_mm)

    @staticmethod
    def _wait_until(condition: Callable[[], bool], stop: threading.Event) -> MotionResult:
        while not condition():
            if stop.wait(IO_POLL_S):
                return MotionResult.STOPPED
        return MotionResult.COMPLETED


def blended_group(steps: Sequence[PlanStep], start: int) -> tuple[PlanStep, ...]:
    """El paso ``start`` y los movimientos rectos que le siguen mientras cada uno pase en curva al otro."""
    end = start + 1
    while end < len(steps):
        previous, following = steps[end - 1], steps[end]
        if not (
            isinstance(previous, MoveLinear)
            and previous.blends_into_next
            and isinstance(following, MoveLinear)
        ):
            break
        end += 1
    return tuple(steps[start:end])

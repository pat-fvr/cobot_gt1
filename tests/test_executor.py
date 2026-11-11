"""Ejecutor de planes: paso por paso, y se corta limpio ante una parada o un error del robot."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import pytest

from gt1.config.schemas import SimRobotConfig
from gt1.domain import (
    DigitalInput,
    DigitalOutput,
    GripperCommand,
    InputPort,
    MoveJoints,
    MoveLinear,
    OutputPort,
    Plan,
    Pose,
    SetOutput,
    WaitForInput,
    joined_plan,
)
from gt1.hardware.gripper.digital_outputs import DigitalOutputGripper
from gt1.hardware.robot import MotionResult
from gt1.hardware.robot.sim import SimRobot
from gt1.runtime.executor import PlanExecutor, PlanOutcome, blended_group

from .fakes import DO1_GRIPPER

DO1 = DigitalOutput(OutputPort.STANDARD, 1)
SIM = SimRobotConfig(
    kind="sim", initial_tcp_mm=(0.0, 0.0, 300.0), initial_joints_deg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
)
GRIPPER = DO1_GRIPPER
TARGET = Pose.from_position((400.0, 200.0, 50.0))
PICK_LIKE = Plan(
    name="prueba",
    steps=(
        MoveLinear(TARGET, 100.0, "bajar"),
        GripperCommand("grip", "tomar"),
        MoveJoints((10.0, -90.0, 90.0, 0.0, 90.0, 0.0), 30.0, "salir"),
    ),
)


class FailingRobot(SimRobot):
    """Robot simulado que informa un error (p. ej. una parada de protección) en el movimiento recto."""

    def move_linear(self, target: Pose, speed_mm_s: float, stop: threading.Event) -> MotionResult:
        raise RuntimeError("parada de protección")


class PathRecordingRobot(SimRobot):
    """Robot simulado que anota qué movimientos rectos recibió juntos, en una sola trayectoria."""

    def __init__(self, config: SimRobotConfig) -> None:
        super().__init__(config)
        self.paths: list[list[str]] = []

    def move_linear_path(self, moves: Sequence[MoveLinear], stop: threading.Event) -> MotionResult:
        self.paths.append([move.description for move in moves])
        return super().move_linear_path(moves, stop)


def above(z_mm: float) -> Pose:
    return Pose.from_position((400.0, 200.0, z_mm))


BLENDED_PICK = Plan(
    name="tomar",
    steps=(
        GripperCommand("release", "abrir"),
        MoveLinear(above(90.0), 300.0, "sobre la pieza", blend_radius_mm=30.0),
        MoveLinear(above(0.0), 100.0, "bajar"),
        GripperCommand("grip", "tomar"),
        MoveLinear(above(90.0), 100.0, "subir", blend_radius_mm=30.0),
    ),
)
OVER_THE_TRAY = Pose.from_position((0.0, 500.0, 90.0))
PLACE = Plan(name="dejar", steps=(MoveLinear(OVER_THE_TRAY, 300.0, "sobre la bandeja"),))


def test_runs_every_step_in_order() -> None:
    with SimRobot(SIM) as robot:
        execution = PlanExecutor(robot, DigitalOutputGripper("g", GRIPPER, robot)).execute(
            PICK_LIKE, threading.Event()
        )
        assert execution.outcome == PlanOutcome.COMPLETED
        assert execution.steps_completed == 3
        assert robot.tcp_pose() is TARGET
        assert robot.digital_output(DO1)
        assert robot.joints_deg()[0] == 10.0


def test_a_stop_before_starting_runs_nothing() -> None:
    stop = threading.Event()
    stop.set()
    with SimRobot(SIM) as robot:
        execution = PlanExecutor(robot, DigitalOutputGripper("g", GRIPPER, robot)).execute(PICK_LIKE, stop)
        assert execution.outcome == PlanOutcome.STOPPED
        assert execution.steps_completed == 0
        assert "bajar" in execution.describe()


def test_a_robot_error_stops_the_plan_and_leaves_the_gripper_as_it_was() -> None:
    with FailingRobot(SIM) as robot:
        execution = PlanExecutor(robot, DigitalOutputGripper("g", GRIPPER, robot)).execute(
            PICK_LIKE, threading.Event()
        )
        assert execution.outcome == PlanOutcome.FAILED
        assert execution.steps_completed == 0
        assert "parada de protección" in execution.describe()
        assert not robot.digital_output(DO1)


def test_a_plan_with_an_unknown_gripper_action_does_not_start() -> None:
    plan = Plan(name="raro", steps=(MoveLinear(TARGET, 100.0, "bajar"), GripperCommand("suck", "aspirar")))
    with SimRobot(SIM) as robot:
        with pytest.raises(ValueError, match="suck"):
            PlanExecutor(robot, DigitalOutputGripper("g", GRIPPER, robot)).execute(plan, threading.Event())
        assert robot.tcp_pose().position_mm[2] == 300.0


def test_blended_moves_go_together_up_to_the_first_one_that_stops() -> None:
    steps = joined_plan("cargar", (BLENDED_PICK, PLACE)).steps
    descriptions = [[step.description for step in blended_group(steps, start)] for start in (0, 1, 3, 4)]
    assert descriptions == [["abrir"], ["sobre la pieza", "bajar"], ["tomar"], ["subir", "sobre la bandeja"]]


def test_a_blended_move_at_the_end_of_a_plan_runs_alone() -> None:
    assert [step.description for step in blended_group(BLENDED_PICK.steps, 4)] == ["subir"]


def test_the_executor_sends_blended_moves_as_one_path_and_grips_with_the_robot_still() -> None:
    with PathRecordingRobot(SIM) as robot:
        execution = PlanExecutor(robot, DigitalOutputGripper("g", GRIPPER, robot)).execute(
            joined_plan("cargar", (BLENDED_PICK, PLACE)), threading.Event()
        )
        assert execution.outcome == PlanOutcome.COMPLETED
        assert execution.steps_completed == 6
        assert robot.paths == [["sobre la pieza", "bajar"], ["subir", "sobre la bandeja"]]
        assert robot.tcp_pose() is OVER_THE_TRAY


DI1 = DigitalInput(InputPort.STANDARD, 1)
DO3 = DigitalOutput(OutputPort.STANDARD, 3)
WAIT_FOR_DI1 = Plan(
    name="esperar",
    steps=(WaitForInput(DI1, True, "esperar DI1"), SetOutput(DO3, True, "fijar DO3")),
)


def test_sets_an_output_once_the_input_it_waits_for_turns_on() -> None:
    with SimRobot(SIM) as robot:
        robot.set_digital_input(DI1, True)
        execution = PlanExecutor(robot, DigitalOutputGripper("g", GRIPPER, robot)).execute(
            WAIT_FOR_DI1, threading.Event()
        )
        assert execution.outcome == PlanOutcome.COMPLETED
        assert robot.digital_output(DO3)


def test_a_stop_ends_the_wait_for_an_input_that_never_turns_on() -> None:
    stop = threading.Event()
    timer = threading.Timer(0.1, stop.set)
    with SimRobot(SIM) as robot:
        timer.start()
        execution = PlanExecutor(robot, DigitalOutputGripper("g", GRIPPER, robot)).execute(WAIT_FOR_DI1, stop)
        timer.cancel()
        assert execution.outcome == PlanOutcome.STOPPED
        assert execution.steps_completed == 0
        assert not robot.digital_output(DO3)

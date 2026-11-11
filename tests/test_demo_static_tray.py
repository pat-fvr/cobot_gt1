"""demo_static_tray sin hardware: robot simulado junto a una cinta con tope, escenas sintéticas a 1 mm/px.

Con la escala nominal la base coincide con la imagen: +X es hacia la derecha, así
que ``left_to_right_deg = 0``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gt1_test.demo_static_tray import PickFrom, Settings, StaticTrayDemo, first_piece

from gt1.calibration import NominalScaleProjector
from gt1.config.schemas import (
    ClearViewSettings,
    InputChannel,
    OutputChannel,
    RobotPoseConfig,
    SimRobotConfig,
    TopDownPickConfig,
    TrayStationSettings,
)
from gt1.domain import DigitalOutput, Frame, InputPort, OutputPort, Zone
from gt1.domain.frame import Image
from gt1.hardware.gripper.digital_outputs import DigitalOutputGripper
from gt1.hardware.robot.sim import SimRobot
from gt1.perception.pieces.skin_outline import SkinOutlinePieceDetector
from gt1.perception.scene import ScenePerception
from gt1.planning import TopDownPick, tray_station_drop_plan
from gt1.runtime.clear_view import ClearViewCamera
from gt1.runtime.console import OperatorConsole
from gt1.runtime.executor import PlanExecutor
from gt1.runtime.run_log import RunLog

from . import scenes
from .fakes import DO1_GRIPPER, SequenceCamera, TrayConveyorRobot

SIZE_PX = (900, 400)
PROJECTOR = NominalScaleProjector(mm_per_px=1.0)
GRIPPER_OUTPUT = DigitalOutput(OutputPort.STANDARD, 1)
DROP_JOINTS_DEG = (40.0, -80.0, 100.0, -110.0, -90.0, 0.0)
PICK_POLYGON = [[5.0, 5.0], [895.0, 5.0], [895.0, 395.0], [5.0, 395.0]]
LEFT_BREAST_PX, RIGHT_BREAST_PX = (250.0, 200.0), (650.0, 200.0)

STATION = TrayStationSettings(
    stopper=OutputChannel(port=OutputPort.STANDARD, number=3),
    tray_sensor=InputChannel(port=InputPort.STANDARD, number=1),
    stopper_open_s=0.01,  # lo más corto que acepta: la prueba no espera
)
SETTINGS = Settings(
    pick_zone="pick",
    left_to_right_deg=0.0,
    pick_from="left",
    clear_view=ClearViewSettings(
        pose="tray_drop",
        safe_height_mm=200.0,
        rise_speed_mm_s=50.0,
        clear_speed_deg_s=30.0,
        capture_settle_s=0.0,
    ),
    tray_station=STATION,
)
SIM = SimRobotConfig(
    kind="sim", initial_tcp_mm=(450.0, 170.0, 300.0), initial_joints_deg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
)


def perception() -> ScenePerception:
    zone = Zone(name="pick", polygon_mm=np.array(PICK_POLYGON), surface_z_mm=0.0)
    return ScenePerception(
        detector=SkinOutlinePieceDetector(scenes.SKIN_DETECTOR, PROJECTOR),
        classifier=None,
        projector=PROJECTOR,
        zone_for=lambda frame: zone,
    )


def scene(breasts: Sequence[tuple[float, float]]) -> Image:
    """Mesa oscura con pechugas lisas (piel arriba) en estos centros."""
    image = scenes.table(SIZE_PX, 70.0)
    for center in breasts:
        image = scenes.with_breast(image, center, 90.0, 170.0)
    return image


@pytest.fixture
def run_log(tmp_path: Path) -> Iterator[RunLog]:
    log = RunLog.create(tmp_path, "demo_static_tray")
    yield log
    log.close()


def run_demo(robot: SimRobot, camera: SequenceCamera, run_log: RunLog, stop: threading.Event) -> None:
    """Aprieta 'p': carga hasta que no queden piezas o se active ``stop``."""
    with robot:
        executor = PlanExecutor(robot, DigitalOutputGripper("g", DO1_GRIPPER, robot))
        console = OperatorConsole("prueba")
        demo = StaticTrayDemo(
            clear_view=ClearViewCamera(
                camera=camera,
                robot=robot,
                executor=executor,
                table_z_mm=0.0,
                clear_pose=RobotPoseConfig(
                    joints_deg=DROP_JOINTS_DEG, tcp_mm=(500.0, 700.0, 250.0), taught_at="2026-11-11 12:00"
                ),
                settings=SETTINGS.clear_view,
                report=console.set_status,
            ),
            executor=executor,
            pick=TopDownPick(
                TopDownPickConfig(
                    travel_speed_mm_s=150.0,
                    approach_height_mm=80.0,
                    grasp_height_mm=0.0,
                    after_grip_descent_mm=30.0,
                    lift_height_mm=80.0,
                    descend_speed_mm_s=50.0,
                    blend_radius_mm=30.0,
                ),
                0.0,
                "-y",
            ),
            wide_side="-y",
            drop=tray_station_drop_plan(STATION),
            perception=perception(),
            projector=PROJECTOR,
            settings=SETTINGS,
            console=console,
            run_log=run_log,
        )
        demo.run(stop)


def conveyor_robot() -> TrayConveyorRobot:
    return TrayConveyorRobot(SIM, STATION.stopper.digital_output(), STATION.tray_sensor.digital_input())


def records(run_log: RunLog) -> list[Any]:
    attempts = run_log.directory / "attempts.jsonl"
    lines = attempts.read_text(encoding="utf-8").splitlines() if attempts.exists() else []
    return [json.loads(line) for line in lines]


def test_loads_every_breast_from_left_to_right_one_tray_each_until_none_is_left(run_log: RunLog) -> None:
    robot = conveyor_robot()
    camera = SequenceCamera([scene([LEFT_BREAST_PX, RIGHT_BREAST_PX]), scene([RIGHT_BREAST_PX]), scene([])])
    run_demo(robot, camera, run_log, threading.Event())

    loads = records(run_log)
    assert [load["outcome"] for load in loads] == ["cargada", "cargada"]
    picked_x_mm = [load["piece"]["center_mm"][0] for load in loads]
    assert picked_x_mm == pytest.approx([LEFT_BREAST_PX[0], RIGHT_BREAST_PX[0]], abs=10.0)
    # Una foto por carga y una última sin piezas; cada bandeja cargada salió por el tope.
    assert camera.captures == 3
    assert robot.trays_released == 2
    with robot:
        assert robot.joints_deg() == DROP_JOINTS_DEG
        assert robot.digital_output(STATION.stopper.digital_output())  # cerrado para la siguiente
        assert not robot.digital_output(GRIPPER_OUTPUT)


def test_waits_for_the_tray_with_the_piece_held_until_stopped(run_log: RunLog) -> None:
    """Sin bandeja en el sensor el robot no suelta: 'x' corta la espera con la pieza tomada."""
    robot = SimRobot(SIM)  # su sensor de bandeja queda siempre en 0
    camera = SequenceCamera([scene([LEFT_BREAST_PX])])
    stop = threading.Event()
    timer = threading.Timer(0.2, stop.set)
    timer.start()
    run_demo(robot, camera, run_log, stop)
    timer.cancel()

    (load,) = records(run_log)
    assert load["outcome"] == "detenido"
    assert load["pick"]["outcome"] == "completado"
    drop = load["drop"]
    assert drop["steps"][drop["steps_completed"]] == "esperar la bandeja (DI1)"
    assert camera.captures == 1
    with robot:
        assert robot.digital_output(GRIPPER_OUTPUT)  # sigue con la pieza


@pytest.mark.parametrize(("pick_from", "expected_px"), [("left", LEFT_BREAST_PX), ("right", RIGHT_BREAST_PX)])
def test_the_first_piece_is_the_one_furthest_to_the_chosen_side(
    pick_from: PickFrom, expected_px: tuple[float, float]
) -> None:
    seen = perception().perceive(Frame(scene([RIGHT_BREAST_PX, LEFT_BREAST_PX]), 0.0))
    piece = first_piece(seen, left_to_right_deg=0.0, pick_from=pick_from)
    assert piece is not None
    assert piece.center_mm[:2] == pytest.approx(expected_px, abs=10.0)


def test_without_pickable_pieces_there_is_no_first_piece() -> None:
    seen = perception().perceive(Frame(scene([]), 0.0))
    assert first_piece(seen, left_to_right_deg=0.0, pick_from="left") is None

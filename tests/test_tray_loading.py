"""tray_loading_test sin hardware: robot simulado, escenas sintéticas a 1 mm/px y la percepción real.

Las pechugas se dejan en la mitad superior de la imagen y las bandejas en la
inferior. La colocación pide los lóbulos hacia 180°: en una bandeja a 0°, hacia
la izquierda de la imagen.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gt1_test.tray_loading_test import Loading, LoadingPerception, Settings, TrayLoadingTest, free_trays

from gt1.calibration import NominalScaleProjector
from gt1.config.schemas import (
    CellConfig,
    ClearViewSettings,
    RobotPoseConfig,
    SimRobotConfig,
    TopDownPickConfig,
    TopDownPlaceConfig,
)
from gt1.domain import DigitalOutput, Frame, OutputPort, Zone
from gt1.domain.frame import Image
from gt1.hardware.gripper.digital_outputs import DigitalOutputGripper
from gt1.hardware.robot.sim import SimRobot
from gt1.perception.pieces.skin_outline import SkinOutlinePieceDetector
from gt1.perception.scene import ScenePerception
from gt1.perception.tray.rim_and_floor import RimAndFloorTrayDetector
from gt1.planning import TopDownPick, TopDownPlace
from gt1.runtime.clear_view import ClearViewCamera
from gt1.runtime.console import OperatorConsole
from gt1.runtime.executor import PlanExecutor
from gt1.runtime.run_log import RunLog

from . import scenes
from .fakes import DO1_GRIPPER, SequenceCamera

SIZE_PX = (900, 700)
PROJECTOR = NominalScaleProjector(mm_per_px=1.0)
DO1 = DigitalOutput(OutputPort.STANDARD, 1)
CLEAR_JOINTS_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)
CLEAR_POSE = RobotPoseConfig(
    joints_deg=CLEAR_JOINTS_DEG, tcp_mm=(0.0, 0.0, 300.0), taught_at="2026-11-10 12:00"
)
PICK_POLYGON = [[5.0, 5.0], [895.0, 5.0], [895.0, 340.0], [5.0, 340.0]]
TRAY_POLYGON = [[5.0, 360.0], [895.0, 360.0], [895.0, 695.0], [5.0, 695.0]]
LEFT_TRAY_PX, RIGHT_TRAY_PX = (250.0, 530.0), (650.0, 530.0)
LOBES_TOWARD_DEG = 180.0

CELL = CellConfig.model_validate_json(
    json.dumps(
        {
            "table_z_mm": 0.0,
            "zones": {
                "pick": {"polygon_mm": PICK_POLYGON},
                "tray": {"polygon_mm": TRAY_POLYGON},
                "flip_landing": {"polygon_mm": [[0, 0], [1, 0], [1, 1]]},
            },
            "tray_placement": {"center_mm": LEFT_TRAY_PX, "axis_angle_deg": 0.0},
        }
    )
)
SETTINGS = Settings(
    pick_zone="pick",
    tray_zone="tray",
    load_order_toward_deg=0.0,  # escala nominal: la base coincide con la imagen, +X es a la derecha
    verify_loads=False,
    max_lobes_error_deg=30.0,
    clear_view=ClearViewSettings(
        pose="home", safe_height_mm=200.0, rise_speed_mm_s=50.0, clear_speed_deg_s=30.0, capture_settle_s=0.0
    ),
)


def zone(name: str, polygon: list[list[float]]) -> Zone:
    return Zone(name=name, polygon_mm=np.array(polygon), surface_z_mm=0.0)


def perception() -> LoadingPerception:
    def in_zone(name: str, polygon: list[list[float]]) -> ScenePerception:
        fixed = zone(name, polygon)
        return ScenePerception(
            detector=SkinOutlinePieceDetector(scenes.SKIN_DETECTOR, PROJECTOR),
            classifier=None,
            projector=PROJECTOR,
            zone_for=lambda frame: fixed,
        )

    # De a 5°: las bandejas de estas escenas están a 0° y la prueba corre más rápido.
    tray_detector = scenes.TRAY_DETECTOR.model_copy(update={"angle_step_deg": 5.0})
    return LoadingPerception(
        pieces=in_zone("pick", PICK_POLYGON),
        tray_contents=in_zone("tray", TRAY_POLYGON),
        trays=RimAndFloorTrayDetector(tray_detector, scenes.TRAY, PROJECTOR),
        tray_zone=zone("tray", TRAY_POLYGON),
    )


def scene(
    pieces: Sequence[tuple[tuple[float, float], float]],
    trays: Sequence[tuple[float, float]] = (LEFT_TRAY_PX,),
) -> Image:
    """Mesa oscura con bandejas a 0° y pechugas (centro, sentido de los lóbulos), dentro o fuera de ellas."""
    image = scenes.table(SIZE_PX, 70.0)
    for center in trays:
        image = scenes.with_tray(image, center, 0.0)
    for center, lobes_deg in pieces:
        image = scenes.with_breast(image, center, lobes_deg, 170.0)
    return image


@pytest.fixture
def run_log(tmp_path: Path) -> Iterator[RunLog]:
    log = RunLog.create(tmp_path, "tray_loading_test")
    yield log
    log.close()


def run_cycle(
    images: list[Image], run_log: RunLog, settings: Settings = SETTINGS, keys: str = "cp"
) -> tuple[SimRobot, SequenceCamera]:
    """Aprieta ``keys`` en orden: 'c' captura y calcula las cargas, 'p' las ejecuta."""
    camera = SequenceCamera(images)
    robot = SimRobot(
        SimRobotConfig(
            kind="sim",
            initial_tcp_mm=(450.0, 170.0, 300.0),
            initial_joints_deg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
    )
    with robot:
        executor = PlanExecutor(robot, DigitalOutputGripper("g", DO1_GRIPPER, robot))
        console = OperatorConsole("prueba")
        test = TrayLoadingTest(
            clear_view=ClearViewCamera(
                camera=camera,
                robot=robot,
                executor=executor,
                table_z_mm=CELL.table_z_mm,
                clear_pose=CLEAR_POSE,
                settings=settings.clear_view,
                report=console.set_status,
            ),
            executor=executor,
            wide_side="-y",
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
                CELL.table_z_mm,
                "-y",
            ),
            place=TopDownPlace(
                TopDownPlaceConfig(
                    travel_speed_mm_s=150.0,
                    release_height_mm=50.0,
                    lobes_toward_deg=LOBES_TOWARD_DEG,
                ),
                "-y",
            ),
            perception=perception(),
            lobes_toward_deg=LOBES_TOWARD_DEG,
            projector=PROJECTOR,
            settings=settings,
            console=console,
            run_log=run_log,
        )
        jobs = {"c": test.plan_cycle, "p": test.run_cycle}
        for key in keys:
            jobs[key](threading.Event())
    return robot, camera


def records(run_log: RunLog) -> list[Any]:
    attempts = run_log.directory / "attempts.jsonl"
    lines = attempts.read_text(encoding="utf-8").splitlines() if attempts.exists() else []
    return [json.loads(line) for line in lines]


VERIFIED = SETTINGS.model_copy(update={"verify_loads": True})
TWO_TRAYS = (LEFT_TRAY_PX, RIGHT_TRAY_PX)
LEFT_BREAST, RIGHT_BREAST = ((300.0, 170.0), 120.0), ((650.0, 170.0), 0.0)


def test_one_capture_loads_every_breast_in_its_own_tray_without_going_back_between_loads(
    run_log: RunLog,
) -> None:
    robot, camera = run_cycle([scene([LEFT_BREAST, RIGHT_BREAST], TWO_TRAYS)], run_log)

    (record,) = records(run_log)
    assert [load["outcome"] for load in record["loads"]] == ["dejada", "dejada"]
    for load in record["loads"]:
        assert load["execution"]["outcome"] == "completado"
    # De izquierda a derecha: la pieza de la izquierda a la bandeja de la izquierda, y después la otra.
    picked = [load["piece"]["center_mm"][0] for load in record["loads"]]
    loaded = [load["tray"]["center_mm"][0] for load in record["loads"]]
    assert picked == pytest.approx([LEFT_BREAST[0][0], RIGHT_BREAST[0][0]], abs=10.0)
    assert loaded == pytest.approx([LEFT_TRAY_PX[0], RIGHT_TRAY_PX[0]], abs=4.0)
    # Sin verificar: la única captura es la del principio; al final vuelve fuera de la vista.
    assert camera.captures == 1
    with robot:
        assert robot.joints_deg() == CLEAR_JOINTS_DEG
        assert not robot.digital_output(DO1)


def test_p_without_c_does_not_move_the_robot_nor_capture(run_log: RunLog) -> None:
    robot, camera = run_cycle([scene([LEFT_BREAST])], run_log, keys="p")

    assert camera.captures == 0
    assert records(run_log) == []
    with robot:
        assert robot.tcp_pose().position_mm == pytest.approx([450.0, 170.0, 300.0])


def test_p_runs_what_c_planned_only_once(run_log: RunLog) -> None:
    """La segunda 'p' no repite las cargas: las piezas ya se movieron y lo calculado dejó de valer."""
    _, camera = run_cycle([scene([LEFT_BREAST])], run_log, keys="cpp")

    assert camera.captures == 1
    assert len(records(run_log)) == 1


def test_breasts_without_a_free_tray_stay_where_they_are(run_log: RunLog) -> None:
    run_cycle([scene([LEFT_BREAST, RIGHT_BREAST])], run_log)

    (record,) = records(run_log)
    assert record["pickable_pieces"] == 2
    assert len(record["loads"]) == 1


def test_the_verification_checks_where_the_lobes_ended(run_log: RunLog) -> None:
    before = scene([LEFT_BREAST])
    after = scene([(LEFT_TRAY_PX, 180.0)])
    _, camera = run_cycle([before, after], run_log, VERIFIED)

    (load,) = records(run_log)[0]["loads"]
    assert load["outcome"] == "cargada"
    assert load["piece"]["lobes_direction_deg"] == pytest.approx(120.0, abs=3.0)
    assert load["target_lobes_deg"] == pytest.approx(180.0)
    assert load["lobes_error_deg"] < 5.0
    assert camera.captures == 2


def test_a_breast_left_with_its_lobes_reversed_is_reported(run_log: RunLog) -> None:
    before = scene([LEFT_BREAST])
    after = scene([(LEFT_TRAY_PX, 0.0)])
    run_cycle([before, after], run_log, VERIFIED)

    (load,) = records(run_log)[0]["loads"]
    assert load["outcome"] == "mal_orientada"
    assert load["lobes_error_deg"] > 170.0


def test_a_tray_with_a_piece_inside_is_not_free() -> None:
    """La pieza tapa el fondo de su bandeja, que puede no detectarse: no detectada tampoco se ofrece."""
    image = scene([(LEFT_TRAY_PX, 180.0)], TWO_TRAYS)
    loading: Loading = perception().perceive(Frame(image, 0.0))
    (free,) = free_trays(loading)
    assert free.center_mm[:2] == pytest.approx(RIGHT_TRAY_PX, abs=4.0)

"""pick_place_test sin hardware: robot simulado, imágenes sintéticas y la percepción real.

Escala nominal de 1 mm/px: la base coincide con la imagen. La zona de toma es la
mitad derecha y la de la bandeja, la izquierda.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from gt1_test.pick_place_test import CellPerception, PickPlaceTest, Settings

from gt1.calibration import NominalScaleProjector
from gt1.config.schemas import (
    CellConfig,
    ClearViewSettings,
    ReferencePieceDetectorConfig,
    RobotPoseConfig,
    SimRobotConfig,
    TextureFaceClassifierConfig,
    TopDownPickConfig,
    TopDownPlaceConfig,
    TrayConfig,
)
from gt1.domain import DigitalOutput, OutputPort, Zone
from gt1.domain.frame import Image
from gt1.hardware.gripper.digital_outputs import DigitalOutputGripper
from gt1.hardware.robot.sim import SimRobot
from gt1.perception.face.texture import TextureFaceClassifier
from gt1.perception.pieces.reference import ReferencePieceDetector
from gt1.perception.scene import ScenePerception
from gt1.planning import TopDownPick, TopDownPlace
from gt1.runtime.clear_view import ClearViewCamera
from gt1.runtime.console import OperatorConsole
from gt1.runtime.executor import PlanExecutor
from gt1.runtime.factories import build_fixed_tray
from gt1.runtime.run_log import RunLog

from .fakes import DO1_GRIPPER, SequenceCamera

SIZE_PX = (640, 480)
PROJECTOR = NominalScaleProjector(mm_per_px=1.0)
DO1 = DigitalOutput(OutputPort.STANDARD, 1)
CLEAR_JOINTS_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)
CLEAR_POSE = RobotPoseConfig(
    joints_deg=CLEAR_JOINTS_DEG, tcp_mm=(0.0, 0.0, 300.0), taught_at="2026-11-10 12:00"
)
PIECE_AXES_PX = (90, 60)
PICK_CENTER_PX, TRAY_CENTER_PX = (480, 240), (160, 240)

CELL = CellConfig.model_validate_json(
    json.dumps(
        {
            "table_z_mm": 0.0,
            "zones": {
                "pick": {"polygon_mm": [[330, 10], [630, 10], [630, 470], [330, 470]]},
                "tray": {"polygon_mm": [[10, 10], [310, 10], [310, 470], [10, 470]]},
                "flip_landing": {"polygon_mm": [[0, 0], [1, 0], [1, 1]]},
            },
            "tray_placement": {"center_mm": TRAY_CENTER_PX, "axis_angle_deg": 0.0},
        }
    )
)
TRAY = TrayConfig(
    description="prueba",
    rim_length_mm=230.0,
    rim_width_mm=170.0,
    floor_length_mm=180.0,
    floor_width_mm=140.0,
    depth_mm=50.0,
    seal_flange_mm=8.0,
    min_clearance_to_flange_mm=3.0,
)
SETTINGS = Settings(
    pick_zone="pick",
    tray_zone="tray",
    clear_view=ClearViewSettings(
        pose="home", safe_height_mm=200.0, rise_speed_mm_s=50.0, clear_speed_deg_s=30.0, capture_settle_s=0.0
    ),
)
DETECTOR = ReferencePieceDetectorConfig(
    kind="reference",
    min_difference_pct=15.0,
    glint_min_level=200,
    speck_kernel_px=3,
    blur_kernel_px=5,
    morphology_kernel_px=9,
    contour_height_mm=0.0,
    min_area_mm2=1000.0,
    max_single_piece_area_mm2=40000.0,
)
CLASSIFIER = TextureFaceClassifierConfig(
    kind="texture",
    smoothing_mm=2.0,
    mask_erosion_mm=5.0,
    threshold_pct_per_mm=3.3,
    min_margin_pct_per_mm=0.5,
)


def empty_table() -> Image:
    rows = np.linspace(60.0, 100.0, SIZE_PX[1])[:, np.newaxis]
    return np.repeat(rows, SIZE_PX[0], axis=1).astype(np.uint8)


def with_piece(table: Image, center_px: tuple[int, int], *, skin_up: bool) -> Image:
    """Piel: clara y lisa. Hueso: carne y grasa en manchas de 8 px, la textura de las costillas."""
    scene = table.astype(np.float64)
    mask = np.zeros(table.shape, dtype=np.uint8)
    cv2.ellipse(mask, center_px, PIECE_AXES_PX, 30, 0, 360, 255, -1)
    if skin_up:
        scene[mask > 0] = 170.0
    else:
        patches = np.random.default_rng(0).choice(
            [0.6, 1.5], size=(table.shape[0] // 8 + 1, table.shape[1] // 8 + 1)
        )
        texture = np.kron(patches, np.ones((8, 8)))[: table.shape[0], : table.shape[1]]
        scene[mask > 0] *= texture[mask > 0]
    return np.clip(scene, 0, 255).astype(np.uint8)


def cell_perception() -> CellPerception:
    def in_zone(name: str) -> ScenePerception:
        polygon = CELL.zones.pick if name == "pick" else CELL.zones.tray
        zone = Zone(name=name, polygon_mm=np.array(polygon.polygon_mm, dtype=np.float64), surface_z_mm=0.0)
        return ScenePerception(
            detector=ReferencePieceDetector(DETECTOR, PROJECTOR, empty_table()),
            classifier=TextureFaceClassifier(CLASSIFIER),
            projector=PROJECTOR,
            zone_for=lambda frame: zone,
        )

    return CellPerception(pick=in_zone("pick"), tray=in_zone("tray"))


@pytest.fixture
def run_log(tmp_path: Path) -> Iterator[RunLog]:
    log = RunLog.create(tmp_path, "pick_place_test")
    yield log
    log.close()


def run_attempt(images: list[Image], run_log: RunLog) -> tuple[SimRobot, SequenceCamera]:
    camera = SequenceCamera(images)
    robot = SimRobot(
        SimRobotConfig(
            kind="sim",
            initial_tcp_mm=(480.0, 240.0, 300.0),
            initial_joints_deg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
    )
    with robot:
        executor = PlanExecutor(robot, DigitalOutputGripper("g", DO1_GRIPPER, robot))
        console = OperatorConsole("prueba")
        clear_view = ClearViewCamera(
            camera=camera,
            robot=robot,
            executor=executor,
            table_z_mm=CELL.table_z_mm,
            clear_pose=CLEAR_POSE,
            settings=SETTINGS.clear_view,
            report=console.set_status,
        )
        test = PickPlaceTest(
            clear_view=clear_view,
            executor=executor,
            wide_side="-y",
            pick=TopDownPick(
                TopDownPickConfig(
                    travel_speed_mm_s=150.0,
                    approach_height_mm=150.0,
                    grasp_height_mm=15.0,
                    after_grip_descent_mm=0.0,
                    lift_height_mm=200.0,
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
                    lobes_toward_deg=180.0,
                ),
                "-y",
            ),
            tray=build_fixed_tray(CELL, TRAY),
            perception=cell_perception(),
            projector=PROJECTOR,
            console=console,
            run_log=run_log,
        )
        test.attempt(threading.Event())
    return robot, camera


def recorded(run_log: RunLog) -> Any:
    return json.loads((run_log.directory / "attempts.jsonl").read_text(encoding="utf-8"))


def test_a_skin_up_piece_ends_in_the_tray_and_the_camera_confirms_it(run_log: RunLog) -> None:
    before = with_piece(empty_table(), PICK_CENTER_PX, skin_up=True)
    after = with_piece(empty_table(), TRAY_CENTER_PX, skin_up=True)
    robot, camera = run_attempt([before, after], run_log)

    record = recorded(run_log)
    assert record["outcome"] == "colocada"
    assert record["execution"]["outcome"] == "completado"
    assert record["piece"]["center_mm"][:2] == pytest.approx(PICK_CENTER_PX, abs=3.0)
    # minAreaRect estima el eje de un contorno redondeado con unos grados de error.
    assert record["piece"]["axis_angle_deg"] == pytest.approx(30.0, abs=5.0)
    assert camera.captures == 2
    with robot:
        assert robot.joints_deg() == CLEAR_JOINTS_DEG
        assert not robot.digital_output(DO1)


def test_a_bone_up_piece_is_skipped_and_the_robot_only_clears_the_view(run_log: RunLog) -> None:
    before = with_piece(empty_table(), PICK_CENTER_PX, skin_up=False)
    robot, camera = run_attempt([before], run_log)

    record = recorded(run_log)
    assert record["outcome"] == "sin_pieza"
    assert record["skipped"] == {"hueso": 1}
    assert camera.captures == 1
    with robot:
        assert robot.tcp_pose().position_mm == pytest.approx([480.0, 240.0, 300.0])
        assert not robot.digital_output(DO1)


def test_a_piece_left_outside_the_tray_is_reported(run_log: RunLog) -> None:
    """Si la cámara no ve la pieza en la bandeja después de soltarla, el intento no cuenta como logrado."""
    before = with_piece(empty_table(), PICK_CENTER_PX, skin_up=True)
    run_attempt([before, before], run_log)
    assert recorded(run_log)["outcome"] == "mal_colocada"

"""Calibración con una cámara sintética: se conoce la verdad y se comprueba que se recupera."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gt1.calibration import CalibrationStore, CameraModelProjector, HomographyProjector
from gt1.calibration.board import BoardDetection, CharucoBoard
from gt1.calibration.extrinsics import TouchPoint, fit_camera_pose, fit_homography, locate_board
from gt1.calibration.intrinsics import Intrinsics
from gt1.config.schemas import CharucoBoardConfig
from gt1.geometry import FloatArray, homogeneous, invert_transform, rotation_about_z, transform_points

# Cámara a 900 mm sobre la mesa mirando hacia abajo (Z de la cámara = -Z de la base).
BASE_FROM_CAMERA = homogeneous(np.diag([1.0, -1.0, -1.0]) @ rotation_about_z(5.0), (500.0, 450.0, 790.0))
INTRINSICS = Intrinsics(
    camera_matrix=np.array([[1800.0, 0.0, 700.0], [0.0, 1800.0, 520.0], [0.0, 0.0, 1.0]]),
    distortion=np.array([-0.2, 0.09, 0.0, 0.0, 0.0]),
    image_size_px=(1392, 1040),
    rms_px=0.1,
)
BOARD = CharucoBoard(
    CharucoBoardConfig(
        squares_x=9, squares_y=6, square_mm=35.0, marker_mm=26.0, dictionary="DICT_5X5_100", thickness_mm=3.0
    )
)
BASE_FROM_BOARD = homogeneous(rotation_about_z(20.0), (380.0, 330.0, -107.0))


def synthetic_detection() -> BoardDetection:
    ids = np.arange(BOARD.corner_count, dtype=np.int32)
    board_points = BOARD.corner_points_mm(ids)
    camera_points = transform_points(
        invert_transform(BASE_FROM_CAMERA), transform_points(BASE_FROM_BOARD, board_points)
    )
    return BoardDetection(
        corner_ids=ids, pixels_px=INTRINSICS.project(camera_points), board_points_mm=board_points
    )


def touches(noise_mm: float) -> list[TouchPoint]:
    rng = np.random.default_rng(0)
    result: list[TouchPoint] = []
    for corner_id in BOARD.touch_corner_ids().tolist():
        board_point = BOARD.corner_points_mm(np.array([corner_id], dtype=np.int32))[0]
        base_point = transform_points(BASE_FROM_BOARD, board_point)[0] + rng.normal(0.0, noise_mm, 3)
        result.append(TouchPoint(corner_id=corner_id, board_point_mm=board_point, base_point_mm=base_point))
    return result


def points_on_table(z_mm: float) -> FloatArray:
    xs, ys = np.meshgrid(np.linspace(300, 700, 5), np.linspace(250, 650, 5))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z_mm)])


@pytest.mark.parametrize("z_mm", [-110.0, -40.0])
def test_camera_model_projector_round_trip(z_mm: float) -> None:
    projector = CameraModelProjector(INTRINSICS, BASE_FROM_CAMERA)
    points = points_on_table(z_mm)
    recovered = projector.pixels_to_base(projector.base_to_pixels(points), z_mm)
    assert np.allclose(recovered, points, atol=1e-3)


def test_camera_to_robot_calibration_recovers_the_camera_pose() -> None:
    board_in_base = locate_board(touches(noise_mm=0.0))
    assert board_in_base.touch_rms_mm < 1e-6
    calibration = fit_camera_pose(INTRINSICS, synthetic_detection(), board_in_base)
    assert calibration.reprojection_rms_px < 1e-3
    assert np.allclose(calibration.base_from_camera, BASE_FROM_CAMERA, atol=1e-3)


def test_homography_is_exact_on_its_plane_and_shifts_above_it() -> None:
    detection = synthetic_detection()
    calibration = fit_homography(INTRINSICS, detection, locate_board(touches(noise_mm=0.0)))
    assert calibration.plane_z_mm == pytest.approx(-107.0)
    assert calibration.fit_rms_mm < 1e-3
    homography = HomographyProjector(INTRINSICS, calibration.base_from_pixel, calibration.plane_z_mm)
    truth = CameraModelProjector(INTRINSICS, BASE_FROM_CAMERA)

    on_plane = points_on_table(-107.0)
    assert np.allclose(homography.pixels_to_base(truth.base_to_pixels(on_plane), -107.0), on_plane, atol=1e-2)
    # 70 mm más arriba (la parte de arriba de una pechuga) la homografía se equivoca;
    # es el error que corrige el modelo de cámara.
    above = points_on_table(-37.0)
    error_mm = np.linalg.norm(homography.pixels_to_base(truth.base_to_pixels(above), -37.0) - above, axis=1)
    assert error_mm.max() > 15.0


def test_calibration_store_round_trip(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path, "cam")
    store.save_intrinsics(INTRINSICS, view_count=12)
    board_in_base = locate_board(touches(noise_mm=0.2))
    store.save_camera_model(fit_camera_pose(INTRINSICS, synthetic_detection(), board_in_base))
    projector = store.load_projector("camera_model")
    points = points_on_table(-110.0)
    expected = CameraModelProjector(INTRINSICS, BASE_FROM_CAMERA).base_to_pixels(points)
    assert np.allclose(projector.base_to_pixels(points), expected, atol=1.0)

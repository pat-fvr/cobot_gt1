"""Relación cámara-robot a partir del tablero sobre la mesa y de esquinas tocadas con el pincho.

1. La cámara ve el tablero: cada esquina tiene un píxel y una posición en el tablero.
2. El robot toca con el pincho algunas esquinas: cada toque da la posición de
   esa esquina en la base. Con 3 o más toques no alineados se obtiene
   ``base_from_board`` (ajuste rígido).
3. Con eso, todas las esquinas vistas tienen posición en la base, y se ajusta
   cualquiera de los dos métodos (homografía o pose de la cámara).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from gt1.geometry import FloatArray, fit_rigid_transform, homogeneous, invert_transform, transform_points

from .boards import BoardDetection
from .intrinsics import Intrinsics


@dataclass(frozen=True, eq=False)
class TouchPoint:
    corner_id: int
    board_point_mm: FloatArray  # (3,) esquina en el sistema del tablero
    base_point_mm: FloatArray  # (3,) punta del pincho en la base al tocarla


@dataclass(frozen=True, eq=False)
class BoardInBase:
    base_from_board: FloatArray
    touch_errors_mm: FloatArray  # distancia de cada toque a su esquina ajustada

    @property
    def touch_rms_mm(self) -> float:
        return float(np.sqrt(np.mean(self.touch_errors_mm**2)))


@dataclass(frozen=True, eq=False)
class HomographyCalibration:
    base_from_pixel: FloatArray  # 3x3, sobre píxeles sin distorsión
    plane_z_mm: float
    fit_rms_mm: float  # error del ajuste sobre todas las esquinas vistas
    touch_rms_mm: float


@dataclass(frozen=True, eq=False)
class CameraModelCalibration:
    base_from_camera: FloatArray  # 4x4
    reprojection_rms_px: float
    touch_rms_mm: float


def locate_board(touches: Sequence[TouchPoint]) -> BoardInBase:
    board_points = np.array([touch.board_point_mm for touch in touches])
    base_points = np.array([touch.base_point_mm for touch in touches])
    base_from_board = fit_rigid_transform(board_points, base_points)
    errors = np.linalg.norm(transform_points(base_from_board, board_points) - base_points, axis=1)
    return BoardInBase(base_from_board=base_from_board, touch_errors_mm=errors)


def fit_homography(
    intrinsics: Intrinsics, detection: BoardDetection, board: BoardInBase
) -> HomographyCalibration:
    corners_base = transform_points(board.base_from_board, detection.board_points_mm)
    ideal_pixels = intrinsics.undistort(detection.pixels_px)
    base_from_pixel, _ = cv2.findHomography(ideal_pixels, corners_base[:, :2])
    if base_from_pixel is None:  # pyright: ignore[reportUnnecessaryComparison]  (los stubs no lo declaran)
        raise ValueError("No se pudo ajustar la homografía: ¿se ven suficientes esquinas del tablero?")
    homography = np.asarray(base_from_pixel, dtype=np.float64)
    fitted = cv2.perspectiveTransform(ideal_pixels.reshape(-1, 1, 2), homography)
    residuals = np.linalg.norm(
        np.asarray(fitted, dtype=np.float64).reshape(-1, 2) - corners_base[:, :2], axis=1
    )
    return HomographyCalibration(
        base_from_pixel=homography,
        plane_z_mm=float(corners_base[:, 2].mean()),
        fit_rms_mm=float(np.sqrt(np.mean(residuals**2))),
        touch_rms_mm=board.touch_rms_mm,
    )


def fit_camera_pose(
    intrinsics: Intrinsics, detection: BoardDetection, board: BoardInBase
) -> CameraModelCalibration:
    found, rotation_vector, translation = cv2.solvePnP(
        detection.board_points_mm, detection.pixels_px, intrinsics.camera_matrix, intrinsics.distortion
    )
    if not found:
        raise ValueError("No se pudo calcular la pose del tablero respecto de la cámara.")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    camera_from_board = homogeneous(
        np.asarray(rotation, dtype=np.float64), np.asarray(translation).reshape(3)
    )
    reprojected = intrinsics.project(transform_points(camera_from_board, detection.board_points_mm))
    residuals = np.linalg.norm(reprojected - detection.pixels_px, axis=1)
    return CameraModelCalibration(
        base_from_camera=board.base_from_board @ invert_transform(camera_from_board),
        reprojection_rms_px=float(np.sqrt(np.mean(residuals**2))),
        touch_rms_mm=board.touch_rms_mm,
    )

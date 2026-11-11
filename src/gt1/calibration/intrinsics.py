"""Modelo de la lente (intrínsecos): distancia focal, centro óptico y distorsión.

Es el primer paso de la calibración y lo usan los dos métodos para pasar de
píxeles a la base del robot. Se obtiene mostrando el tablero a la cámara en
varias posiciones e inclinaciones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike

from gt1.geometry import FloatArray, as_points

from .boards import BoardDetection

MIN_CORNERS_PER_VIEW = 6


@dataclass(frozen=True, eq=False)
class Intrinsics:
    camera_matrix: FloatArray  # 3x3
    distortion: FloatArray  # (k1, k2, p1, p2, k3)
    image_size_px: tuple[int, int]  # (ancho, alto)
    rms_px: float  # error de reproyección de la calibración

    def undistort(self, pixels_px: ArrayLike) -> FloatArray:
        """Píxeles medidos -> píxeles de una cámara ideal sin distorsión (misma matriz de cámara)."""
        points = as_points(pixels_px, 2).reshape(-1, 1, 2)
        ideal = cv2.undistortPoints(points, self.camera_matrix, self.distortion, P=self.camera_matrix)
        return np.asarray(ideal, dtype=np.float64).reshape(-1, 2)

    def normalized_rays(self, pixels_px: ArrayLike) -> FloatArray:
        """Dirección (x, y, 1) del rayo de cada píxel en el sistema de la cámara."""
        points = as_points(pixels_px, 2).reshape(-1, 1, 2)
        normalized = np.asarray(
            cv2.undistortPoints(points, self.camera_matrix, self.distortion), dtype=np.float64
        )
        return np.hstack([normalized.reshape(-1, 2), np.ones((len(points), 1))])

    def project(self, points_camera_mm: ArrayLike) -> FloatArray:
        """Puntos en el sistema de la cámara -> píxeles medidos (con distorsión)."""
        points = as_points(points_camera_mm, 3)
        pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), self.camera_matrix, self.distortion)
        return np.asarray(pixels, dtype=np.float64).reshape(-1, 2)

    def distort(self, ideal_pixels_px: ArrayLike) -> FloatArray:
        """Inversa de ``undistort``: píxeles ideales -> píxeles medidos."""
        ideal = as_points(ideal_pixels_px, 2)
        rays = np.hstack([ideal, np.ones((len(ideal), 1))]) @ np.linalg.inv(self.camera_matrix).T
        return self.project(rays)


def calibrate_intrinsics(views: Sequence[BoardDetection], image_size_px: tuple[int, int]) -> Intrinsics:
    """Calibra la lente con varias vistas del tablero."""
    usable = [view for view in views if len(view.corner_ids) >= MIN_CORNERS_PER_VIEW]
    if len(usable) < 3:
        raise ValueError(f"Se necesitan al menos 3 vistas con {MIN_CORNERS_PER_VIEW} o más esquinas.")
    object_points = [view.board_points_mm.astype(np.float32) for view in usable]
    image_points = [view.pixels_px.astype(np.float32).reshape(-1, 1, 2) for view in usable]
    rms_px, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size_px, np.eye(3), np.zeros(5)
    )
    return Intrinsics(
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion=np.asarray(distortion, dtype=np.float64).reshape(-1)[:5],
        image_size_px=image_size_px,
        rms_px=float(rms_px),
    )

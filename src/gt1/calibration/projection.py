"""Paso de píxeles de la imagen a puntos en la base del robot, y vuelta.

Dos métodos, intercambiables detrás de ``PixelProjector``:

* ``HomographyProjector`` (homografía plana): una tabla de conversión
  píxel -> (X, Y) ajustada sobre el plano donde estaba el tablero. Es exacta
  solo en ese plano: un punto más alto (la parte de arriba de una pieza) se
  ve desplazado, más cuanto más lejos del centro de la imagen.
* ``CameraModelProjector`` (modelo 3D de la cámara): dónde está y hacia dónde
  apunta la cámara respecto del robot. Cada píxel es un rayo, y se corta a la
  altura que se pida, sin ese desplazamiento.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
from numpy.typing import ArrayLike

from gt1.geometry import FloatArray, as_points, invert_transform, transform_points

from .intrinsics import Intrinsics


class PixelProjector(ABC):
    @abstractmethod
    def pixels_to_base(self, pixels_px: ArrayLike, z_mm: float) -> FloatArray:
        """Píxeles (N, 2) -> puntos (N, 3) en la base, sobre el plano horizontal Z = ``z_mm``."""

    @abstractmethod
    def base_to_pixels(self, points_mm: ArrayLike) -> FloatArray:
        """Puntos (N, 3) en la base -> píxeles (N, 2) de la imagen."""


class HomographyProjector(PixelProjector):
    """Homografía entre píxeles sin distorsión y el plano Z = ``plane_z_mm`` de la base.

    ``pixels_to_base`` no puede corregir la altura: devuelve el (X, Y) del
    plano de calibración con la Z pedida.
    """

    def __init__(self, intrinsics: Intrinsics, base_from_pixel: FloatArray, plane_z_mm: float) -> None:
        self._intrinsics = intrinsics
        self._base_from_pixel = base_from_pixel
        self._pixel_from_base = np.linalg.inv(base_from_pixel)
        self.plane_z_mm = plane_z_mm

    def pixels_to_base(self, pixels_px: ArrayLike, z_mm: float) -> FloatArray:
        ideal = self._intrinsics.undistort(pixels_px).reshape(-1, 1, 2)
        xy = np.asarray(cv2.perspectiveTransform(ideal, self._base_from_pixel), dtype=np.float64).reshape(
            -1, 2
        )
        return np.hstack([xy, np.full((len(xy), 1), z_mm)])

    def base_to_pixels(self, points_mm: ArrayLike) -> FloatArray:
        xy = as_points(points_mm, 3)[:, :2].reshape(-1, 1, 2)
        ideal = np.asarray(cv2.perspectiveTransform(xy, self._pixel_from_base), dtype=np.float64).reshape(
            -1, 2
        )
        return self._intrinsics.distort(ideal)


class CameraModelProjector(PixelProjector):
    """Intrínsecos + pose de la cámara en la base."""

    def __init__(self, intrinsics: Intrinsics, base_from_camera: FloatArray) -> None:
        self._intrinsics = intrinsics
        self._base_from_camera = base_from_camera
        self._camera_from_base = invert_transform(base_from_camera)

    def pixels_to_base(self, pixels_px: ArrayLike, z_mm: float) -> FloatArray:
        rays = self._intrinsics.normalized_rays(pixels_px) @ self._base_from_camera[:3, :3].T
        origin = self._base_from_camera[:3, 3]
        if np.any(np.abs(rays[:, 2]) < 1e-9):
            raise ValueError("Un rayo de la cámara es paralelo al plano pedido.")
        scale = (z_mm - origin[2]) / rays[:, 2]
        return origin + rays * scale[:, None]

    def base_to_pixels(self, points_mm: ArrayLike) -> FloatArray:
        return self._intrinsics.project(transform_points(self._camera_from_base, points_mm))

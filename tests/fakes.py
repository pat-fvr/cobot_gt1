"""Implementaciones mínimas de interfaces para los tests."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import ArrayLike

from gt1.calibration import PixelProjector
from gt1.domain import Frame
from gt1.domain.frame import Image
from gt1.geometry import FloatArray, as_points
from gt1.hardware.camera import Camera


class ScaleProjector(PixelProjector):
    """Cámara ideal mirando la mesa de frente: 1 px = ``mm_per_px`` mm, sin distorsión."""

    def __init__(self, mm_per_px: float) -> None:
        self._mm_per_px = mm_per_px

    def pixels_to_base(self, pixels_px: ArrayLike, z_mm: float) -> FloatArray:
        xy = as_points(pixels_px, 2) * self._mm_per_px
        return np.column_stack([xy, np.full(len(xy), z_mm)])

    def base_to_pixels(self, points_mm: ArrayLike) -> FloatArray:
        return as_points(points_mm, 3)[:, :2] / self._mm_per_px


class StaticCamera(Camera):
    """Entrega siempre la misma imagen; se puede declarar real o simulada."""

    def __init__(self, image: Image, simulated: bool) -> None:
        self._image = image
        self._simulated = simulated

    @property
    def camera_id(self) -> str:
        return "static"

    @property
    def is_simulated(self) -> bool:
        return self._simulated

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def capture(self) -> Frame:
        return Frame(image=self._image.copy(), captured_at_s=time.time())

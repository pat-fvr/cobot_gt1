"""Implementaciones mínimas de interfaces para los tests."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import cv2

from gt1.domain import Frame
from gt1.domain.frame import Image
from gt1.hardware.camera import Camera


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


def write_video(path: Path, images: Sequence[Image], fps: float) -> None:
    """Video MJPG con estas imágenes (BGR, todas del mismo tamaño)."""
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (width, height))
    for image in images:
        writer.write(image)
    writer.release()

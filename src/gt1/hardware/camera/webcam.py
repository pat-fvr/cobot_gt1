"""Cámara USB a través de OpenCV.

OpenCV no entrega el modelo ni el número de serie de una webcam: no se pueden verificar.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from gt1.config.schemas import WebcamCameraConfig
from gt1.domain import Frame

from .base import Camera

logger = logging.getLogger(__name__)


class WebcamCamera(Camera):
    def __init__(self, config: WebcamCameraConfig) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None

    @property
    def camera_id(self) -> str:
        return self._config.camera_id

    @property
    def is_simulated(self) -> bool:
        return False

    def connect(self) -> None:
        if self._capture is not None:
            return
        capture = cv2.VideoCapture(self._config.device_index)
        if not capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la webcam {self._config.device_index}.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width_px)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height_px)
        self._capture = capture
        for _ in range(self._config.warmup_frames):
            capture.read()
        logger.info("Webcam %d conectada.", self._config.device_index)

    def disconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def capture(self) -> Frame:
        if self._capture is None:
            raise RuntimeError("La webcam no está conectada.")
        ok, image = self._capture.read()
        if not ok:
            raise RuntimeError(f"La webcam {self._config.device_index} no entregó imagen.")
        return Frame(image=np.asarray(image, dtype=np.uint8), captured_at_s=time.time())

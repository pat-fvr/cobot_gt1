"""Cámara USB (webcam) a través de OpenCV con DirectShow.

La cámara se busca por su nombre en Windows (p. ej. "Logi C270 HD WebCam"),
no por índice: los índices cambian al conectar o desconectar cámaras. Si no
está, el error lista las cámaras conectadas.

Se usa DirectShow y no Media Foundation: con Media Foundation abrir la C270
tarda minutos, salvo que OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0 esté
fijada antes de importar cv2 (medido en octubre de 2026, OpenCV 5.0).
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras

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
        index = self._find_index()
        for attempt in range(1, self._config.open_attempts + 1):
            capture = self._open(index)
            if self._received_first_frames(capture):
                self._capture = capture
                break
            capture.release()
            logger.warning(
                "La webcam no entregó imágenes en %g s (intento %d de %d).",
                self._config.first_frame_timeout_s,
                attempt,
                self._config.open_attempts,
            )
        else:
            raise RuntimeError(
                f"La webcam '{self._config.device_name}' no entregó imágenes en {self._config.open_attempts} "
                "intentos: desconectarla, volver a conectarla y revisar que ningún otro programa la use."
            )
        self._check_size(self.capture())
        logger.info("Webcam '%s' conectada (índice %d).", self._config.device_name, index)

    def disconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def capture(self) -> Frame:
        if self._capture is None:
            raise RuntimeError("La webcam no está conectada.")
        ok, image = self._capture.read()
        if not ok:
            raise RuntimeError(f"La webcam '{self._config.device_name}' no entregó imagen.")
        return Frame(image=np.asarray(image, dtype=np.uint8), captured_at_s=time.time())

    def _open(self, index: int) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la webcam '{self._config.device_name}' (índice {index}).")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width_px)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height_px)
        # El formato va después de la resolución: al revés, la C270 deja de entregar imágenes.
        if not capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self._config.fourcc)):
            logger.warning("La webcam no aceptó el formato %s.", self._config.fourcc)
        for name, value in self._config.capture_properties.items():
            if not capture.set(_property_id(name), value):
                logger.warning("La webcam no aceptó %s = %s.", name, value)
        return capture

    def _received_first_frames(self, capture: cv2.VideoCapture) -> bool:
        """Descarta ``warmup_frames`` imágenes; False si no llegan en el plazo.

        Por DirectShow, la C270 a veces (~1 de cada 10 aperturas, medido) no entrega
        imágenes después de negociar el formato: se cierra y se vuelve a abrir.
        """
        deadline = time.monotonic() + self._config.first_frame_timeout_s
        received = 0
        while received < self._config.warmup_frames:
            ok, _ = capture.read()
            received += ok
            if time.monotonic() > deadline:
                return False
        return True

    def _find_index(self) -> int:
        cameras = enumerate_cameras(cv2.CAP_DSHOW)
        for camera in cameras:
            if camera.name == self._config.device_name:
                return camera.index
        found = ", ".join(f"'{camera.name}'" for camera in cameras) or "ninguna"
        raise RuntimeError(
            f"No está conectada la webcam '{self._config.device_name}' "
            f"(cameras/{self._config.camera_id}.toml). Conectadas: {found}."
        )

    def _check_size(self, frame: Frame) -> None:
        if frame.size_px != (self._config.width_px, self._config.height_px):
            self.disconnect()
            raise RuntimeError(
                f"La webcam entrega {frame.size_px[0]} x {frame.size_px[1]} px y la configuración pide "
                f"{self._config.width_px} x {self._config.height_px}: corregir width_px / height_px."
            )


def _property_id(name: str) -> int:
    """'EXPOSURE' -> cv2.CAP_PROP_EXPOSURE."""
    attribute = f"CAP_PROP_{name}"
    if not hasattr(cv2, attribute):
        raise ValueError(f"Propiedad de cámara desconocida: {name} (no existe cv2.{attribute}).")
    property_id: int = getattr(cv2, attribute)
    return property_id

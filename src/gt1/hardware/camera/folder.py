"""Imágenes guardadas en una carpeta, en orden alfabético: para probar la percepción sin cámara."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from gt1 import paths
from gt1.config.schemas import FolderCameraConfig
from gt1.domain import Frame

from .base import Camera

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class FolderCamera(Camera):
    def __init__(self, config: FolderCameraConfig) -> None:
        self._config = config
        self._files: list[str] = []
        self._next_index = 0

    @property
    def camera_id(self) -> str:
        return self._config.camera_id

    @property
    def is_simulated(self) -> bool:
        return True

    def connect(self) -> None:
        directory = paths.resolve(self._config.directory)
        if not directory.is_dir():
            raise RuntimeError(f"No existe la carpeta de imágenes {directory}.")
        self._files = sorted(
            str(path) for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self._files:
            raise RuntimeError(f"La carpeta {directory} no tiene imágenes.")
        self._next_index = 0
        logger.info("Cámara de carpeta: %d imágenes en %s.", len(self._files), directory)

    def disconnect(self) -> None:
        self._files = []

    def capture(self) -> Frame:
        if not self._files:
            raise RuntimeError("La cámara de carpeta no está conectada.")
        if self._next_index >= len(self._files):
            if not self._config.loop:
                raise RuntimeError("No quedan imágenes en la carpeta.")
            self._next_index = 0
        path = self._files[self._next_index]
        self._next_index += 1
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"No se pudo leer la imagen {path}.")
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        logger.debug("Imagen %s", path)
        return Frame(image=np.asarray(image, dtype=np.uint8), captured_at_s=time.time())

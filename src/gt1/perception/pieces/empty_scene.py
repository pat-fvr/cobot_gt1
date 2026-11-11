"""Imagen de la escena vacía de cada cámara, para el detector por referencia: ``references/<camera_id>/``.

Se guarda el promedio de varias imágenes: el ruido del sensor baja con la raíz
del número de imágenes, y lo que queda en la referencia no se confunde con una pieza.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from gt1.domain.frame import Image

EMPTY_SCENE_FILE = "empty_scene.png"


class EmptySceneMissingError(Exception):
    """La cámara no tiene imagen de la escena vacía."""


class EmptySceneStore:
    def __init__(self, references_dir: Path, camera_id: str) -> None:
        self.path = references_dir / camera_id / EMPTY_SCENE_FILE
        self.camera_id = camera_id

    def save(self, image: Image) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.path), image):
            raise OSError(f"No se pudo guardar la escena vacía en {self.path}.")
        return self.path

    def load(self) -> Image:
        image = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise EmptySceneMissingError(
                f"No hay imagen de la escena vacía de la cámara {self.camera_id} ({self.path}): "
                "capturarla con gt1_tools/capture_empty_scene.py."
            )
        return np.asarray(image, dtype=np.uint8)


def average_images(images: Sequence[Image]) -> Image:
    """Promedio de imágenes del mismo tamaño, redondeado a 8 bits."""
    if not images:
        raise ValueError("No hay imágenes que promediar.")
    shapes = {image.shape for image in images}
    if len(shapes) > 1:
        raise ValueError(f"Las imágenes no son del mismo tamaño: {sorted(shapes)}.")
    total = np.zeros(images[0].shape, dtype=np.float64)
    for image in images:
        total += image
    return np.clip(np.round(total / len(images)), 0, 255).astype(np.uint8)

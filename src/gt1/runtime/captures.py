"""Capturas que dependen de cómo está el robot: con el robot quieto, y la escena vacía."""

from __future__ import annotations

import threading
from collections.abc import Callable

from gt1.domain import Frame
from gt1.domain.frame import Image
from gt1.hardware.camera import Camera
from gt1.perception.pieces.empty_scene import EmptySceneStore, average_images


def capture_still(camera: Camera, settle_s: float, stop: threading.Event) -> Frame | None:
    """Imagen tomada ``settle_s`` después de que el robot llegó (sin vibración); None si se detuvo antes.

    Se llama apenas termina el movimiento: la imagen es posterior a la llamada, y la espera cubre la
    que estaba en camino y la vibración del robot al frenar.
    """
    if stop.wait(settle_s):
        return None
    return camera.capture()


def capture_empty_scene(
    camera: Camera,
    store: EmptySceneStore,
    frames_to_average: int,
    stop: threading.Event,
    report: Callable[[str], None],
) -> Image | None:
    """Promedia ``frames_to_average`` imágenes y las guarda como la escena vacía; None si se detuvo antes."""
    images: list[Image] = []
    while len(images) < frames_to_average:
        if stop.is_set():
            return None
        report(f"Capturando la escena vacia {len(images) + 1}/{frames_to_average}…")
        images.append(camera.capture().image)
    empty_scene = average_images(images)
    store.save(empty_scene)
    return empty_scene

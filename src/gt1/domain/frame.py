"""Una captura de la cámara."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


@dataclass(frozen=True, eq=False)
class Frame:
    """Imagen capturada: ``(alto, ancho)`` si es monocromática, ``(alto, ancho, 3)`` BGR si es color.

    ``captured_at_s`` es la hora del equipo (``time.time()``) al recibir la
    imagen; sirve para comprobar que se tomó después de que el robot llegó.
    """

    image: Image
    captured_at_s: float

    @property
    def is_color(self) -> bool:
        return self.image.ndim == 3

    @property
    def size_px(self) -> tuple[int, int]:
        """(ancho, alto) en píxeles."""
        return int(self.image.shape[1]), int(self.image.shape[0])

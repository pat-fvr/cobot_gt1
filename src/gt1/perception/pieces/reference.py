"""Piezas sobre una escena conocida: diferencia con una imagen de la escena vacía.

La mesa no necesita ser de un color ni estar iluminada pareja: la luz
despareja, las manchas, los pliegues y los reflejos fijos del film están
también en la imagen de referencia y se cancelan. Sirve donde el umbral sobre
el brillo falla, como el lado del hueso (carne oscura) sobre una mesa oscura.

La diferencia es relativa, ``|imagen / referencia - 1|`` en %, para que un
mismo contraste valga igual en la parte oscura y en la clara de la mesa. Antes,
el cociente se divide por su mediana en la zona: compensa un cambio parejo de
luz o de exposición (supone que las piezas cubren menos de la mitad de la
zona). En color se toma el canal que más cambió.

Los reflejos del film no son fijos: cualquier roce los mueve. Los píxeles
muy brillantes y angostos, en la imagen o en la escena vacía, no cuentan como
cambio (una pieza clara es ancha y sí cuenta), y los
destellos tenues sueltos se borran antes de juntar los trozos de cada pieza.

Las piezas se miden en la base del robot (``contours.py``).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.calibration import PixelProjector
from gt1.config.schemas import ReferencePieceDetectorConfig
from gt1.domain import Frame, Piece, Zone
from gt1.domain.frame import Image
from gt1.geometry import FloatArray
from gt1.perception.image_ops import Mask, polygon_mask, zone_polygon_px

from .base import PieceDetector
from .contours import close_mask, open_mask, pieces_in_mask

# Bajo este brillo de la referencia, el cociente amplifica el ruido del sensor.
MIN_REFERENCE_LEVEL = 8.0
# La mediana del cociente se calcula con uno de cada tantos píxeles por lado: sobra para un valor parejo.
MEDIAN_SAMPLE_STEP_PX = 4


class ReferencePieceDetector(PieceDetector):
    def __init__(
        self, config: ReferencePieceDetectorConfig, projector: PixelProjector, empty_scene: Image
    ) -> None:
        self._config = config
        self._projector = projector
        self._reference = np.maximum(self._blurred(empty_scene), MIN_REFERENCE_LEVEL)
        self._empty_scene_glints = self._glint_pixels(empty_scene)

    def detect(self, frame: Frame, zone: Zone) -> list[Piece]:
        zone_mask = polygon_mask(frame.image.shape, zone_polygon_px(zone, self._projector))
        pieces_mask = self._segment(frame.image, zone_mask)
        return pieces_in_mask(pieces_mask, zone_mask, zone, self._projector, self._config)

    def foreground_mask(self, frame: Frame, zone: Zone) -> Mask:
        zone_mask = polygon_mask(frame.image.shape, zone_polygon_px(zone, self._projector))
        return self._segment(frame.image, zone_mask)

    def difference_pct(self, image: Image, zone_mask: Mask) -> FloatArray:
        """Cuánto cambió cada píxel respecto de la escena vacía, en % (0 = igual)."""
        if image.shape != self._reference.shape:
            raise ValueError(
                f"La imagen ({image.shape}) no tiene el tamaño de la escena vacía ({self._reference.shape}): "
                "¿cambió la resolución de la cámara? Capturarla de nuevo con "
                "gt1_tools/capture_empty_scene.py."
            )
        ratio = self._blurred(image) / self._reference
        ratio /= _median_in_zone(ratio, zone_mask)
        change_pct = 100.0 * np.abs(ratio - 1.0)
        return change_pct if change_pct.ndim == 2 else change_pct.max(axis=2)

    def _segment(self, image: Image, zone_mask: Mask) -> Mask:
        """Máscara 0/255 de lo que cambió respecto de la escena vacía, dentro de la zona."""
        changed = self.difference_pct(image, zone_mask) >= self._config.min_difference_pct
        changed &= ~self._glints(image)
        binary = np.where(changed & (zone_mask > 0), 255, 0).astype(np.uint8)
        # Sin los destellos sueltos primero, el cierre los junta en manchas pegadas a las piezas.
        without_specks = open_mask(binary, self._config.speck_kernel_px)
        # Del lado del hueso hay carne más oscura y grasa más clara que la mesa: entre ellas la
        # diferencia pasa por cero y deja líneas que parten la pieza. Se cierra antes de abrir,
        # o la apertura borra los trozos antes de que el cierre los junte.
        kernel_px = self._config.morphology_kernel_px
        return open_mask(close_mask(without_specks, kernel_px), kernel_px)

    def _glints(self, image: Image) -> NDArray[np.bool_]:
        """Reflejos en la imagen o en la escena vacía, con el halo que les deja el suavizado."""
        glints = (self._glint_pixels(image) | self._empty_scene_glints).astype(np.uint8)
        halo = np.ones((self._config.blur_kernel_px, self._config.blur_kernel_px), dtype=np.uint8)
        return np.asarray(cv2.dilate(glints, halo), dtype=np.uint8) > 0

    def _glint_pixels(self, image: Image) -> NDArray[np.bool_]:
        """Puntos y rayas brillantes más angostos que morphology_kernel_px."""
        brightest = image if image.ndim == 2 else image.max(axis=2)
        bright = np.where(brightest >= self._config.glint_min_level, 255, 0).astype(np.uint8)
        wide_bright = open_mask(bright, self._config.morphology_kernel_px)
        return (bright > 0) & (wide_bright == 0)

    def _blurred(self, image: Image) -> FloatArray:
        kernel = self._config.blur_kernel_px
        return np.asarray(cv2.GaussianBlur(image, (kernel, kernel), 0), dtype=np.float64)


def _median_in_zone(ratio: FloatArray, zone_mask: Mask) -> FloatArray:
    """Mediana del cociente dentro de la zona, por canal."""
    step = MEDIAN_SAMPLE_STEP_PX
    inside = zone_mask[::step, ::step] > 0
    if not np.any(inside):
        raise ValueError("La zona queda fuera de la imagen: revisar la calibración y cell.toml.")
    return np.asarray(np.median(ratio[::step, ::step][inside], axis=0), dtype=np.float64)

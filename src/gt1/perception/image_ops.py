"""Operaciones de imagen que comparten los detectores."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.calibration import PixelProjector
from gt1.domain import Zone
from gt1.domain.frame import Image

Mask = NDArray[np.uint8]


def to_gray(image: Image) -> Image:
    return image if image.ndim == 2 else np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def zone_polygon_px(zone: Zone, projector: PixelProjector) -> NDArray[np.int32]:
    """Vértices de la zona en la imagen."""
    corners_mm = np.column_stack([zone.polygon_mm, np.full(len(zone.polygon_mm), zone.surface_z_mm)])
    return np.round(projector.base_to_pixels(corners_mm)).astype(np.int32)


def full_view_zone(projector: PixelProjector, size_px: tuple[int, int], surface_z_mm: float) -> Zone:
    """Zona que cubre toda la imagen, sobre la superficie a ``surface_z_mm``."""
    width, height = size_px
    corners_px = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64
    )
    corners_mm = projector.pixels_to_base(corners_px, surface_z_mm)
    return Zone(name="full_view", polygon_mm=corners_mm[:, :2], surface_z_mm=surface_z_mm)


def polygon_mask(shape: tuple[int, ...], polygon_px: NDArray[np.int32]) -> Mask:
    """Máscara 0/255 del tamaño de la imagen con el polígono relleno."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_px.reshape(-1, 1, 2)], 255)
    return mask


def erode(mask: Mask, radius_px: int) -> Mask:
    """Achica la máscara ``radius_px``; lo que está fuera de la imagen cuenta como vacío."""
    if radius_px == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    # Por defecto OpenCV no erosiona desde el borde de la imagen: una zona que llega al borde no se
    # achicaba ahí y las piezas cortadas por el borde quedaban como tomables.
    eroded = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return np.asarray(eroded, dtype=np.uint8)

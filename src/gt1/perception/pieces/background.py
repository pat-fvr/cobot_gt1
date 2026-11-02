"""Piezas sobre un fondo de alto contraste: umbral, limpieza y contornos.

Tres formas de separar las piezas del fondo (``pieces_are``):

    brighter    más claras que el fondo (escala de grises)
    darker      más oscuras que el fondo
    different   de otro color que el fondo: distancia a la mediana de la zona, en el plano a*b*
                de Lab (solo el color, sin el brillo, así las sombras sobre el fondo no cuentan) si
                la cámara es color, y en gris si es mono. Sirve con fondos de color (p. ej. turquesa)
                y con piezas de zonas claras y oscuras a la vez (piel y costillas). Supone que
                las piezas cubren menos de la mitad de la zona.

Los contornos se proyectan a la base del robot antes de medirlos, así que
áreas, largos y ángulos están en mm y grados de la base, y los umbrales no
dependen de la cámara ni de su distancia a la mesa.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.calibration import PixelProjector
from gt1.config.schemas import BackgroundPieceDetectorConfig
from gt1.domain import Frame, Piece, PieceStatus, Zone
from gt1.domain.frame import Image
from gt1.geometry import wrap_axis_angle_deg
from gt1.perception.image_ops import Mask, erode, polygon_mask, to_gray, zone_polygon_px

from .base import PieceDetector

# Un contorno a menos de esto del borde de la zona se considera cortado por el borde.
ZONE_EDGE_MARGIN_PX = 2


class BackgroundPieceDetector(PieceDetector):
    def __init__(self, config: BackgroundPieceDetectorConfig, projector: PixelProjector) -> None:
        self._config = config
        self._projector = projector

    def detect(self, frame: Frame, zone: Zone) -> list[Piece]:
        zone_mask = polygon_mask(frame.image.shape, zone_polygon_px(zone, self._projector))
        pieces_mask = self._segment(frame.image, zone_mask)
        contours, _ = cv2.findContours(pieces_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inner_zone = erode(zone_mask, ZONE_EDGE_MARGIN_PX)
        contour_z_mm = zone.surface_z_mm + self._config.contour_height_mm
        pieces: list[Piece] = []
        for contour in contours:
            contour_px = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
            piece = self._measure(contour_px, contour_z_mm, inner_zone)
            if piece is not None:
                pieces.append(piece)
        return pieces

    def foreground_mask(self, frame: Frame, zone: Zone) -> Mask:
        zone_mask = polygon_mask(frame.image.shape, zone_polygon_px(zone, self._projector))
        return self._segment(frame.image, zone_mask)

    def _segment(self, image: Image, zone_mask: Mask) -> Mask:
        """Máscara 0/255 de lo que no es fondo, dentro de la zona."""
        kernel_size = self._config.blur_kernel_px
        blurred = np.asarray(cv2.GaussianBlur(image, (kernel_size, kernel_size), 0), dtype=np.uint8)
        if self._config.pieces_are == "different":
            score = _distance_to_background(blurred, zone_mask)
            mode = cv2.THRESH_BINARY
        else:
            score = to_gray(blurred)
            mode = cv2.THRESH_BINARY if self._config.pieces_are == "brighter" else cv2.THRESH_BINARY_INV
        level = self._threshold_level(score, zone_mask)
        _, binary = cv2.threshold(score, level, 255, mode)
        binary = cv2.bitwise_and(binary, zone_mask)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._config.morphology_kernel_px, self._config.morphology_kernel_px)
        )
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return np.asarray(cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel), dtype=np.uint8)

    def _threshold_level(self, gray: Image, zone_mask: Mask) -> float:
        if self._config.threshold.method == "fixed":
            return float(self._config.threshold.level)
        # Otsu solo con los píxeles de la zona: lo de afuera no debe mover el umbral.
        values = gray[zone_mask > 0].reshape(-1, 1)
        if values.size == 0:
            raise ValueError("La zona queda fuera de la imagen: revisar la calibración y cell.toml.")
        level, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(level)

    def _measure(self, contour_px: NDArray[np.int32], contour_z_mm: float, inner_zone: Mask) -> Piece | None:
        """Pieza con medidas en la base, o None si es una mancha más chica que una pieza."""
        outline_mm = self._projector.pixels_to_base(contour_px, contour_z_mm)
        outline_xy = outline_mm[:, :2].astype(np.float32)
        area_mm2 = float(cv2.contourArea(outline_xy))
        if area_mm2 < self._config.min_area_mm2:
            return None
        (center_x, center_y), (side_a, side_b), angle_deg = cv2.minAreaRect(outline_xy)
        if side_a < side_b:
            side_a, side_b, angle_deg = side_b, side_a, angle_deg + 90.0
        moments = cv2.moments(outline_xy)
        if moments["m00"] > 0:
            center_x, center_y = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        return Piece(
            contour_px=contour_px,
            center_mm=np.array([center_x, center_y, contour_z_mm]),
            axis_angle_deg=wrap_axis_angle_deg(float(angle_deg)),
            length_mm=float(side_a),
            width_mm=float(side_b),
            area_mm2=area_mm2,
            status=self._status(contour_px, area_mm2, inner_zone),
        )

    def _status(self, contour_px: NDArray[np.int32], area_mm2: float, inner_zone: Mask) -> PieceStatus:
        if area_mm2 > self._config.max_single_piece_area_mm2:
            return PieceStatus.CLUSTER
        if np.any(inner_zone[contour_px[:, 1], contour_px[:, 0]] == 0):
            return PieceStatus.CUT_BY_ZONE_EDGE
        return PieceStatus.OK


def _distance_to_background(image: Image, zone_mask: Mask) -> Image:
    """Qué tan distinto es cada píxel del color típico de la zona (0 = igual al fondo), 0..255."""
    if cv2.countNonZero(zone_mask) == 0:
        raise ValueError("La zona queda fuera de la imagen: revisar la calibración y cell.toml.")
    if image.ndim == 2:
        distance = np.abs(_offset_from_median(image, zone_mask))
    else:
        _, chroma_a, chroma_b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
        offset_a = _offset_from_median(np.asarray(chroma_a, dtype=np.uint8), zone_mask)
        offset_b = _offset_from_median(np.asarray(chroma_b, dtype=np.uint8), zone_mask)
        distance = np.asarray(cv2.magnitude(offset_a, offset_b), dtype=np.float32)
    return np.clip(distance, 0, 255).astype(np.uint8)


def _offset_from_median(channel: Image, mask: Mask) -> NDArray[np.float32]:
    """Cada valor del canal menos la mediana del canal dentro de la máscara."""
    return channel.astype(np.float32) - np.float32(_median_level(channel, mask))


def _median_level(channel: Image, mask: Mask) -> float:
    """Mediana de los valores (0..255) del canal dentro de la máscara.

    Se calcula con el histograma: es exacta para valores enteros y ~25 veces más
    rápida que ``np.median`` sobre la imagen completa (medido: 2 ms contra 53 ms en 1280 x 720).
    """
    histogram = np.asarray(cv2.calcHist([channel], [0], mask, [256], [0, 256]), dtype=np.float64)
    cumulative = np.cumsum(histogram.reshape(-1))
    return float(np.argmax(cumulative >= cumulative[-1] / 2))  # primer nivel que junta la mitad

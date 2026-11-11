"""Pechugas con la piel arriba, por su contorno: sin escena vacía y sin depender del color del fondo.

La pieza se reconoce por su borde, no por su brillo: sobre la mesa oscura la
piel es más clara que el fondo y sobre cartón blanco es más oscura, pero en
los dos casos el contorno es un salto de brillo. El salto se mide en
proporción (gradiente del logaritmo del brillo, en % por mm), así que vale lo
mismo en la parte iluminada que en la sombra.

Los contornos se buscan en la imagen reducida a ``search_scale`` (con los
kernels reducidos igual): la pieza mide cientos de píxeles y su contorno no
necesita más; buscarlo en la imagen completa tardaba unas seis veces más. La
textura y las medidas se toman en la imagen completa.

1. Se borran los reflejos del film y las rayas finas (apertura y cierre en
   gris con un kernel chico): son más angostos que el kernel; el contorno de
   la pieza, no.
2. Para cada nivel de contraste, de mayor a menor, los píxeles sobre el nivel
   son bordes: se unen los trozos cercanos, se rellena lo que encierran y una
   apertura quita los bordes sueltos. Cada región es una candidata.
3. Si la pieza es más clara que la mesa, se le quita su sombra: lo más
   oscuro que la mesa alrededor. La sombra cierra contorno junto con la pieza
   y corre su centro y su forma: con la acA1600 agrandaba el extremo de la
   punta hasta invertir el sentido de los lóbulos.
4. Una candidata es una pechuga con la piel arriba si mide como una (área,
   largo, ancho, convexa) y su superficie es lisa como la piel (la textura del
   clasificador por textura): el lado del hueso, el film o una bandeja tienen
   más textura. Se acepta en el primer nivel en que cumple: un contorno
   nítido cierra con el nivel alto y uno de poco contraste (piel clara sobre
   cartón blanco) recién con uno bajo; bajar el nivel para todas pegaría las
   nítidas a lo que tienen alrededor.
5. La orientación es la línea central entre los lóbulos (``midline.py``).

Las piezas pegadas entre sí no tienen la forma de una pechuga y no se
detectan: no se toman. Las medidas son en mm de la base (``contours.py``).
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.calibration import PixelProjector
from gt1.config.schemas import SkinOutlinePieceDetectorConfig
from gt1.domain import Frame, Piece, PieceStatus, Zone
from gt1.domain.frame import Image
from gt1.geometry import FloatArray
from gt1.perception.face.texture import surface_texture_pct_per_mm
from gt1.perception.image_ops import (
    Mask,
    erode,
    gradient_magnitude_per_px,
    polygon_mask,
    to_gray,
    zone_pixels_per_mm,
    zone_polygon_px,
)

from .base import PieceDetector
from .contours import ZONE_EDGE_MARGIN_PX, close_mask, fill_holes, measure_piece, open_mask
from .midline import lobes_direction_deg, midline_axis_deg

# Se suma al brillo antes del logaritmo: en lo casi negro el ruido del sensor no cuenta como contraste.
DARK_LEVEL_OFFSET = 10.0
# Una candidata que comparte más que esta parte de su superficie con una pieza ya aceptada (en un nivel
# más alto) es la misma pieza vista con más bordes alrededor.
MAX_SHARED_FRACTION = 0.2
# Sigma que OpenCV le da a un suavizado gaussiano por el tamaño de su kernel (documentación de
# getGaussianKernel): 0,3 * ((kernel - 1) / 2 - 1) + 0,8.
OPENCV_SIGMA_PER_KERNEL_PX = 0.3
OPENCV_MIN_SIGMA_PX = 0.8


@dataclasses.dataclass(frozen=True)
class _SearchImage:
    """La imagen reducida en que se buscan los contornos, con su brillo suavizado (en logaritmo) y la zona
    en ella."""

    gray: Image
    log_brightness: FloatArray
    scale: float
    zone_mask: Mask
    pixels_per_mm: float

    def to_full_px(self, outline_px: NDArray[np.int32]) -> NDArray[np.int32]:
        """El contorno en píxeles de la imagen completa (de centro de píxel a centro de píxel)."""
        return np.round((outline_px + 0.5) / self.scale - 0.5).astype(np.int32)


class SkinOutlinePieceDetector(PieceDetector):
    def __init__(self, config: SkinOutlinePieceDetectorConfig, projector: PixelProjector) -> None:
        self._config = config
        self._projector = projector

    def detect(self, frame: Frame, zone: Zone) -> list[Piece]:
        gray = to_gray(frame.image)
        search = self._search_image(gray, zone)
        contrast = self._contrast_pct_per_mm(search)
        inner_zone = erode(
            polygon_mask(gray.shape, zone_polygon_px(zone, self._projector)), ZONE_EDGE_MARGIN_PX
        )
        contour_z_mm = zone.surface_z_mm + self._config.contour_height_mm
        claimed = np.zeros(search.gray.shape, dtype=np.uint8)
        pieces: list[Piece] = []
        for level in self._config.edge_levels_pct_per_mm:
            for outline_px in self._outlines(contrast >= level, search.zone_mask):
                region = polygon_mask(search.gray.shape, outline_px)
                if _shared_fraction(region, claimed) > MAX_SHARED_FRACTION:
                    continue
                piece = self._skin_up_piece(gray, search, outline_px, contour_z_mm, inner_zone)
                if piece is not None:
                    pieces.append(piece)
                    claimed |= region
        return pieces

    def foreground_mask(self, frame: Frame, zone: Zone) -> Mask:
        """Los bordes con el nivel de contraste más bajo: lo que puede llegar a cerrar un contorno."""
        gray = to_gray(frame.image)
        search = self._search_image(gray, zone)
        edges = (self._contrast_pct_per_mm(search) >= self._config.edge_levels_pct_per_mm[-1]) & (
            search.zone_mask > 0
        )
        reduced = np.where(edges, 255, 0).astype(np.uint8)
        height, width = gray.shape[:2]
        return np.asarray(cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST), np.uint8)

    def _search_image(self, gray: Image, zone: Zone) -> _SearchImage:
        scale = self._config.search_scale
        zone_polygon = zone_polygon_px(zone, self._projector)
        height, width = gray.shape[:2]
        size = (max(round(width * scale), 1), max(round(height * scale), 1))
        reduced = np.asarray(cv2.resize(gray, size, interpolation=cv2.INTER_AREA), np.uint8)
        zone_in_search = np.round((zone_polygon + 0.5) * scale - 0.5).astype(np.int32)
        return _SearchImage(
            gray=reduced,
            log_brightness=self._smoothed_log_brightness(reduced),
            scale=scale,
            zone_mask=polygon_mask(reduced.shape, zone_in_search),
            pixels_per_mm=zone_pixels_per_mm(zone, zone_polygon) * scale,
        )

    def _kernel_px(self, full_image_px: int) -> int:
        """Un kernel de la configuración (en px de la imagen completa) en la imagen de búsqueda, impar."""
        return max(round(full_image_px * self._config.search_scale) // 2 * 2 + 1, 1)

    def _contrast_pct_per_mm(self, search: _SearchImage) -> FloatArray:
        """Contraste relativo de cada píxel: cuánto cambia el brillo, en % por mm de la mesa."""
        glint_px = self._kernel_px(self._config.glint_kernel_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (glint_px, glint_px))
        opened = cv2.morphologyEx(search.gray, cv2.MORPH_OPEN, kernel)
        without_glints = np.asarray(cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel), np.uint8)
        smoothed = self._smoothed(without_glints)
        log_brightness = np.log(smoothed + np.float32(DARK_LEVEL_OFFSET)).astype(np.float32)
        return 100.0 * gradient_magnitude_per_px(log_brightness) * search.pixels_per_mm

    def _outlines(self, edges: NDArray[np.bool_], zone_mask: Mask) -> list[NDArray[np.int32]]:
        """Contornos de las regiones que encierran los bordes y que tocan la zona."""
        binary = np.where(edges, 255, 0).astype(np.uint8)
        closed = close_mask(binary, self._kernel_px(self._config.outline_gap_kernel_px))
        regions = open_mask(fill_holes(closed), self._kernel_px(self._config.morphology_kernel_px))
        contours, _ = cv2.findContours(regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines = [np.asarray(contour, dtype=np.int32).reshape(-1, 2) for contour in contours]
        return [outline for outline in outlines if np.any(zone_mask[outline[:, 1], outline[:, 0]] > 0)]

    def _skin_up_piece(
        self,
        gray: Image,
        search: _SearchImage,
        candidate_px: NDArray[np.int32],
        contour_z_mm: float,
        inner_zone: Mask,
    ) -> Piece | None:
        """La pieza, con su línea central, si la candidata mide como una pechuga y es lisa como la piel."""
        outline_in_search = self._without_shadow(search.log_brightness, candidate_px)
        if outline_in_search is None:
            return None
        outline_px = search.to_full_px(outline_in_search)
        piece = measure_piece(outline_px, contour_z_mm, inner_zone, self._projector, self._config)
        if piece is None or piece.status == PieceStatus.CLUSTER or not self._has_breast_shape(piece):
            return None
        config = self._config
        texture = surface_texture_pct_per_mm(
            gray, piece, config.skin_smoothing_mm, config.skin_mask_erosion_mm
        )
        if texture > config.max_skin_texture_pct_per_mm:
            return None
        outline_mm = self._projector.pixels_to_base(outline_px, contour_z_mm)[:, :2]
        axis_deg = midline_axis_deg(outline_mm)
        return dataclasses.replace(
            piece, axis_angle_deg=axis_deg, lobes_direction_deg=lobes_direction_deg(outline_mm, axis_deg)
        )

    def _smoothed_log_brightness(self, gray: Image) -> FloatArray:
        """Logaritmo del brillo suavizado: una sombra resta lo mismo en la pieza y en el fondo."""
        return np.log(self._smoothed(gray) + np.float32(DARK_LEVEL_OFFSET)).astype(np.float64)

    def _smoothed(self, gray: Image) -> NDArray[np.float32]:
        """El suavizado de ``blur_kernel_px`` en la imagen completa, hecho en la de búsqueda: con el sigma
        reducido, no el kernel (un kernel de pocos px suaviza más, y la reducción ya promedia)."""
        blur_px = self._config.blur_kernel_px
        sigma_px = OPENCV_SIGMA_PER_KERNEL_PX * ((blur_px - 1) * 0.5 - 1) + OPENCV_MIN_SIGMA_PX
        smoothed = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma_px * self._config.search_scale)
        return np.asarray(smoothed, dtype=np.float32)

    def _without_shadow(
        self, log_brightness: FloatArray, candidate_px: NDArray[np.int32]
    ) -> NDArray[np.int32] | None:
        """El contorno de la candidata sin su sombra: lo más oscuro que el fondo, si la pieza es más clara.

        Sobre un fondo más claro que la pieza (cartón blanco) la sombra es tan oscura como la pieza y no
        se separa por el brillo: la candidata queda como está. El fondo es la mediana de una franja
        alrededor de la candidata y la pieza, la de su interior, en logaritmo del brillo. Se queda con la
        región más grande. Todo en la imagen de búsqueda.
        """
        config = self._config
        ring_px = max(round(config.background_ring_px * config.search_scale), 1)
        # Solo el rectángulo de la candidata y su franja: recorrer la imagen entera por cada candidata tarda.
        left, top, right, bottom = _bounding_box(candidate_px, ring_px, log_brightness.shape)
        window = log_brightness[top:bottom, left:right]
        offset_px = np.array([left, top], dtype=np.int32)
        region = polygon_mask(window.shape, candidate_px - offset_px) > 0
        ring_kernel = np.ones((2 * ring_px + 1,) * 2, dtype=np.uint8)
        ring = (cv2.dilate(region.astype(np.uint8), ring_kernel) > 0) & ~region
        background = float(np.median(window[ring]))
        piece_offset = float(np.median(window[region])) - background
        if piece_offset <= 0:
            return candidate_px
        kept = region & (window - background >= -config.shadow_level_fraction * piece_offset)
        kept_mask = open_mask(
            np.where(kept, 255, 0).astype(np.uint8), self._kernel_px(config.morphology_kernel_px)
        )
        contours, _ = cv2.findContours(kept_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        return np.asarray(largest, dtype=np.int32).reshape(-1, 2) + offset_px

    def _has_breast_shape(self, piece: Piece) -> bool:
        config = self._config
        (min_length, max_length), (min_width, max_width) = config.length_range_mm, config.width_range_mm
        return (
            min_length <= piece.length_mm <= max_length
            and min_width <= piece.width_mm <= max_width
            and self._solidity(piece) >= config.min_solidity
        )

    def _solidity(self, piece: Piece) -> float:
        hull_px = cv2.convexHull(piece.contour_px.reshape(-1, 1, 2))
        hull_area_px = cv2.contourArea(hull_px)
        area_px = cv2.contourArea(piece.contour_px.reshape(-1, 1, 2))
        return float(area_px / hull_area_px) if hull_area_px > 0 else 0.0


def _bounding_box(
    outline_px: NDArray[np.int32], margin_px: int, shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """Izquierda, arriba, derecha y abajo (exclusivos) del contorno más ``margin_px``, dentro de la imagen."""
    left, top = np.maximum(outline_px.min(axis=0) - margin_px - 1, 0)
    right, bottom = np.minimum(outline_px.max(axis=0) + margin_px + 2, [shape[1], shape[0]])
    return int(left), int(top), int(right), int(bottom)


def _shared_fraction(region: Mask, claimed: Mask) -> float:
    """Parte de la superficie de la región que ya es de otra pieza."""
    area = cv2.countNonZero(region)
    return cv2.countNonZero(cv2.bitwise_and(region, claimed)) / area if area else 1.0

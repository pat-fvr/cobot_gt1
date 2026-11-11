"""De la máscara de lo que no es fondo a piezas medidas en la base: lo común a los detectores.

Los contornos se proyectan a la base del robot antes de medirlos, así que
áreas, largos y ángulos están en mm y grados de la base, y los umbrales no
dependen de la cámara ni de su distancia a la mesa.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.calibration import PixelProjector
from gt1.config.schemas import PieceContourConfig
from gt1.domain import Piece, PieceStatus, Zone
from gt1.geometry import wrap_axis_angle_deg
from gt1.perception.image_ops import Mask, erode

# Un contorno a menos de esto del borde de la zona se considera cortado por el borde.
ZONE_EDGE_MARGIN_PX = 2


def clean_mask(binary: Mask, kernel_px: int) -> Mask:
    """Quita puntos sueltos (apertura) y después rellena huecos (cierre)."""
    return close_mask(open_mask(binary, kernel_px), kernel_px)


def open_mask(binary: Mask, kernel_px: int) -> Mask:
    """Apertura: borra lo más angosto que el kernel."""
    return _morphology(binary, cv2.MORPH_OPEN, kernel_px)


def close_mask(binary: Mask, kernel_px: int) -> Mask:
    """Cierre: rellena huecos y líneas más angostos que el kernel."""
    return _morphology(binary, cv2.MORPH_CLOSE, kernel_px)


def fill_holes(binary: Mask) -> Mask:
    """La máscara con lo que encierra relleno: lo que no se alcanza desde el borde de la imagen."""
    outside = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(outside, None, (0, 0), 255)
    enclosed = cv2.bitwise_not(outside[1:-1, 1:-1])
    return np.asarray(cv2.bitwise_or(binary, enclosed), dtype=np.uint8)


def _morphology(binary: Mask, operation: int, kernel_px: int) -> Mask:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    return np.asarray(cv2.morphologyEx(binary, operation, kernel), dtype=np.uint8)


def pieces_in_mask(
    pieces_mask: Mask, zone_mask: Mask, zone: Zone, projector: PixelProjector, config: PieceContourConfig
) -> list[Piece]:
    """Una pieza por cada contorno exterior de la máscara que no sea una mancha chica."""
    contours, _ = cv2.findContours(pieces_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inner_zone = erode(zone_mask, ZONE_EDGE_MARGIN_PX)
    contour_z_mm = zone.surface_z_mm + config.contour_height_mm
    pieces: list[Piece] = []
    for contour in contours:
        contour_px = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
        piece = measure_piece(contour_px, contour_z_mm, inner_zone, projector, config)
        if piece is not None:
            pieces.append(piece)
    return pieces


def measure_piece(
    contour_px: NDArray[np.int32],
    contour_z_mm: float,
    inner_zone: Mask,
    projector: PixelProjector,
    config: PieceContourConfig,
) -> Piece | None:
    """Pieza con medidas en la base, o None si es una mancha más chica que una pieza."""
    outline_mm = projector.pixels_to_base(contour_px, contour_z_mm)
    outline_xy = outline_mm[:, :2].astype(np.float32)
    area_mm2 = float(cv2.contourArea(outline_xy))
    if area_mm2 < config.min_area_mm2:
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
        status=piece_status(contour_px, area_mm2, inner_zone, config),
        lobes_direction_deg=None,
    )


def piece_status(
    contour_px: NDArray[np.int32], area_mm2: float, inner_zone: Mask, config: PieceContourConfig
) -> PieceStatus:
    if area_mm2 > config.max_single_piece_area_mm2:
        return PieceStatus.CLUSTER
    if np.any(inner_zone[contour_px[:, 1], contour_px[:, 0]] == 0):
        return PieceStatus.CUT_BY_ZONE_EDGE
    return PieceStatus.OK

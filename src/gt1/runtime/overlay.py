"""Dibujo sobre la imagen: zonas, piezas y su cara, bandejas y orientaciones. Solo presentación."""

from __future__ import annotations

import unicodedata

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.calibration import PixelProjector
from gt1.config.schemas import WideSide
from gt1.domain import Face, FaceResult, Piece, PieceStatus, TrayDetection
from gt1.domain.frame import Image
from gt1.geometry import FloatArray
from gt1.perception.image_ops import Mask
from gt1.perception.scene import Scene
from gt1.planning import grasp_tool_y_deg

Color = tuple[int, int, int]

ZONE_COLOR: Color = (255, 200, 0)
FACE_COLORS: dict[Face, Color] = {
    Face.SKIN: (80, 200, 80),
    Face.BONE: (40, 90, 230),
    Face.UNCERTAIN: (0, 220, 255),
}
NOT_PICKABLE_COLOR: Color = (150, 150, 150)
TRAY_COLOR: Color = (255, 160, 0)
GRIPPER_AXIS_COLOR: Color = (0, 220, 255)
LOBES_COLOR: Color = (255, 0, 255)
# Largo de la flecha hacia los lóbulos, como parte del largo de la pieza.
LOBES_ARROW_FRACTION = 0.45
# Ancho de la barra que marca la parte ancha de un gripper asimétrico, en mm sobre la pieza.
WIDE_SIDE_BAR_MM = 80.0


def ascii_text(text: str) -> str:
    """OpenCV no dibuja tildes ni eñes: 'Calibración' -> 'Calibracion'."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def to_color(image: Image) -> Image:
    """Copia BGR de la imagen, para dibujar en color aunque la cámara sea mono."""
    if image.ndim == 3:
        return image.copy()
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), dtype=np.uint8)


def draw_zone(canvas: Image, polygon_px: NDArray[np.int32], label: str) -> None:
    cv2.polylines(canvas, [polygon_px.reshape(-1, 1, 2)], True, ZONE_COLOR, 2, cv2.LINE_AA)
    x, y = polygon_px.min(axis=0)
    _label(canvas, label, (int(x) + 6, int(y) + 22), ZONE_COLOR)


def draw_piece(canvas: Image, piece: Piece, face: FaceResult | None) -> None:
    """Contorno de la pieza (gris si no se puede tomar) y una etiqueta con su cara y por qué no se toma."""
    text, color = _piece_label(piece, face)
    contour_color = color if piece.status == PieceStatus.OK else NOT_PICKABLE_COLOR
    cv2.drawContours(canvas, [piece.contour_px.reshape(-1, 1, 2)], -1, contour_color, 2, cv2.LINE_AA)
    center_x, center_y = piece.contour_px.mean(axis=0)
    _label(canvas, text, (int(center_x) - 40, int(center_y)), color)


def _piece_label(piece: Piece, face: FaceResult | None) -> tuple[str, Color]:
    status = "" if piece.status == PieceStatus.OK else piece.status.value
    if face is None:
        return (status, NOT_PICKABLE_COLOR) if status else ("sin clasificar", FACE_COLORS[Face.UNCERTAIN])
    return f"{face.face.value} ({face.margin:+.2f}) {status}".rstrip(), FACE_COLORS[face.face]


def draw_scene(scene: Scene) -> Image:
    """Copia en color de la imagen de la escena con la zona y cada pieza con su cara."""
    canvas = to_color(scene.frame.image)
    draw_zone(canvas, scene.zone_polygon_px, scene.zone.name)
    for detection in scene.detections:
        draw_piece(canvas, detection.piece, detection.face)
    return canvas


def draw_tray(canvas: Image, tray: TrayDetection, projector: PixelProjector, label: str = "") -> None:
    """El borde de la bandeja (a la altura del borde) y una etiqueta en su centro."""
    rim_px = np.round(projector.base_to_pixels(tray.corners_mm)).astype(np.int32)
    cv2.polylines(canvas, [rim_px.reshape(-1, 1, 2)], True, TRAY_COLOR, 2, cv2.LINE_AA)
    if label:
        center_x, center_y = rim_px.mean(axis=0)
        _label(canvas, label, (int(center_x) - 40, int(center_y)), TRAY_COLOR)


def draw_gripper_axis(canvas: Image, piece: Piece, wide_side: WideSide, projector: PixelProjector) -> None:
    """El eje Y del gripper como quedará sobre la pieza, con una barra en su parte ancha si la tiene."""
    tool_y_deg = grasp_tool_y_deg(piece, wide_side)
    half_length_mm = 0.5 * piece.length_mm
    start, end = _segment_px(piece.center_mm, tool_y_deg, -half_length_mm, half_length_mm, projector)
    cv2.line(canvas, start, end, GRIPPER_AXIS_COLOR, 3, cv2.LINE_AA)
    if wide_side == "symmetric":
        return
    wide_end_mm = half_length_mm if wide_side == "+y" else -half_length_mm
    wide_center_mm = piece.center_mm + wide_end_mm * _unit_vector(tool_y_deg)
    bar_start, bar_end = _segment_px(
        wide_center_mm, tool_y_deg + 90.0, -0.5 * WIDE_SIDE_BAR_MM, 0.5 * WIDE_SIDE_BAR_MM, projector
    )
    cv2.line(canvas, bar_start, bar_end, GRIPPER_AXIS_COLOR, 5, cv2.LINE_AA)


def draw_lobes_direction(canvas: Image, piece: Piece, projector: PixelProjector) -> None:
    """Flecha desde el centro de la pieza hacia sus lóbulos, a lo largo de la línea central."""
    if piece.lobes_direction_deg is not None:
        length_mm = LOBES_ARROW_FRACTION * piece.length_mm
        draw_direction(canvas, piece.center_mm, piece.lobes_direction_deg, length_mm, projector)


def draw_direction(
    canvas: Image, center_mm: FloatArray, direction_deg: float, length_mm: float, projector: PixelProjector
) -> None:
    """Flecha de ``length_mm`` desde ``center_mm`` hacia ``direction_deg`` (en el plano de la base)."""
    start, end = _segment_px(center_mm, direction_deg, 0.0, length_mm, projector)
    cv2.arrowedLine(canvas, start, end, LOBES_COLOR, 3, cv2.LINE_AA, tipLength=0.25)


def _segment_px(
    center_mm: FloatArray, angle_deg: float, start_mm: float, end_mm: float, projector: PixelProjector
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Tramo de la recta por ``center_mm`` a ``angle_deg``, entre ``start_mm`` y ``end_mm``, en píxeles."""
    ends_mm = center_mm + np.outer([start_mm, end_mm], _unit_vector(angle_deg))
    (start_x, start_y), (end_x, end_y) = projector.base_to_pixels(ends_mm)
    return (round(start_x), round(start_y)), (round(end_x), round(end_y))


def _unit_vector(angle_deg: float) -> FloatArray:
    """Vector unitario del plano de la base a ``angle_deg`` del eje X."""
    angle = np.radians(angle_deg)
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def tint_mask(image: Image, mask: Mask) -> Image:
    """Imagen oscurecida con lo que está en ``mask`` resaltado en magenta."""
    canvas = (to_color(image) * 0.4).astype(np.uint8)
    canvas[mask > 0] = (canvas[mask > 0] * 0.3 + np.array([255, 0, 255]) * 0.7).astype(np.uint8)
    return canvas


def text_panel(lines: list[str], size_px: tuple[int, int]) -> Image:
    """Imagen con líneas de texto, para programas sin cámara."""
    width, height = size_px
    canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            ascii_text(line),
            (24, 60 + 34 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _label(canvas: Image, text: str, origin: tuple[int, int], color: Color) -> None:
    text = ascii_text(text)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

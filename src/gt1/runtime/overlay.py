"""Dibujo sobre la imagen: zonas, piezas y su cara. Solo presentación."""

from __future__ import annotations

import unicodedata

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.domain import Face, FaceResult, Piece, PieceStatus
from gt1.domain.frame import Image

Color = tuple[int, int, int]

ZONE_COLOR: Color = (255, 200, 0)
FACE_COLORS: dict[Face, Color] = {
    Face.SKIN: (80, 200, 80),
    Face.BONE: (40, 90, 230),
    Face.UNCERTAIN: (0, 220, 255),
}
NOT_PICKABLE_COLOR: Color = (150, 150, 150)


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
    """Contorno de la pieza y una etiqueta con su cara (o por qué no se puede tomar)."""
    if piece.status != PieceStatus.OK:
        color, text = NOT_PICKABLE_COLOR, piece.status.value
    elif face is None:
        color, text = FACE_COLORS[Face.UNCERTAIN], "sin clasificar"
    else:
        color, text = FACE_COLORS[face.face], f"{face.face.value} ({face.margin:+.2f})"
    cv2.drawContours(canvas, [piece.contour_px.reshape(-1, 1, 2)], -1, color, 2, cv2.LINE_AA)
    center_x, center_y = piece.contour_px.mean(axis=0)
    _label(canvas, text, (int(center_x) - 40, int(center_y)), color)


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

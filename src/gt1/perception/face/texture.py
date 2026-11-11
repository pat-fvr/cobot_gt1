"""Cara por textura: la piel es lisa; el lado del hueso tiene costillas, cartílago y pliegues.

La textura es la variación media del brillo dentro de la pieza, en % del
brillo medio por mm de mesa:

    textura = promedio(|gradiente del brillo suavizado|) / promedio(brillo)

Dividir por el brillo medio la hace independiente de la exposición y la
ganancia; usar solo el brillo (no el color), del balance de blancos, y sirve
igual para cámaras mono y color. El suavizado y la erosión están en mm y el
gradiente se expresa por mm, así que el umbral no depende de la resolución de
la cámara ni de su distancia a la mesa. No necesita referencias.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.config.schemas import TextureFaceClassifierConfig
from gt1.domain import Face, FaceResult, Frame, Piece
from gt1.domain.frame import Image
from gt1.geometry import FloatArray
from gt1.perception.image_ops import erode, gradient_magnitude_per_px, polygon_mask, to_gray

from .base import FaceClassifier

# El suavizado se extiende hasta ~3 sigmas: el recorte deja ese margen para que el borde de la imagen
# recortada no aparezca como gradiente.
SMOOTHING_REACH_SIGMAS = 3


class TextureFaceClassifier(FaceClassifier):
    def __init__(self, config: TextureFaceClassifierConfig) -> None:
        self._config = config

    def classify(self, frame: Frame, piece: Piece) -> FaceResult:
        texture = surface_texture_pct_per_mm(
            frame.image, piece, self._config.smoothing_mm, self._config.mask_erosion_mm
        )
        margin = self._config.threshold_pct_per_mm - texture
        if abs(margin) < self._config.min_margin_pct_per_mm:
            face = Face.UNCERTAIN
        else:
            face = Face.SKIN if margin > 0 else Face.BONE
        return FaceResult(face, margin, {"texture_pct_per_mm": texture})


def surface_texture_pct_per_mm(image: Image, piece: Piece, smoothing_mm: float, erosion_mm: float) -> float:
    """Variación media del brillo dentro de la pieza, en % del brillo medio por mm."""
    px_per_mm = pixels_per_mm(piece)
    margin_px = round(SMOOTHING_REACH_SIGMAS * smoothing_mm * px_per_mm) + 1
    gray, contour_px = _crop_around(to_gray(image), piece, margin_px)
    mask = erode(polygon_mask(gray.shape, contour_px), round(erosion_mm * px_per_mm)) > 0
    if not np.any(mask):
        raise ValueError(
            f"La pieza ({piece.area_mm2:.0f} mm2) queda vacía al quitarle {erosion_mm} mm de borde: "
            "bajar mask_erosion_mm en la configuración del clasificador."
        )
    gradient_per_mm = _gradient_magnitude_per_px(gray, smoothing_mm * px_per_mm) * px_per_mm
    return 100.0 * float(gradient_per_mm[mask].mean()) / float(gray[mask].mean())


def pixels_per_mm(piece: Piece) -> float:
    """Escala local de la imagen en la pieza: su área en píxeles contra su área en mm de la base."""
    area_px = cv2.contourArea(piece.contour_px.reshape(-1, 1, 2).astype(np.float32))
    return float(np.sqrt(area_px / piece.area_mm2))


def _crop_around(gray: Image, piece: Piece, margin_px: int) -> tuple[Image, NDArray[np.int32]]:
    """Recorte de la imagen alrededor de la pieza, y el contorno en las coordenadas del recorte."""
    x, y, width, height = cv2.boundingRect(piece.contour_px.reshape(-1, 1, 2))
    left, top = max(x - margin_px, 0), max(y - margin_px, 0)
    right, bottom = x + width + margin_px, y + height + margin_px
    return gray[top:bottom, left:right], piece.contour_px - np.array([left, top], dtype=np.int32)


def _gradient_magnitude_per_px(gray: Image, smoothing_px: float) -> FloatArray:
    smoothed = np.asarray(cv2.GaussianBlur(gray.astype(np.float32), (0, 0), smoothing_px), dtype=np.float32)
    return gradient_magnitude_per_px(smoothed)

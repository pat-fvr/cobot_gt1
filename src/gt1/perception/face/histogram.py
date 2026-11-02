"""Cara por histograma: la piel es clara y pareja; el hueso tiene costillas rojas y más oscuras.

Se calcula el histograma de la pieza (solo dentro de su contorno) y se compara
con dos histogramas de referencia, uno de piel y uno de hueso, aprendidos con
piezas etiquetadas (ver ``references.py``). La distancia es Bhattacharyya
(0 = iguales, 1 = sin nada en común).

    gray   intensidad (1 canal): cámaras monocromáticas
    hsv    tono y saturación (2 canales): cámaras color, independiente del brillo
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from gt1.config.schemas import HistogramFaceClassifierConfig
from gt1.domain import Face, FaceResult, Frame, Piece
from gt1.domain.frame import Image
from gt1.geometry import FloatArray
from gt1.perception.image_ops import Mask, erode, polygon_mask, to_gray

from .base import FaceClassifier

ColorSpace = Literal["gray", "hsv"]


@dataclass(frozen=True, eq=False)
class FaceReferences:
    """Histogramas promedio de piel y de hueso, y con qué se calcularon."""

    color_space: ColorSpace
    bins_per_channel: int
    skin: FloatArray
    bone: FloatArray
    skin_samples: int
    bone_samples: int


def piece_histogram(image: Image, mask: Mask, color_space: ColorSpace, bins_per_channel: int) -> FloatArray:
    """Histograma normalizado (suma 1) de los píxeles de la máscara."""
    if not np.any(mask):
        raise ValueError("La máscara de la pieza está vacía (¿pieza demasiado chica para la erosión?).")
    if color_space == "gray":
        histogram = cv2.calcHist([to_gray(image)], [0], mask, [bins_per_channel], [0, 256])
    else:
        if image.ndim != 3:
            raise ValueError("color_space = 'hsv' necesita una cámara color; esta imagen es monocromática.")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], mask, [bins_per_channel, bins_per_channel], [0, 180, 0, 256])
    values = np.asarray(histogram, dtype=np.float64).reshape(-1)
    return values / values.sum()


def piece_mask(image: Image, piece: Piece, erosion_px: int) -> Mask:
    return erode(polygon_mask(image.shape, piece.contour_px), erosion_px)


def build_references(
    skin_histograms: Sequence[FloatArray],
    bone_histograms: Sequence[FloatArray],
    color_space: ColorSpace,
    bins_per_channel: int,
) -> FaceReferences:
    if not skin_histograms or not bone_histograms:
        raise ValueError("Hacen falta muestras de piel y de hueso para generar las referencias.")
    return FaceReferences(
        color_space=color_space,
        bins_per_channel=bins_per_channel,
        skin=np.mean(skin_histograms, axis=0),
        bone=np.mean(bone_histograms, axis=0),
        skin_samples=len(skin_histograms),
        bone_samples=len(bone_histograms),
    )


def compare(histogram: FloatArray, references: FaceReferences, min_margin: float) -> FaceResult:
    """Cara según a qué referencia se parece más el histograma."""
    distance_to_skin = _bhattacharyya(histogram, references.skin)
    distance_to_bone = _bhattacharyya(histogram, references.bone)
    margin = distance_to_bone - distance_to_skin
    if abs(margin) < min_margin:
        face = Face.UNCERTAIN
    else:
        face = Face.SKIN if margin > 0 else Face.BONE
    return FaceResult(
        face, margin, {"distance_to_skin": distance_to_skin, "distance_to_bone": distance_to_bone}
    )


class HistogramFaceClassifier(FaceClassifier):
    def __init__(self, config: HistogramFaceClassifierConfig, references: FaceReferences) -> None:
        if (references.color_space, references.bins_per_channel) != (
            config.color_space,
            config.bins_per_channel,
        ):
            raise ValueError(
                f"Las referencias se generaron con {references.color_space}/"
                f"{references.bins_per_channel} bins y "
                f"el clasificador usa {config.color_space}/{config.bins_per_channel}: vuelva a generarlas."
            )
        self._config = config
        self._references = references

    def classify(self, frame: Frame, piece: Piece) -> FaceResult:
        mask = piece_mask(frame.image, piece, self._config.mask_erosion_px)
        histogram = piece_histogram(
            frame.image, mask, self._config.color_space, self._config.bins_per_channel
        )
        return compare(histogram, self._references, self._config.min_margin)


def _bhattacharyya(first: FloatArray, second: FloatArray) -> float:
    return float(
        cv2.compareHist(first.astype(np.float32), second.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    )

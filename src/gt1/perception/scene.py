"""Escena: lo que vio la cámara en una imagen (la zona revisada y sus piezas, cada una con su cara).

``ScenePerception`` hace todo el recorrido desde la imagen: resuelve la zona,
detecta las piezas y clasifica su cara. Los programas (y más adelante el ciclo)
reciben la escena completa en vez de armar ese recorrido cada uno.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gt1.calibration import PixelProjector
from gt1.domain import FaceResult, Frame, Piece, PieceStatus, Zone
from gt1.perception.face import FaceClassifier
from gt1.perception.image_ops import Mask, zone_polygon_px
from gt1.perception.pieces import PieceDetector

# Un grupo mezcla varias piezas (y quizás las dos caras): su cara no significa nada.
UNCLASSIFIABLE = {PieceStatus.CLUSTER}


@dataclass(frozen=True, eq=False)
class Detection:
    """Una pieza de la escena y su cara; None si no se clasificó."""

    piece: Piece
    face: FaceResult | None


@dataclass(frozen=True, eq=False)
class Scene:
    """Lo que vio la cámara en ``frame`` dentro de ``zone``.

    ``perception_s`` es lo que tardó la percepción: zona, detección y cara.
    """

    frame: Frame
    zone: Zone
    zone_polygon_px: NDArray[np.int32]
    detections: tuple[Detection, ...]
    perception_s: float

    @property
    def pickable(self) -> tuple[Detection, ...]:
        """Las detecciones que el robot puede tomar tal como se ven, cualquiera sea su cara."""
        return tuple(detection for detection in self.detections if detection.piece.status == PieceStatus.OK)


class ScenePerception:
    """Detección de piezas y de su cara en una zona; sin clasificador, las piezas quedan sin cara."""

    def __init__(
        self,
        *,
        detector: PieceDetector,
        classifier: FaceClassifier | None,
        projector: PixelProjector,
        zone_for: Callable[[Frame], Zone],
    ) -> None:
        self._detector = detector
        self._classifier = classifier
        self._projector = projector
        self._zone_for = zone_for

    @property
    def classifies_faces(self) -> bool:
        return self._classifier is not None

    def perceive(self, frame: Frame) -> Scene:
        """Piezas de la zona y la cara de cada una, también de las del borde (que igual no se toman)."""
        started = time.perf_counter()
        zone = self._zone_for(frame)
        detections = tuple(
            Detection(piece, self._face(frame, piece)) for piece in self._detector.detect(frame, zone)
        )
        perception_s = time.perf_counter() - started
        return Scene(frame, zone, zone_polygon_px(zone, self._projector), detections, perception_s)

    def foreground_mask(self, frame: Frame) -> Mask:
        """Lo que el detector considera pieza en la zona (0/255), para ajustar sus parámetros."""
        return self._detector.foreground_mask(frame, self._zone_for(frame))

    def _face(self, frame: Frame, piece: Piece) -> FaceResult | None:
        if self._classifier is None or piece.status in UNCLASSIFIABLE:
            return None
        return self._classifier.classify(frame, piece)

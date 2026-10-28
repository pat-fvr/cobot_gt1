"""Interfaz de los clasificadores de cara (piel / hueso)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from gt1.domain import FaceResult, Frame, Piece


class FaceClassifier(ABC):
    @abstractmethod
    def classify(self, frame: Frame, piece: Piece) -> FaceResult:
        """Cara que la pieza muestra hacia la cámara."""

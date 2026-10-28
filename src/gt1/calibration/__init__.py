"""Calibración de la cámara: lente (intrínsecos) y relación con el robot (homografía o modelo de cámara)."""

from .projection import CameraModelProjector, HomographyProjector, PixelProjector
from .store import CalibrationMissingError, CalibrationStore

__all__ = [
    "CalibrationMissingError",
    "CalibrationStore",
    "CameraModelProjector",
    "HomographyProjector",
    "PixelProjector",
]

"""Tablero ChArUco: un tablero de ajedrez con marcadores ArUco en los cuadros blancos.

Los marcadores identifican cada esquina, así que el tablero se detecta aunque
se vea en parte, y el origen es siempre la misma esquina exterior.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.config.schemas import CharucoBoardConfig
from gt1.domain.frame import Image
from gt1.geometry import FloatArray

from .base import BoardDetection, CalibrationBoard, to_gray


class CharucoBoard(CalibrationBoard):
    def __init__(self, config: CharucoBoardConfig) -> None:
        if not hasattr(cv2.aruco, config.dictionary):
            raise ValueError(f"Diccionario ArUco desconocido: {config.dictionary}.")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.dictionary))
        self.config = config
        self._board = cv2.aruco.CharucoBoard(
            (config.squares_x, config.squares_y), config.square_mm, config.marker_mm, dictionary
        )
        self._detector = cv2.aruco.CharucoDetector(self._board)

    @property
    def corner_count(self) -> int:
        return (self.config.squares_x - 1) * (self.config.squares_y - 1)

    @property
    def thickness_mm(self) -> float:
        return self.config.thickness_mm

    def corner_points_mm(self, corner_ids: NDArray[np.int32]) -> FloatArray:
        """Posición de esquinas internas en el sistema del tablero."""
        corners = np.asarray(self._board.getChessboardCorners(), dtype=np.float64)
        return corners[corner_ids]

    def detect(self, image: Image) -> BoardDetection | None:
        gray = to_gray(image)
        corners, ids, _, _ = self._detector.detectBoard(gray)
        # Los stubs de OpenCV no declaran que ids es None cuando no se ve el tablero.
        if ids is None or len(ids) == 0:  # pyright: ignore[reportUnnecessaryComparison]
            return None
        corner_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        return BoardDetection(
            corner_ids=corner_ids,
            pixels_px=np.asarray(corners, dtype=np.float64).reshape(-1, 2),
            board_points_mm=self.corner_points_mm(corner_ids),
        )

    def render(self, pixels_per_mm: float) -> Image:
        margin_px = round(self.config.square_mm * pixels_per_mm)
        width_px = round(self.config.squares_x * self.config.square_mm * pixels_per_mm) + 2 * margin_px
        height_px = round(self.config.squares_y * self.config.square_mm * pixels_per_mm) + 2 * margin_px
        return np.asarray(
            self._board.generateImage((width_px, height_px), marginSize=margin_px), dtype=np.uint8
        )

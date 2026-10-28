"""Tablero ChArUco: un tablero de ajedrez con marcadores ArUco en los cuadros blancos.

Los marcadores identifican cada esquina, así que el tablero se detecta aunque
se vea en parte. El sistema del tablero tiene el origen en una esquina
exterior, X e Y a lo largo de los cuadros y Z = 0 en la superficie impresa.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.config.schemas import CharucoBoardConfig
from gt1.domain.frame import Image
from gt1.geometry import FloatArray


@dataclass(frozen=True, eq=False)
class BoardDetection:
    """Esquinas internas del tablero encontradas en una imagen."""

    corner_ids: NDArray[np.int32]  # (N,) número de cada esquina en el tablero
    pixels_px: FloatArray  # (N, 2) posición en la imagen
    board_points_mm: FloatArray  # (N, 3) posición en el sistema del tablero


class CharucoBoard:
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

    def corner_points_mm(self, corner_ids: NDArray[np.int32]) -> FloatArray:
        """Posición de esquinas internas en el sistema del tablero."""
        corners = np.asarray(self._board.getChessboardCorners(), dtype=np.float64)
        return corners[corner_ids]

    def detect(self, image: Image) -> BoardDetection | None:
        """Esquinas visibles, o None si no se ve el tablero."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    def touch_corner_ids(self) -> NDArray[np.int32]:
        """Esquinas que se tocan con el pincho: las cuatro extremas y una central, bien repartidas."""
        columns, rows = self.config.squares_x - 1, self.config.squares_y - 1
        last_column, last_row = columns - 1, rows - 1
        ids = [
            0,
            last_column,
            last_row * columns,
            last_row * columns + last_column,
            (rows // 2) * columns + columns // 2,
        ]
        return np.asarray(ids, dtype=np.int32)

    def render(self, pixels_per_mm: float) -> Image:
        """Imagen del tablero para imprimir a escala (con un margen blanco de un cuadro)."""
        margin_px = round(self.config.square_mm * pixels_per_mm)
        width_px = round(self.config.squares_x * self.config.square_mm * pixels_per_mm) + 2 * margin_px
        height_px = round(self.config.squares_y * self.config.square_mm * pixels_per_mm) + 2 * margin_px
        return np.asarray(
            self._board.generateImage((width_px, height_px), marginSize=margin_px), dtype=np.uint8
        )

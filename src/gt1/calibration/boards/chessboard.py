"""Tablero de ajedrez sin marcadores, como las placas de calibración Cognex.

Sin marcadores no se sabe qué esquina del tablero es cuál: el origen es la
primera esquina de lo que se ve (cambia de una imagen a otra) y el tablero
puede quedar girado o volteado. Para calibrar no importa, porque cada vista se
usa sola: la lente ajusta una pose por vista, y la relación cámara-robot toca
las esquinas de la misma detección que ve la cámara.
"""

from __future__ import annotations

import numpy as np

from gt1.config.schemas import ChessboardConfig
from gt1.domain.frame import Image

from .base import BoardDetection, CalibrationBoard, to_gray
from .corner_grid import CornerGrid, find_corner_grid


class Chessboard(CalibrationBoard):
    def __init__(self, config: ChessboardConfig) -> None:
        self.config = config
        self.inner_columns = config.squares_x - 1
        self.inner_rows = config.squares_y - 1

    @property
    def corner_count(self) -> int:
        return self.inner_columns * self.inner_rows

    @property
    def thickness_mm(self) -> float:
        return self.config.thickness_mm

    def detect(self, image: Image) -> BoardDetection | None:
        gray = to_gray(image)
        grid = find_corner_grid(gray, self.config.min_corner_contrast)
        if grid is None:
            return None
        return self._detection(fit_to_board(grid, self.inner_rows, self.inner_columns))

    def render(self, pixels_per_mm: float) -> Image:
        square_px = round(self.config.square_mm * pixels_per_mm)
        rows, columns = np.indices((self.config.squares_y, self.config.squares_x))
        squares = np.where((rows + columns) % 2 == 0, 0, 255).astype(np.uint8)
        board = np.kron(squares, np.ones((square_px, square_px), dtype=np.uint8))
        return np.pad(board, square_px, constant_values=255)

    def _detection(self, grid: CornerGrid) -> BoardDetection:
        indices = sorted(grid)
        rows = np.array([row for row, _ in indices])
        columns = np.array([column for _, column in indices])
        square_mm = self.config.square_mm
        return BoardDetection(
            corner_ids=(rows * self.inner_columns + columns).astype(np.int32),
            pixels_px=np.array([grid[index] for index in indices]),
            board_points_mm=np.column_stack(
                [columns * square_mm, rows * square_mm, np.zeros(len(indices))]
            ).astype(np.float64),
        )


def fit_to_board(grid: CornerGrid, inner_rows: int, inner_columns: int) -> CornerGrid:
    """Grilla dentro del tamaño del tablero, con origen en (0, 0).

    Si la grilla es más larga que el tablero (alguna fila suelta del fondo pasó
    los filtros), se quita la línea del borde con menos esquinas hasta que calce.
    """
    rows, columns = extent(grid)
    if (rows > columns) != (inner_rows > inner_columns):
        grid = {(column, row): pixel for (row, column), pixel in grid.items()}
    grid = trim_axis(grid, axis=0, size=inner_rows)
    grid = trim_axis(grid, axis=1, size=inner_columns)
    first_row = min(row for row, _ in grid)
    first_column = min(column for _, column in grid)
    return {(row - first_row, column - first_column): pixel for (row, column), pixel in grid.items()}


def extent(grid: CornerGrid) -> tuple[int, int]:
    """Filas y columnas que abarca la grilla."""
    rows = [row for row, _ in grid]
    columns = [column for _, column in grid]
    return max(rows) - min(rows) + 1, max(columns) - min(columns) + 1


def trim_axis(grid: CornerGrid, axis: int, size: int) -> CornerGrid:
    kept = dict(grid)
    while extent(kept)[axis] > size:
        first = min(index[axis] for index in kept)
        last = max(index[axis] for index in kept)
        drop = first if line_count(kept, axis, first) < line_count(kept, axis, last) else last
        kept = {index: pixel for index, pixel in kept.items() if index[axis] != drop}
    return kept


def line_count(grid: CornerGrid, axis: int, value: int) -> int:
    return sum(index[axis] == value for index in grid)

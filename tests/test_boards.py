"""Tableros de calibración: detección del tablero de ajedrez sin marcadores y esquinas a tocar."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gt1.calibration.boards import BoardDetection, touch_corner_ids
from gt1.calibration.boards.chessboard import Chessboard, fit_to_board
from gt1.calibration.intrinsics import calibrate_intrinsics
from gt1.config.schemas import ChessboardConfig
from gt1.domain.frame import Image
from gt1.geometry import FloatArray

DATA = Path(__file__).parent / "data"
COGNEX = Chessboard(
    ChessboardConfig(
        kind="chessboard",
        squares_x=28,
        squares_y=28,
        square_mm=20.0,
        thickness_mm=3.0,
        min_corner_contrast=20.0,
    )
)
PIXELS_PER_MM = 1.5  # cuadros de 30 px, como ve la Basler del banco la placa Cognex
IMAGE_SIZE_PX = (1392, 1040)
# Tablero (en px de la imagen dibujada) -> imagen de la cámara: perspectiva moderada.
CAMERA_FROM_RENDER = np.array([[0.95, 0.06, 180.0], [-0.04, 0.9, 80.0], [2e-5, 3e-5, 1.0]])


def with_t_fiducial(board: Image) -> Image:
    """Dibuja una T blanca al centro que tapa esquinas, como la de la placa Cognex."""
    marked = board.copy()
    center = board.shape[0] // 2
    square_px = round(COGNEX.config.square_mm * PIXELS_PER_MM)
    cv2.rectangle(
        marked, (center - 2 * square_px, center - 20), (center + 2 * square_px, center + 20), 255, -1
    )
    cv2.rectangle(marked, (center - 20, center), (center + 20, center + 3 * square_px), 255, -1)
    return marked


def camera_image(board: Image, camera_from_render: np.ndarray) -> Image:
    """El tablero visto por la cámara: con perspectiva, oscuro y algo desenfocado, como en el banco."""
    warped = cv2.warpPerspective(board, camera_from_render, IMAGE_SIZE_PX, borderValue=(60,))
    dimmed = (20 + warped.astype(np.float64) * 0.6).astype(np.uint8)
    return np.asarray(cv2.GaussianBlur(dimmed, (0, 0), 1.0), dtype=np.uint8)


def grid_residual_px(detection: BoardDetection) -> float:
    """Error de una homografía tablero -> imagen: sin distorsión, cero si la numeración es consistente."""
    board_xy = detection.board_points_mm[:, :2]
    homography, _ = cv2.findHomography(board_xy, detection.pixels_px)
    fitted = cv2.perspectiveTransform(board_xy.reshape(-1, 1, 2), homography).reshape(-1, 2)
    return float(np.linalg.norm(fitted - detection.pixels_px, axis=1).max())


def test_rendered_board_is_detected_complete() -> None:
    image = camera_image(COGNEX.render(PIXELS_PER_MM), CAMERA_FROM_RENDER)
    detection = COGNEX.detect(image)
    assert detection is not None
    assert len(detection.corner_ids) == COGNEX.corner_count
    assert grid_residual_px(detection) < 0.3


def test_grid_grows_around_the_fiducial() -> None:
    image = camera_image(with_t_fiducial(COGNEX.render(PIXELS_PER_MM)), CAMERA_FROM_RENDER)
    detection = COGNEX.detect(image)
    assert detection is not None
    # La T tapa unas 25 esquinas; las demás se encuentran con su número correcto.
    assert COGNEX.corner_count - 40 < len(detection.corner_ids) < COGNEX.corner_count
    assert grid_residual_px(detection) < 0.3


def test_partially_visible_board_is_detected() -> None:
    shifted = np.array([[1.0, 0.0, 700.0], [0.0, 1.0, 400.0], [0.0, 0.0, 1.0]]) @ CAMERA_FROM_RENDER
    detection = COGNEX.detect(camera_image(COGNEX.render(PIXELS_PER_MM), shifted))
    assert detection is not None
    assert 50 < len(detection.corner_ids) < COGNEX.corner_count / 2
    assert grid_residual_px(detection) < 0.3


def tilted_view(rotation_deg: tuple[float, float, float], center_mm: tuple[float, float, float]) -> Image:
    """El tablero girado ``rotation_deg`` (vector de rotación) con su centro en ``center_mm`` de la cámara."""
    render_pixels_per_mm = 3.0
    board = COGNEX.render(render_pixels_per_mm)
    half_mm = board.shape[0] / render_pixels_per_mm / 2
    render_to_board_mm = np.array(
        [
            [1 / render_pixels_per_mm, 0.0, -half_mm],
            [0.0, 1 / render_pixels_per_mm, -half_mm],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation, _ = cv2.Rodrigues(np.radians(np.array(rotation_deg)))
    board_to_camera = np.column_stack([rotation[:, 0], rotation[:, 1], np.array(center_mm)])
    return camera_image(board, SYNTHETIC_CAMERA_MATRIX @ board_to_camera @ render_to_board_mm)


SYNTHETIC_CAMERA_MATRIX: FloatArray = np.array([[1500.0, 0.0, 690.0], [0.0, 1500.0, 515.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("rotation_deg", [0.0, 20.0, 30.0, 45.0])
def test_board_turned_in_its_plane_keeps_its_numbering(rotation_deg: float) -> None:
    """Girada ~20-30°, OpenCV entregaba una subgrilla oblicua: esquinas reales mal numeradas."""
    detection = COGNEX.detect(tilted_view((0.0, 0.0, rotation_deg), (0.0, 0.0, 950.0)))
    assert detection is not None
    assert len(detection.corner_ids) > 600
    # Sin giro, los bordes calzan con los píxeles y el muestreo de la imagen sintética deja ~0,35 px.
    assert grid_residual_px(detection) < 0.5


def test_tilted_and_partial_views_calibrate_the_lens() -> None:
    """Los números de esquina cambian de una vista a otra, y aun así la lente se recupera.

    A 950 mm los cuadros miden ~30 px, como la placa sobre la mesa del banco; las vistas
    inclinadas y cercanas quedan en parte fuera de la imagen.
    """
    poses = [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 950.0)),
        ((25.0, 0.0, 0.0), (0.0, 0.0, 950.0)),
        ((-25.0, 0.0, 10.0), (0.0, 0.0, 950.0)),
        ((0.0, 25.0, 0.0), (0.0, 0.0, 950.0)),
        ((0.0, -25.0, -10.0), (0.0, 0.0, 950.0)),
        ((15.0, 15.0, 30.0), (200.0, 120.0, 800.0)),
        ((-15.0, 15.0, 0.0), (-200.0, -120.0, 800.0)),
    ]
    views = [COGNEX.detect(tilted_view(rotation, center)) for rotation, center in poses]
    assert all(view is not None for view in views)
    intrinsics = calibrate_intrinsics([view for view in views if view is not None], IMAGE_SIZE_PX)
    assert intrinsics.rms_px < 0.3
    assert np.allclose(intrinsics.camera_matrix, SYNTHETIC_CAMERA_MATRIX, rtol=0.01, atol=3.0)


def test_bench_image_of_the_cognex_plate_stays_within_the_board() -> None:
    """Imagen real del banco: letras y sombra del margen dieron esquinas falsas fuera del tablero."""
    image = cv2.imread(str(DATA / "cognex_28x28_basler.png"), cv2.IMREAD_GRAYSCALE)
    detection = COGNEX.detect(np.asarray(image, dtype=np.uint8))
    assert detection is not None
    rows, columns = np.divmod(detection.corner_ids, COGNEX.inner_columns)
    assert rows.max() < COGNEX.inner_rows and columns.max() < COGNEX.inner_columns
    assert len(detection.corner_ids) > 690


def test_fit_to_board_drops_a_stray_line_and_moves_the_origin() -> None:
    grid = {(row, column): np.zeros(2) for row in range(3) for column in range(4)}
    grid[(-1, 1)] = np.zeros(2)  # fila suelta del fondo
    fitted = fit_to_board(grid, inner_rows=3, inner_columns=4)
    assert sorted(fitted) == sorted((row, column) for row in range(3) for column in range(4))


def test_touch_corners_are_the_extremes_and_the_center_of_what_is_seen() -> None:
    rows, columns = np.divmod(np.arange(4 * 5), 5)
    detection = BoardDetection(
        corner_ids=(rows * 5 + columns).astype(np.int32),
        pixels_px=np.zeros((20, 2)),
        board_points_mm=np.column_stack([columns * 10.0, rows * 10.0, np.zeros(20)]),
    )
    assert touch_corner_ids(detection).tolist() == [0, 4, 15, 19, 7]

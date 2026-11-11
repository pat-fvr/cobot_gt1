"""Grilla de esquinas de un tablero de ajedrez, aunque se vea en parte o un fiducial tape algunas.

El detector de OpenCV necesita la grilla completa y sin huecos: con el fiducial
de una placa Cognex solo entrega el rectángulo más grande que no lo toca. Aquí
ese rectángulo es la semilla, y la grilla crece desde ella esquina por esquina:

1. La posición de cada vecina se predice con las esquinas ya encontradas a su
   alrededor (ajuste afín local: sigue la perspectiva y la distorsión).
2. Se refina con ``cornerSubPix``.
3. Se acepta solo si no se movió mucho y los cuatro cuadros que la rodean
   alternan claro/oscuro con los colores que les tocan en el tablero.

Las esquinas de la semilla pasan la misma prueba (partiendo de donde las dejó
OpenCV), así que todas terminan refinadas igual. La grilla se detiene sola en
el fiducial, en el borde del tablero y en el de la imagen. Al final se
descartan las esquinas que no calzan con sus vecinas (junto al fiducial, el
borde de una barra entra en la ventana de ``cornerSubPix`` y corre la esquina
1-2 px) y las que quedan sueltas. Los índices (fila, columna) son relativos a
la semilla.
"""

from __future__ import annotations

import cv2
import numpy as np

from gt1.domain.frame import Image
from gt1.geometry import FloatArray

GridIndex = tuple[int, int]  # (fila, columna)
CornerGrid = dict[GridIndex, FloatArray]  # esquina -> píxel (x, y)

# Rectángulo mínimo de esquinas que debe encontrar OpenCV para empezar.
SEED_CORNERS = (5, 5)
SEED_FLAGS = cv2.CALIB_CB_LARGER | cv2.CALIB_CB_EXHAUSTIVE
# Con el tablero girado ~15-30° respecto de la imagen, OpenCV a veces entrega una subgrilla
# oblicua (cada "fila" salta una fila del tablero): esquinas reales con la numeración
# equivocada. Ahí el centro de cada "cuadro" cae sobre una esquina y la semilla no pasa la
# prueba de colores; se reintenta con la imagen girada, donde OpenCV sí acierta.
SEED_ROTATIONS_DEG = (0.0, 15.0, -15.0, 30.0)
# Una semilla buena pasa la prueba casi entera; una oblicua, casi nada.
MIN_CONFIRMED_SEED_FRACTION = 0.5
# Esquinas a cada lado de una candidata que se usan para predecir su posición.
NEIGHBORHOOD = 2
# Medidas relativas al lado de un cuadro en la imagen.
SUBPIX_HALF_WINDOW = 0.3  # semiancho de la ventana de cornerSubPix
MAX_REFINE_SHIFT = 0.3  # cuánto puede moverse una esquina al refinarla
BRIGHTNESS_PATCH = 0.3  # lado del parche que mide el brillo de un cuadro
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
# Distancia máxima entre una esquina y lo que predicen sus vecinas. Con la placa Cognex en la Basler
# (cuadros de 30 px) el 99 % queda bajo 0,5 px y las corridas por el fiducial sobre 1,3 px.
MAX_NEIGHBOR_MISMATCH = 0.03
# Una esquina real tiene al menos dos vecinas (las del borde de la grilla, tres).
MIN_NEIGHBORS = 2
STEPS: tuple[GridIndex, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def find_corner_grid(gray: Image, min_contrast: float) -> CornerGrid | None:
    """Esquinas del tablero en la imagen, o None si no se ve ninguna zona de ajedrez."""
    for rotation_deg in SEED_ROTATIONS_DEG:
        seed = opencv_seed(gray, rotation_deg)
        if seed is None:
            continue
        parity = seed_parity(gray, seed)
        grid = confirmed_seed(gray, seed, parity, min_contrast)
        if len(grid) >= MIN_CONFIRMED_SEED_FRACTION * len(seed):
            grow_grid(gray, grid, parity, min_contrast)
            return drop_isolated(drop_mismatched(grid))
    return None


def opencv_seed(gray: Image, rotation_deg: float) -> CornerGrid | None:
    """Rectángulo de esquinas que encuentra OpenCV en la imagen girada, en píxeles de la original."""
    height, width = gray.shape[:2]
    original_to_rotated = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_deg, 1.0)
    rotated = cv2.warpAffine(gray, original_to_rotated, (width, height), borderMode=cv2.BORDER_REPLICATE)
    found, corners, meta = cv2.findChessboardCornersSBWithMeta(rotated, SEED_CORNERS, SEED_FLAGS)
    if not found:
        return None
    rows, columns = meta.shape[:2]
    rotated_to_original = cv2.invertAffineTransform(original_to_rotated)
    rotated_px = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    pixels = np.asarray(cv2.transform(rotated_px, rotated_to_original), dtype=np.float64)
    pixels = pixels.reshape(rows, columns, 2)
    return {(row, column): pixels[row, column] for row in range(rows) for column in range(columns)}


def seed_parity(gray: Image, seed: CornerGrid) -> int:
    """+1 si en la esquina (0, 0) los cuadros de la diagonal (+u+v, -u-v) son claros, si no -1.

    Se mide en el centro de la semilla, lejos de su borde (que puede tocar el fiducial).
    """
    last_row, last_column = max(seed)
    index = (last_row // 2, last_column // 2)
    frame = local_frame(seed, index)
    assert frame is not None, "La semilla de OpenCV es un rectángulo completo de al menos 5x5 esquinas."
    brightness = square_brightness(gray, seed[index], frame)
    diagonal_is_bright = brightness[0] + brightness[1] > brightness[2] + brightness[3]
    return (1 if diagonal_is_bright else -1) * checker_sign(index)


def confirmed_seed(gray: Image, seed: CornerGrid, parity: int, min_contrast: float) -> CornerGrid:
    """Esquinas de la semilla que pasan la prueba, ya refinadas."""
    confirmed: CornerGrid = {}
    for index, start_px in seed.items():
        frame = local_frame(seed, index)
        if frame is None:
            continue
        corner = confirmed_corner(gray, frame, index, start_px, parity=parity, min_contrast=min_contrast)
        if corner is not None:
            confirmed[index] = corner
    return confirmed


def grow_grid(gray: Image, grid: CornerGrid, parity: int, min_contrast: float) -> None:
    """Agrega a ``grid`` las vecinas que se confirman, por rondas, hasta que no aparece ninguna."""
    tried = set(grid)
    while True:
        candidates = sorted(neighbors_of(grid) - tried)
        tried.update(candidates)
        found = {
            index: corner
            for index in candidates
            if (corner := predicted_corner(gray, grid, index, parity, min_contrast)) is not None
        }
        if not found:
            return
        grid.update(found)


def predicted_corner(
    gray: Image, grid: CornerGrid, index: GridIndex, parity: int, min_contrast: float
) -> FloatArray | None:
    """La esquina ``index`` buscada donde la predicen sus vecinas, si se confirma."""
    frame = local_frame(grid, index)
    if frame is None:
        return None
    start_px = predict(frame, index)
    return confirmed_corner(gray, frame, index, start_px, parity=parity, min_contrast=min_contrast)


def confirmed_corner(
    gray: Image,
    frame: FloatArray,
    index: GridIndex,
    start_px: FloatArray,
    *,
    parity: int,
    min_contrast: float,
) -> FloatArray | None:
    """Refina desde ``start_px``; acepta si la esquina se movió poco y la rodean los colores que le tocan."""
    half_window = subpix_half_window(frame)
    if not inside_image(gray, start_px, half_window + 2):
        return None
    refined = refine_corner(gray, start_px, half_window)
    if np.linalg.norm(refined - start_px) > MAX_REFINE_SHIFT * square_size_px(frame):
        return None
    brightness = square_brightness(gray, refined, frame)
    diagonal_bright = parity * checker_sign(index) > 0
    if checker_contrast(brightness, diagonal_bright) < min_contrast:
        return None
    return refined


def predict(frame: FloatArray, index: GridIndex) -> FloatArray:
    return frame.T @ np.array([index[1], index[0], 1.0])


def local_frame(grid: CornerGrid, index: GridIndex) -> FloatArray | None:
    """Ajuste afín (columna, fila, 1) -> píxel con las esquinas cercanas: filas du, dv y origen.

    No usa la esquina misma, para poder comparar su posición con la predicción.
    None si las cercanas son menos de tres o están en línea.
    """
    row, column = index
    window = range(-NEIGHBORHOOD, NEIGHBORHOOD + 1)
    near = [
        (row + d_row, column + d_column)
        for d_row in window
        for d_column in window
        if (d_row, d_column) != (0, 0) and (row + d_row, column + d_column) in grid
    ]
    grid_points = np.array([[other[1], other[0], 1.0] for other in near])
    if len(near) < 3 or np.linalg.matrix_rank(grid_points[:, :2] - grid_points[:, :2].mean(axis=0)) < 2:
        return None
    pixels = np.array([grid[other] for other in near])
    frame, *_ = np.linalg.lstsq(grid_points, pixels, rcond=None)
    return np.asarray(frame, dtype=np.float64)


def subpix_half_window(frame: FloatArray) -> int:
    return max(2, round(SUBPIX_HALF_WINDOW * square_size_px(frame)))


def square_size_px(frame: FloatArray) -> float:
    return float(np.linalg.norm(frame[:2], axis=1).mean())


def inside_image(gray: Image, point_px: FloatArray, margin_px: int) -> bool:
    height, width = gray.shape[:2]
    x, y = point_px
    return margin_px <= x < width - margin_px and margin_px <= y < height - margin_px


def refine_corner(gray: Image, start_px: FloatArray, half_window: int) -> FloatArray:
    start = start_px.reshape(1, 1, 2).astype(np.float32)
    refined = cv2.cornerSubPix(gray, start, (half_window, half_window), (-1, -1), SUBPIX_CRITERIA)
    return np.asarray(refined, dtype=np.float64).reshape(2)


def square_brightness(gray: Image, corner_px: FloatArray, frame: FloatArray) -> FloatArray:
    """Brillo medio de los cuatro cuadros alrededor de una esquina: (+u+v, -u-v, +u-v, -u+v)."""
    du, dv = frame[0], frame[1]
    patch = max(3, round(BRIGHTNESS_PATCH * square_size_px(frame)))
    centers = [corner_px + 0.5 * (su * du + sv * dv) for su, sv in ((1, 1), (-1, -1), (1, -1), (-1, 1))]
    return np.array(
        [cv2.getRectSubPix(gray, (patch, patch), (float(x), float(y))).mean() for x, y in centers]
    )


def checker_contrast(brightness: FloatArray, diagonal_bright: bool) -> float:
    """Cuánto más claro es el cuadro claro más oscuro que el cuadro oscuro más claro."""
    diagonal, cross = brightness[:2], brightness[2:]
    bright, dark = (diagonal, cross) if diagonal_bright else (cross, diagonal)
    return float(bright.min() - dark.max())


def checker_sign(index: GridIndex) -> int:
    """Los colores alrededor de una esquina se invierten de una esquina a la siguiente."""
    return 1 if (index[0] + index[1]) % 2 == 0 else -1


def neighbors_of(grid: CornerGrid) -> set[GridIndex]:
    return {(row + d_row, column + d_column) for row, column in grid for d_row, d_column in STEPS}


def drop_mismatched(grid: CornerGrid) -> CornerGrid:
    """Quita las esquinas que se alejan de lo que predicen sus vecinas (sin contarse a sí mismas)."""
    return {index: pixel for index, pixel in grid.items() if matches_neighbors(grid, index)}


def matches_neighbors(grid: CornerGrid, index: GridIndex) -> bool:
    frame = local_frame(grid, index)
    if frame is None:
        return False
    mismatch_px = float(np.linalg.norm(grid[index] - predict(frame, index)))
    return mismatch_px <= MAX_NEIGHBOR_MISMATCH * square_size_px(frame)


def drop_isolated(grid: CornerGrid) -> CornerGrid:
    """Quita, hasta que no quede ninguna, las esquinas con menos de ``MIN_NEIGHBORS`` vecinas.

    Un parche del fondo que por casualidad parece una esquina (letras o sombra en
    el margen del tablero) queda colgando de la grilla con una sola vecina.
    """
    kept = dict(grid)
    while True:
        isolated = [index for index in kept if neighbor_count(kept, index) < MIN_NEIGHBORS]
        if not isolated:
            return kept
        for index in isolated:
            del kept[index]


def neighbor_count(grid: CornerGrid, index: GridIndex) -> int:
    return sum((index[0] + d_row, index[1] + d_column) in grid for d_row, d_column in STEPS)

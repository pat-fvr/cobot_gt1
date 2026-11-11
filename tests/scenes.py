"""Escenas sintéticas a 1 mm/px: mesa, pechugas con la piel arriba y bandejas, y los detectores para ellas.

La pechuga es un óvalo más ancho en el extremo de los lóbulos que en la punta,
con la superficie lisa de la piel. La bandeja son las líneas oscuras que ve
el detector de bandejas: el pliegue del fondo en los lados hacia la sombra y
el labio del borde en los otros. La escala nominal no tiene perspectiva: el
borde se dibuja en la mesa, con su tamaño.
"""

from __future__ import annotations

import cv2
import numpy as np

from gt1.config.schemas import RimAndFloorTrayDetectorConfig, SkinOutlinePieceDetectorConfig, TrayConfig
from gt1.domain.frame import Image

BREAST_LENGTH_MM = 220.0
BREAST_WIDTH_MM = 160.0
# Cuánto más ancho es el extremo de los lóbulos que el de la punta (fracción del ancho).
LOBES_WIDENING = 0.35
FLOOR_LINE_WIDTH_MM = 6
# Ancho del borde de la sombra de la cámara, en px (= mm).
SHADOW_PENUMBRA_PX = 40

# Los valores de config/perception/, con el contorno a la altura de la mesa (la escena es plana).
SKIN_DETECTOR = SkinOutlinePieceDetectorConfig(
    kind="skin_outline",
    search_scale=0.5,
    glint_kernel_px=13,
    blur_kernel_px=11,
    edge_levels_pct_per_mm=(3.0, 1.8, 1.2),
    outline_gap_kernel_px=21,
    morphology_kernel_px=41,
    background_ring_px=15,
    shadow_level_fraction=0.15,
    contour_height_mm=0.0,
    min_area_mm2=15000.0,
    max_single_piece_area_mm2=45000.0,
    length_range_mm=(150.0, 280.0),
    width_range_mm=(110.0, 220.0),
    min_solidity=0.9,
    skin_smoothing_mm=2.0,
    skin_mask_erosion_mm=5.0,
    max_skin_texture_pct_per_mm=3.3,
)
RIM_MM = (217.0, 167.0)
FLOOR_MM = (175.0, 122.0)
TRAY = TrayConfig(
    description="prueba",
    rim_length_mm=RIM_MM[0],
    rim_width_mm=RIM_MM[1],
    floor_length_mm=FLOOR_MM[0],
    floor_width_mm=FLOOR_MM[1],
    depth_mm=50.0,
    seal_flange_mm=8.0,
    min_clearance_to_flange_mm=3.0,
)
TRAY_DETECTOR = RimAndFloorTrayDetectorConfig(
    kind="rim_and_floor",
    grid_mm=2.0,
    angle_step_deg=2.0,
    coarse_grid_mm=6.0,
    coarse_angle_step_deg=8.0,
    glint_width_mm=10.0,
    dark_line_width_mm=24.0,
    shadow_toward_deg=70.0,
    side_tolerance_mm=4.0,
    side_length_fraction=0.6,
    neighbor_offset_mm=10.0,
    min_fold_contrast_pct=25.0,
    min_line_contrast_pct=15.0,
    min_wall_vs_table_pct=-10.5,
    max_floor_vs_table_pct=30.0,
)
# Lados de un rectángulo de cv2.boxPoints, por sus esquinas: cada uno une la esquina i con la i + 1.
ALL_SIDES = (0, 1, 2, 3)


def table(size_px: tuple[int, int], brightness: float, seed: int = 0) -> Image:
    """Mesa pareja con un poco de ruido de sensor."""
    width, height = size_px
    noise = np.random.default_rng(seed).normal(0.0, 2.0, (height, width))
    return np.clip(brightness + noise, 0, 255).astype(np.uint8)


def breast_outline_px(center_px: tuple[float, float], lobes_deg: float) -> np.ndarray:
    """Contorno (N, 2) de una pechuga con los lóbulos hacia ``lobes_deg`` (ángulo en la imagen)."""
    along = np.linspace(-0.5, 0.5, 200)
    half_width = 0.5 * BREAST_WIDTH_MM * np.sqrt(1.0 - (2.0 * along) ** 2) * (1.0 + LOBES_WIDENING * along)
    upper = np.column_stack([along * BREAST_LENGTH_MM, half_width])
    lower = np.column_stack([along[::-1] * BREAST_LENGTH_MM, -half_width[::-1]])
    local = np.vstack([upper, lower])
    angle = np.radians(lobes_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return local @ rotation.T + np.asarray(center_px)


def with_breast(image: Image, center_px: tuple[float, float], lobes_deg: float, brightness: float) -> Image:
    """La imagen con una pechuga lisa de ``brightness``."""
    scene = image.copy()
    outline = np.round(breast_outline_px(center_px, lobes_deg)).astype(np.int32)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [outline.reshape(-1, 1, 2)], 255)
    skin = np.clip(brightness + np.random.default_rng(1).normal(0.0, 2.0, image.shape), 0, 255)
    scene[mask > 0] = skin[mask > 0].astype(np.uint8)
    return scene


def with_tray(image: Image, center_px: tuple[float, float], axis_deg: float, darkening: float = 0.6) -> Image:
    """La imagen con una bandeja como la ve la cámara: el pliegue del fondo en los lados hacia la sombra y el
    labio del borde en los otros, ``darkening`` veces más oscuros."""
    shadow_sides = sides_toward(center_px, axis_deg, TRAY_DETECTOR.shadow_toward_deg)
    rim_sides = tuple(side for side in ALL_SIDES if side not in shadow_sides)
    scene = with_tray_lines(image, center_px, axis_deg, FLOOR_MM, sides=shadow_sides, darkening=darkening)
    return with_tray_lines(scene, center_px, axis_deg, RIM_MM, sides=rim_sides, darkening=darkening)


def sides_toward(center_px: tuple[float, float], axis_deg: float, direction_deg: float) -> tuple[int, ...]:
    """Los lados de ``cv2.boxPoints`` cuya normal hacia afuera está a menos de 90° de ``direction_deg``."""
    corners = cv2.boxPoints((center_px, FLOOR_MM, axis_deg))
    direction = np.array([np.cos(np.radians(direction_deg)), np.sin(np.radians(direction_deg))])
    return tuple(
        side
        for side in ALL_SIDES
        if float((0.5 * (corners[side] + corners[(side + 1) % 4]) - center_px) @ direction) > 0.0
    )


def with_tray_lines(
    image: Image,
    center_px: tuple[float, float],
    axis_deg: float,
    size_mm: tuple[float, float],
    *,
    sides: tuple[int, ...],
    darkening: float = 0.6,
) -> Image:
    """La imagen con ``sides`` de un rectángulo de ``size_mm`` (el borde o el fondo) como líneas oscuras."""
    corners = np.round(cv2.boxPoints((center_px, size_mm, axis_deg))).astype(np.int32)
    line = np.zeros(image.shape, dtype=np.uint8)
    for side in sides:
        start, end = corners[side], corners[(side + 1) % 4]
        cv2.line(line, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 255, FLOOR_LINE_WIDTH_MM)
    scene = image.astype(np.float64)
    scene[line > 0] *= darkening
    return np.clip(scene, 0, 255).astype(np.uint8)


def with_glints(image: Image, count: int, seed: int = 0) -> Image:
    """La imagen con reflejos del film: trazos claros y angostos, casi paralelos, como los del banco."""
    scene = image.copy()
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    for _ in range(count):
        x, y = rng.uniform(0, width), rng.uniform(0, height)
        length, angle = rng.uniform(15.0, 60.0), np.radians(rng.normal(10.0, 8.0))
        end = (round(x + length * np.cos(angle)), round(y + length * np.sin(angle)))
        cv2.line(scene, (round(x), round(y)), end, 250, int(rng.integers(1, 4)))
    return scene


def in_shadow(image: Image, from_column: int, factor: float) -> Image:
    """La imagen a ``factor`` del brillo desde ``from_column`` hacia la derecha, como la sombra de la cámara.

    El borde de la sombra es una rampa de ``SHADOW_PENUMBRA_PX``: con la luz difusa del banco no hay escalón
    (un escalón recto junto al labio de otra bandeja también arma un rectángulo).
    """
    columns = np.arange(image.shape[1], dtype=np.float64)
    into_shadow = np.clip((columns - from_column) / SHADOW_PENUMBRA_PX, 0.0, 1.0)
    brightness = 1.0 - (1.0 - factor) * into_shadow
    return np.clip(image.astype(np.float64) * brightness[None, :], 0, 255).astype(np.uint8)

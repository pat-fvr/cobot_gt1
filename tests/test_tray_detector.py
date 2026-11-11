"""Detector de bandejas por el labio del borde y el pliegue del fondo, en escenas sintéticas a 1 mm/px."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from gt1.calibration import NominalScaleProjector
from gt1.config.schemas import RimAndFloorTrayDetectorConfig
from gt1.domain import Frame, TrayDetection, Zone
from gt1.domain.frame import Image
from gt1.geometry import wrap_axis_angle_deg
from gt1.perception.tray.rim_and_floor import RimAndFloorTrayDetector

from . import scenes

SIZE_PX = (900, 700)
PROJECTOR = NominalScaleProjector(mm_per_px=1.0)


def zone(x0: float, y0: float, x1: float, y1: float) -> Zone:
    return Zone(
        name="tray", polygon_mm=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), surface_z_mm=-100.0
    )


FULL = zone(5.0, 5.0, 895.0, 695.0)


def detect(image: Image, in_zone: Zone = FULL) -> list[TrayDetection]:
    return RimAndFloorTrayDetector(scenes.TRAY_DETECTOR, scenes.TRAY, PROJECTOR).detect(
        Frame(image, 0.0), in_zone
    )


def assert_tray_at(tray: TrayDetection, center_px: tuple[float, float], axis_deg: float) -> None:
    assert tray.center_mm[:2] == pytest.approx(center_px, abs=4.0)
    assert wrap_axis_angle_deg(tray.axis_angle_deg - axis_deg) == pytest.approx(0.0, abs=4.0)


def test_two_trays_are_found_also_the_one_in_the_shadow_of_the_camera() -> None:
    image = scenes.with_tray(scenes.table(SIZE_PX, 80.0), (300.0, 300.0), 20.0)
    image = scenes.with_tray(image, (650.0, 420.0), -10.0)
    bright, shadowed = detect(scenes.in_shadow(image, 500, 0.35))
    assert_tray_at(bright, (300.0, 300.0), 20.0)
    assert_tray_at(shadowed, (650.0, 420.0), -10.0)
    assert shadowed.measurements["line_contrast_pct"] >= scenes.TRAY_DETECTOR.min_line_contrast_pct


@pytest.mark.parametrize("axis_deg", [8.0, 38.0, -36.0, 80.0])
def test_trays_at_any_rotation_are_found_by_the_fold_toward_the_shadow_and_the_rim_elsewhere(
    axis_deg: float,
) -> None:
    """Como en el banco (4 bandejas a 8°, 22°, 38° y -36°): una línea sola cabe en dos posiciones."""
    center = (450.0, 350.0)
    (tray,) = detect(scenes.with_tray(scenes.table(SIZE_PX, 80.0), center, axis_deg))
    assert_tray_at(tray, center, axis_deg)


def test_the_floor_fold_on_the_sides_away_from_the_shadow_does_not_make_a_tray() -> None:
    """Sin pliegue hacia la sombra no hay bandeja: el pliegue de esos lados no se busca."""
    center, axis_deg = (450.0, 350.0), 0.0
    away = scenes.sides_toward(center, axis_deg, scenes.TRAY_DETECTOR.shadow_toward_deg + 180.0)
    image = scenes.with_tray_lines(scenes.table(SIZE_PX, 80.0), center, axis_deg, scenes.FLOOR_MM, sides=away)
    assert detect(image) == []


def test_the_tray_sits_on_the_table_with_its_rim_at_its_depth() -> None:
    (tray,) = detect(scenes.with_tray(scenes.table(SIZE_PX, 80.0), (450.0, 350.0), 0.0))
    assert tray.center_mm[2] == pytest.approx(-100.0)
    assert tray.corners_mm[:, 2] == pytest.approx([-50.0] * 4)
    assert np.ptp(tray.corners_mm[:, 0]) == pytest.approx(scenes.RIM_MM[0], abs=1.0)


def test_two_parallel_sides_do_not_fix_a_tray() -> None:
    """Sin un lado perpendicular, la bandeja podría estar en cualquier punto a lo largo de las dos líneas."""
    center, axis_deg = (450.0, 350.0), 0.0
    # Con el eje a 0°, cv2.boxPoints deja el lado largo de arriba en 1 y el de abajo (hacia la sombra) en 3.
    fold_side, rim_side = (3,), (1,)
    image = scenes.with_tray_lines(
        scenes.table(SIZE_PX, 80.0), center, axis_deg, scenes.FLOOR_MM, sides=fold_side
    )
    image = scenes.with_tray_lines(image, center, axis_deg, scenes.RIM_MM, sides=rim_side)
    assert detect(image) == []


def test_a_dark_box_with_tray_lines_is_not_a_tray() -> None:
    """Lo que encierra no es PET vacío: sus paredes son más oscuras que la mesa (la caja de la botonera)."""
    center, axis_deg = (450.0, 350.0), 0.0
    box = scenes.table(SIZE_PX, 80.0)
    box_corners = np.round(cv2.boxPoints((center, scenes.RIM_MM, axis_deg))).astype(np.int32)
    cv2.fillPoly(box, [box_corners.reshape(-1, 1, 2)], 40)
    assert detect(scenes.with_tray(box, center, axis_deg)) == []


def test_an_empty_table_a_breast_or_film_glints_are_not_trays() -> None:
    assert detect(scenes.table(SIZE_PX, 80.0)) == []
    assert detect(scenes.with_breast(scenes.table(SIZE_PX, 60.0), (450.0, 350.0), 30.0, 170.0)) == []
    assert detect(scenes.with_glints(scenes.table(SIZE_PX, 80.0), count=600)) == []


def test_only_trays_centered_inside_the_zone_count() -> None:
    image = scenes.with_tray(scenes.table(SIZE_PX, 80.0), (250.0, 350.0), 0.0)
    assert detect(image, zone(450.0, 5.0, 895.0, 695.0)) == []


def test_a_coarse_grid_that_is_not_a_multiple_of_the_grid_is_rejected() -> None:
    """Cada celda gruesa junta un número entero de puntos de la grilla fina."""
    settings = scenes.TRAY_DETECTOR.model_dump() | {"coarse_grid_mm": 5.0}
    with pytest.raises(ValueError, match=r"coarse_grid_mm \(5.0\) debe ser un múltiplo de grid_mm"):
        RimAndFloorTrayDetectorConfig.model_validate(settings)

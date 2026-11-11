"""Detector de pechugas con la piel arriba por su contorno, y su línea central, en escenas sintéticas."""

from __future__ import annotations

import numpy as np
import pytest

from gt1.calibration import NominalScaleProjector
from gt1.domain import Frame, Piece, PieceStatus, Zone
from gt1.domain.frame import Image
from gt1.geometry import wrap_angle_deg, wrap_axis_angle_deg
from gt1.perception.pieces.midline import lobes_direction_deg, midline_axis_deg
from gt1.perception.pieces.skin_outline import SkinOutlinePieceDetector

from . import scenes

SIZE_PX = (900, 700)
PROJECTOR = NominalScaleProjector(mm_per_px=1.0)
ZONE = Zone(
    name="pick",
    polygon_mm=np.array([[5.0, 5.0], [895.0, 5.0], [895.0, 695.0], [5.0, 695.0]]),
    surface_z_mm=0.0,
)
CENTER_PX = (450.0, 350.0)


def detect(image: Image) -> list[Piece]:
    return SkinOutlinePieceDetector(scenes.SKIN_DETECTOR, PROJECTOR).detect(Frame(image, 0.0), ZONE)


def assert_breast_at(piece: Piece, lobes_deg: float) -> None:
    assert piece.status == PieceStatus.OK
    assert piece.center_mm[:2] == pytest.approx(CENTER_PX, abs=8.0)
    assert wrap_axis_angle_deg(piece.axis_angle_deg - lobes_deg) == pytest.approx(0.0, abs=3.0)
    assert piece.lobes_direction_deg is not None
    assert wrap_angle_deg(piece.lobes_direction_deg - lobes_deg) == pytest.approx(0.0, abs=3.0)


def test_light_skin_on_a_dark_table_is_found_with_its_midline_and_lobes() -> None:
    (piece,) = detect(scenes.with_breast(scenes.table(SIZE_PX, 60.0), CENTER_PX, 30.0, 170.0))
    assert_breast_at(piece, 30.0)


def test_skin_darker_than_a_white_cardboard_is_found_the_same_way() -> None:
    (piece,) = detect(scenes.with_breast(scenes.table(SIZE_PX, 220.0), CENTER_PX, -120.0, 150.0))
    assert_breast_at(piece, -120.0)


def test_a_piece_half_in_the_shadow_is_still_one_piece() -> None:
    lit = scenes.with_breast(scenes.table(SIZE_PX, 70.0), CENTER_PX, 90.0, 170.0)
    (piece,) = detect(scenes.in_shadow(lit, 450, 0.4))
    assert_breast_at(piece, 90.0)


def test_the_bone_side_has_too_much_texture_and_is_not_taken() -> None:
    """Hueso: carne y grasa en manchas de 8 mm, la textura de las costillas."""
    skin = scenes.with_breast(scenes.table(SIZE_PX, 60.0), CENTER_PX, 0.0, 170.0).astype(np.float64)
    patches = np.random.default_rng(0).choice([0.6, 1.3], size=(SIZE_PX[1] // 8 + 1, SIZE_PX[0] // 8 + 1))
    texture = np.kron(patches, np.ones((8, 8)))[: SIZE_PX[1], : SIZE_PX[0]]
    inside = skin > 120
    skin[inside] *= texture[inside]
    assert detect(np.clip(skin, 0, 255).astype(np.uint8)) == []


def test_a_blob_smaller_than_a_breast_is_not_a_piece() -> None:
    image = scenes.table(SIZE_PX, 60.0)
    yy, xx = np.mgrid[: SIZE_PX[1], : SIZE_PX[0]]
    image[(xx - 450) ** 2 + (yy - 350) ** 2 < 50**2] = 170
    assert detect(image) == []


@pytest.mark.parametrize("lobes_deg", [0.0, 37.0, -100.0, 170.0])
def test_midline_is_the_symmetry_axis_and_lobes_point_to_the_wider_end(lobes_deg: float) -> None:
    outline_mm = scenes.breast_outline_px((0.0, 0.0), lobes_deg)
    axis_deg = midline_axis_deg(outline_mm)
    assert wrap_axis_angle_deg(axis_deg - lobes_deg) == pytest.approx(0.0, abs=0.5)
    assert wrap_angle_deg(lobes_direction_deg(outline_mm, axis_deg) - lobes_deg) == pytest.approx(
        0.0, abs=0.5
    )

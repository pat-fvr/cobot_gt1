"""Detector por referencia: piezas por diferencia con la escena vacía, en una mesa despareja."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gt1.calibration import NominalScaleProjector
from gt1.config.schemas import ReferencePieceDetectorConfig
from gt1.domain import Frame, PieceStatus
from gt1.domain.frame import Image
from gt1.perception.image_ops import full_view_zone
from gt1.perception.pieces.empty_scene import EmptySceneMissingError, EmptySceneStore, average_images
from gt1.perception.pieces.reference import ReferencePieceDetector

MM_PER_PX = 0.5
SIZE_PX = (640, 480)
PROJECTOR = NominalScaleProjector(MM_PER_PX)
ZONE = full_view_zone(PROJECTOR, SIZE_PX, 0.0)
CONFIG = ReferencePieceDetectorConfig(
    kind="reference",
    min_difference_pct=15.0,
    glint_min_level=200,
    speck_kernel_px=3,
    blur_kernel_px=5,
    morphology_kernel_px=9,
    contour_height_mm=0.0,
    min_area_mm2=1000.0,
    max_single_piece_area_mm2=20000.0,
)
SKIN_CENTER_PX, BONE_CENTER_PX, PIECE_AXES_PX = (180, 200), (460, 260), (70, 100)


def empty_table() -> Image:
    """Mesa como la del banco: el doble de luz abajo que arriba, una mancha oscura y reflejos del film."""
    width, height = SIZE_PX
    rows = np.linspace(55.0, 110.0, height)[:, np.newaxis]
    table = np.repeat(rows, width, axis=1)
    stain = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(stain, (320, 420), (40, 25), 0, 0, 360, 255, -1)
    table[stain > 0] *= 0.7
    for x, y in [(60, 40), (300, 90), (600, 300), (120, 440)]:
        cv2.circle(table, (x, y), 3, 255.0, -1)
    return np.clip(table, 0, 255).astype(np.uint8)


def with_pieces(table: Image) -> Image:
    """Piel clara y lisa; hueso con carne oscura y grasa clara, de brillo medio igual al de la mesa."""
    scene = table.astype(np.float64)
    skin = np.zeros(table.shape, dtype=np.uint8)
    cv2.ellipse(skin, SKIN_CENTER_PX, PIECE_AXES_PX, 0, 0, 360, 255, -1)
    scene[skin > 0] = 165.0
    bone = np.zeros(table.shape, dtype=np.uint8)
    cv2.ellipse(bone, BONE_CENTER_PX, PIECE_AXES_PX, 0, 0, 360, 255, -1)
    rng = np.random.default_rng(0)
    patches = rng.choice([0.7, 1.35], size=(table.shape[0] // 8 + 1, table.shape[1] // 8 + 1))
    texture = np.kron(patches, np.ones((8, 8)))[: table.shape[0], : table.shape[1]]
    scene[bone > 0] *= texture[bone > 0]
    return np.clip(scene, 0, 255).astype(np.uint8)


def with_shifted_film_glints(scene: Image, seed: int) -> Image:
    """El film junto a la pieza de hueso brilla en otros puntos: reflejos saturados y destellos tenues."""
    rng = np.random.default_rng(seed)
    sparkled = scene.copy()
    for _ in range(600):
        x, y = int(rng.integers(250, 400)), int(rng.integers(160, 360))
        if rng.random() < 0.5:
            cv2.circle(sparkled, (x, y), int(rng.integers(1, 3)), 250, -1)
        else:
            sparkled[y : y + 2, x : x + 2] = 180
    return sparkled


def detector() -> ReferencePieceDetector:
    return ReferencePieceDetector(CONFIG, PROJECTOR, empty_table())


def frame_of(image: Image) -> Frame:
    return Frame(image=image, captured_at_s=0.0)


def test_finds_light_and_dark_textured_pieces_and_ignores_what_was_already_there() -> None:
    pieces = detector().detect(frame_of(with_pieces(empty_table())), ZONE)
    assert len(pieces) == 2
    expected_area_mm2 = np.pi * PIECE_AXES_PX[0] * PIECE_AXES_PX[1] * MM_PER_PX**2
    for piece, center_px in zip(
        sorted(pieces, key=lambda p: p.center_mm[0]), (SKIN_CENTER_PX, BONE_CENTER_PX), strict=True
    ):
        assert piece.status == PieceStatus.OK
        assert piece.area_mm2 == pytest.approx(expected_area_mm2, rel=0.1)
        assert np.allclose(piece.center_mm[:2], np.array(center_px) * MM_PER_PX, atol=3.0)


def test_film_glints_that_moved_do_not_join_the_piece() -> None:
    """En el banco, los reflejos del film que cambiaron de lugar se pegaban a la pieza y la agrandaban."""
    reference = with_shifted_film_glints(empty_table(), seed=1)
    scene = with_shifted_film_glints(with_pieces(empty_table()), seed=2)
    pieces = ReferencePieceDetector(CONFIG, PROJECTOR, reference).detect(frame_of(scene), ZONE)
    expected_area_mm2 = np.pi * PIECE_AXES_PX[0] * PIECE_AXES_PX[1] * MM_PER_PX**2
    assert [piece.area_mm2 for piece in pieces] == pytest.approx([expected_area_mm2] * 2, rel=0.1)


def test_the_empty_scene_has_no_pieces() -> None:
    assert detector().detect(frame_of(empty_table()), ZONE) == []


def test_an_even_change_of_light_or_exposure_is_not_a_piece() -> None:
    brighter = np.clip(with_pieces(empty_table()).astype(np.float64) * 1.3, 0, 255).astype(np.uint8)
    assert len(detector().detect(frame_of(brighter), ZONE)) == 2


def test_an_image_of_another_size_is_rejected() -> None:
    small = np.zeros((240, 320), dtype=np.uint8)
    with pytest.raises(ValueError, match="tamaño de la escena vacía"):
        detector().detect(frame_of(small), ZONE)


def test_empty_scene_store_round_trip_and_missing_file(tmp_path: Path) -> None:
    store = EmptySceneStore(tmp_path, "cam")
    with pytest.raises(EmptySceneMissingError, match="capture_empty_scene"):
        store.load()
    store.save(empty_table())
    assert np.array_equal(store.load(), empty_table())


def test_averaging_lowers_the_noise_and_needs_images_of_one_size() -> None:
    rng = np.random.default_rng(1)
    noisy = [np.clip(100 + rng.normal(0, 8, (60, 80)), 0, 255).astype(np.uint8) for _ in range(16)]
    assert average_images(noisy).std() < noisy[0].std() / 3
    with pytest.raises(ValueError, match="mismo tamaño"):
        average_images([noisy[0], np.zeros((10, 10), dtype=np.uint8)])

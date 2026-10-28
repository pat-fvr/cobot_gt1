"""Detector de piezas y clasificador de cara con imágenes sintéticas."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from gt1.config.schemas import BackgroundPieceDetectorConfig, HistogramFaceClassifierConfig, OtsuThreshold
from gt1.domain import Face, Frame, PieceStatus, Zone
from gt1.domain.frame import Image
from gt1.geometry import FloatArray
from gt1.perception.face.histogram import (
    FaceReferences,
    HistogramFaceClassifier,
    build_references,
    piece_histogram,
    piece_mask,
)
from gt1.perception.face.references import ReferenceStore
from gt1.perception.image_ops import erode
from gt1.perception.pieces.background import BackgroundPieceDetector

from .fakes import ScaleProjector

MM_PER_PX = 0.5
ZONE = Zone(
    name="pick",
    polygon_mm=np.array([[10.0, 10.0], [390.0, 10.0], [390.0, 290.0], [10.0, 290.0]]),
    surface_z_mm=0.0,
)
DETECTOR_CONFIG = BackgroundPieceDetectorConfig(
    kind="background",
    pieces_are="brighter",
    threshold=OtsuThreshold(method="otsu"),
    blur_kernel_px=5,
    morphology_kernel_px=5,
    contour_height_mm=30.0,
    min_area_mm2=500.0,
    max_single_piece_area_mm2=4000.0,
)
CLASSIFIER_CONFIG = HistogramFaceClassifierConfig(
    kind="histogram", color_space="gray", bins_per_channel=32, mask_erosion_px=3, min_margin=0.05
)


def frame_with(*ellipses: tuple[tuple[int, int], tuple[int, int], float]) -> Frame:
    """Fondo oscuro con elipses claras: (centro px, semiejes px, ángulo)."""
    image = np.full((600, 800), 20, dtype=np.uint8)
    for center, axes, angle in ellipses:
        cv2.ellipse(image, center, axes, angle, 0, 360, 200, -1)
    return Frame(image=image, captured_at_s=time.time())


def test_detects_an_isolated_piece_in_mm() -> None:
    detector = BackgroundPieceDetector(DETECTOR_CONFIG, ScaleProjector(MM_PER_PX))
    pieces = detector.detect(frame_with(((300, 300), (80, 40), 30.0)), ZONE)
    assert len(pieces) == 1
    piece = pieces[0]
    assert piece.status == PieceStatus.OK
    assert piece.center_mm == pytest.approx([150.0, 150.0, 30.0], abs=1.0)
    assert piece.length_mm == pytest.approx(80.0, abs=3.0)
    assert piece.width_mm == pytest.approx(40.0, abs=3.0)
    assert piece.axis_angle_deg == pytest.approx(30.0, abs=2.0)
    assert piece.area_mm2 == pytest.approx(np.pi * 40 * 20, rel=0.05)


def test_flags_clusters_and_pieces_cut_by_the_zone_edge() -> None:
    detector = BackgroundPieceDetector(DETECTOR_CONFIG, ScaleProjector(MM_PER_PX))
    frame = frame_with(((250, 300), (150, 80), 0.0), ((770, 150), (60, 30), 0.0))
    statuses = sorted(piece.status for piece in detector.detect(frame, ZONE))
    assert statuses == sorted([PieceStatus.CLUSTER, PieceStatus.CUT_BY_ZONE_EDGE])


def test_ignores_specks_and_everything_outside_the_zone() -> None:
    detector = BackgroundPieceDetector(DETECTOR_CONFIG, ScaleProjector(MM_PER_PX))
    small_zone = Zone(
        name="pick",
        polygon_mm=np.array([[10.0, 10.0], [250.0, 10.0], [250.0, 290.0], [10.0, 290.0]]),
        surface_z_mm=0.0,
    )
    frame = frame_with(((200, 300), (10, 10), 0.0), ((650, 300), (80, 40), 0.0))
    assert detector.detect(frame, small_zone) == []


def labeled_piece_image(face: Face, seed: int) -> Image:
    """Piel: clara y pareja. Hueso: franjas más oscuras (costillas), aún sobre el umbral del fondo."""
    rng = np.random.default_rng(seed)
    image = np.full((600, 800), 20, dtype=np.uint8)
    mask = np.zeros_like(image)
    cv2.ellipse(mask, (300, 300), (80, 40), 0, 0, 360, 255, -1)
    if face == Face.SKIN:
        texture = rng.normal(200, 8, image.shape)
    else:
        stripes = np.where((np.arange(image.shape[1]) // 12) % 2 == 0, 110, 165)
        texture = rng.normal(0, 8, image.shape) + stripes[None, :]
    image[mask > 0] = np.clip(texture, 0, 255).astype(np.uint8)[mask > 0]
    return image


def references_from(images: dict[Face, list[Image]]) -> FaceReferences:
    detector = BackgroundPieceDetector(DETECTOR_CONFIG, ScaleProjector(MM_PER_PX))
    histograms: dict[Face, list[FloatArray]] = {}
    for face, face_images in images.items():
        histograms[face] = []
        for image in face_images:
            frame = Frame(image=image, captured_at_s=0.0)
            (piece,) = detector.detect(frame, ZONE)
            mask = piece_mask(image, piece, CLASSIFIER_CONFIG.mask_erosion_px)
            histograms[face].append(piece_histogram(image, mask, "gray", CLASSIFIER_CONFIG.bins_per_channel))
    return build_references(
        histograms[Face.SKIN], histograms[Face.BONE], "gray", CLASSIFIER_CONFIG.bins_per_channel
    )


@pytest.mark.parametrize("face", [Face.SKIN, Face.BONE])
def test_histogram_classifier_tells_skin_from_bone(face: Face) -> None:
    references = references_from(
        {f: [labeled_piece_image(f, seed) for seed in range(3)] for f in (Face.SKIN, Face.BONE)}
    )
    classifier = HistogramFaceClassifier(CLASSIFIER_CONFIG, references)
    detector = BackgroundPieceDetector(DETECTOR_CONFIG, ScaleProjector(MM_PER_PX))
    frame = Frame(image=labeled_piece_image(face, seed=99), captured_at_s=0.0)
    (piece,) = detector.detect(frame, ZONE)
    result = classifier.classify(frame, piece)
    assert result.face == face
    assert abs(result.margin) > CLASSIFIER_CONFIG.min_margin


def test_classifier_refuses_references_built_with_other_settings() -> None:
    references = references_from({f: [labeled_piece_image(f, 0)] for f in (Face.SKIN, Face.BONE)})
    with pytest.raises(ValueError, match="vuelva a generarlas"):
        HistogramFaceClassifier(CLASSIFIER_CONFIG.model_copy(update={"bins_per_channel": 16}), references)


def test_hsv_needs_a_color_camera() -> None:
    image = labeled_piece_image(Face.SKIN, 0)
    with pytest.raises(ValueError, match="cámara color"):
        piece_histogram(image, np.full(image.shape, 255, dtype=np.uint8), "hsv", 8)


def test_reference_store_round_trip(tmp_path: Path) -> None:
    store = ReferenceStore(tmp_path, "cam")
    detector = BackgroundPieceDetector(DETECTOR_CONFIG, ScaleProjector(MM_PER_PX))
    for face in (Face.SKIN, Face.BONE):
        frame = Frame(image=labeled_piece_image(face, 1), captured_at_s=0.0)
        (piece,) = detector.detect(frame, ZONE)
        store.save_sample(frame, piece, face)
    assert store.sample_count(Face.SKIN) == store.sample_count(Face.BONE) == 1

    histograms = {
        face: [piece_histogram(image, erode(mask, 3), "gray", 32) for image, mask in store.load_samples(face)]
        for face in (Face.SKIN, Face.BONE)
    }
    references = build_references(histograms[Face.SKIN], histograms[Face.BONE], "gray", 32)
    store.save_references(references)
    loaded = store.load_references()
    assert loaded.color_space == "gray"
    assert loaded.skin_samples == 1
    assert np.allclose(loaded.skin, references.skin)

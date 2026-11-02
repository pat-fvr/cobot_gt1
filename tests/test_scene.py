"""Escena: zona, piezas y su cara a partir de una imagen sintética."""

from __future__ import annotations

import time

import cv2
import numpy as np

from gt1.calibration import NominalScaleProjector
from gt1.config.schemas import BackgroundPieceDetectorConfig, OtsuThreshold, TextureFaceClassifierConfig
from gt1.domain import Frame, PieceStatus
from gt1.perception.face.texture import TextureFaceClassifier
from gt1.perception.image_ops import full_view_zone
from gt1.perception.pieces.background import BackgroundPieceDetector
from gt1.perception.scene import ScenePerception

MM_PER_PX = 0.5
PROJECTOR = NominalScaleProjector(MM_PER_PX)
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
TEXTURE_CONFIG = TextureFaceClassifierConfig(
    kind="texture",
    smoothing_mm=1.0,
    mask_erosion_mm=3.0,
    threshold_pct_per_mm=2.4,
    min_margin_pct_per_mm=0.4,
)


def scene_perception(*, classify: bool) -> ScenePerception:
    return ScenePerception(
        detector=BackgroundPieceDetector(DETECTOR_CONFIG, PROJECTOR),
        classifier=TextureFaceClassifier(TEXTURE_CONFIG) if classify else None,
        projector=PROJECTOR,
        zone_for=lambda frame: full_view_zone(PROJECTOR, frame.size_px, 0.0),
    )


def frame_with_one_of_each_status() -> Frame:
    """Una pieza tomable, una cortada por el borde de la imagen y un grupo (elipse demasiado grande)."""
    image = np.full((600, 800), 20, dtype=np.uint8)
    for center, axes in (((200, 200), (60, 30)), ((400, 590), (60, 30)), ((600, 300), (150, 80))):
        cv2.ellipse(image, center, axes, 0.0, 0, 360, 200, -1)
    return Frame(image=image, captured_at_s=time.time())


def test_pieces_on_the_edge_get_a_face_but_clusters_do_not() -> None:
    scene = scene_perception(classify=True).perceive(frame_with_one_of_each_status())
    faces = {detection.piece.status: detection.face for detection in scene.detections}
    assert set(faces) == {PieceStatus.OK, PieceStatus.CUT_BY_ZONE_EDGE, PieceStatus.CLUSTER}
    assert faces[PieceStatus.OK] is not None
    assert faces[PieceStatus.CUT_BY_ZONE_EDGE] is not None
    assert faces[PieceStatus.CLUSTER] is None


def test_without_a_classifier_no_piece_gets_a_face() -> None:
    perception = scene_perception(classify=False)
    scene = perception.perceive(frame_with_one_of_each_status())
    assert not perception.classifies_faces
    assert len(scene.detections) == 3
    assert all(detection.face is None for detection in scene.detections)


def test_pickable_keeps_only_pieces_the_robot_can_take() -> None:
    scene = scene_perception(classify=True).perceive(frame_with_one_of_each_status())
    assert [detection.piece.status for detection in scene.pickable] == [PieceStatus.OK]


def test_the_scene_keeps_its_frame_zone_and_timing() -> None:
    frame = frame_with_one_of_each_status()
    scene = scene_perception(classify=False).perceive(frame)
    assert scene.frame is frame
    assert scene.zone.name == "full_view"
    assert scene.zone_polygon_px.max(axis=0).tolist() == [799, 599]
    assert scene.perception_s > 0


def test_the_foreground_mask_covers_the_pieces() -> None:
    frame = frame_with_one_of_each_status()
    mask = scene_perception(classify=False).foreground_mask(frame)
    assert mask[200, 200] == 255
    assert mask[50, 50] == 0

"""face_preview: con un video, procesa todas sus imágenes y graba el resultado en la corrida."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
from gt1_tools.face_preview import FacePreview, Settings

from gt1.calibration import NominalScaleProjector
from gt1.config.schemas import (
    BackgroundPieceDetectorConfig,
    NominalScaleCalibrationConfig,
    OtsuThreshold,
    VideoFileCameraConfig,
)
from gt1.domain.frame import Image
from gt1.hardware.camera.video import VideoFileCamera
from gt1.perception.image_ops import full_view_zone
from gt1.perception.pieces.background import BackgroundPieceDetector
from gt1.perception.scene import ScenePerception
from gt1.runtime.console import OperatorConsole
from gt1.runtime.run_log import RunLog

from .fakes import write_video

MM_PER_PX = 0.5
VIDEO_FPS = 20.0
FRAME_COUNT = 5
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
SETTINGS = Settings(
    zone="full_view", captures_dir="captures", video_fourcc="MJPG", video_file_extension="avi"
)


def piece_moving_right(index: int) -> Image:
    image = np.full((240, 320, 3), 20, dtype=np.uint8)
    cv2.ellipse(image, (100 + 20 * index, 120), (60, 30), 0, 0, 360, (200, 200, 200), -1)
    return image


def test_a_video_is_processed_whole_and_its_result_saved_in_the_run(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.avi"
    write_video(video_path, [piece_moving_right(index) for index in range(FRAME_COUNT)], VIDEO_FPS)
    camera = VideoFileCamera(
        VideoFileCameraConfig(
            kind="video",
            camera_id="cam",
            calibration=NominalScaleCalibrationConfig(method="nominal_scale", mm_per_px=MM_PER_PX),
            path=str(video_path),
        )
    )
    projector = NominalScaleProjector(MM_PER_PX)
    run_log = RunLog.create(tmp_path / "runs", "face_preview")
    console = OperatorConsole("test")
    try:
        with camera:
            preview = FacePreview(
                camera=camera,
                perception=ScenePerception(
                    detector=BackgroundPieceDetector(DETECTOR_CONFIG, projector),
                    classifier=None,
                    projector=projector,
                    zone_for=lambda frame: full_view_zone(projector, frame.size_px, 0.0),
                ),
                console=console,
                captures_dir=tmp_path / "captures",
                run_log=run_log,
                settings=SETTINGS,
            )
            preview.live_job()(threading.Event())
    finally:
        run_log.close()

    result = cv2.VideoCapture(str(run_log.directory / "resultado_01.avi"))
    assert int(result.get(cv2.CAP_PROP_FRAME_COUNT)) == FRAME_COUNT
    assert result.get(cv2.CAP_PROP_FPS) == VIDEO_FPS
    result.release()
    (attempt,) = (run_log.directory / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(attempt)["frames_written"] == FRAME_COUNT

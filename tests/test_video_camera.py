"""Cámara de video: todas las imágenes en orden, con la hora del video, y vuelta al inicio al terminar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gt1.config.schemas import NominalScaleCalibrationConfig, VideoFileCameraConfig
from gt1.hardware.camera import NoMoreFramesError
from gt1.hardware.camera.video import VideoFileCamera

from .fakes import write_video

VIDEO_FPS = 20.0
FRAME_COUNT = 3


def video_camera(path: Path) -> VideoFileCamera:
    return VideoFileCamera(
        VideoFileCameraConfig(
            kind="video",
            camera_id="cam",
            calibration=NominalScaleCalibrationConfig(method="nominal_scale", mm_per_px=1.0),
            path=str(path),
        )
    )


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    """Video cuya imagen ``i`` tiene brillo ``i * 80``, para reconocerla al leerla."""
    path = tmp_path / "clip.avi"
    images = [np.full((48, 64, 3), index * 80, dtype=np.uint8) for index in range(FRAME_COUNT)]
    write_video(path, images, VIDEO_FPS)
    return path


def test_gives_every_frame_in_order_on_the_video_timeline(video_path: Path) -> None:
    with video_camera(video_path) as camera:
        frames = [camera.capture() for _ in range(FRAME_COUNT)]
        assert camera.source_video_fps == VIDEO_FPS

    assert [int(frame.image.mean()) // 40 for frame in frames] == [0, 2, 4]
    intervals_s = np.diff([frame.captured_at_s for frame in frames])
    assert intervals_s == pytest.approx([1 / VIDEO_FPS] * (FRAME_COUNT - 1))


def test_at_the_end_reports_it_and_starts_over(video_path: Path) -> None:
    with video_camera(video_path) as camera:
        for _ in range(FRAME_COUNT):
            camera.capture()
        with pytest.raises(NoMoreFramesError, match=r"clip\.avi"):
            camera.capture()
        assert int(camera.capture().image.mean()) // 40 == 0


def test_a_missing_video_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"no_existe\.avi"):
        video_camera(tmp_path / "no_existe.avi").connect()

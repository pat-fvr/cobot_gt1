"""Grabación de video: va en tiempo real aunque la cámara entregue más o menos imágenes que el video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gt1.domain import Frame
from gt1.runtime.video_recording import VideoRecording, frames_due

VIDEO_FPS = 30.0
FOURCC = "MJPG"


def frame_at(time_s: float, *, color: bool) -> Frame:
    shape = (48, 64, 3) if color else (48, 64)
    return Frame(image=np.full(shape, 128, dtype=np.uint8), captured_at_s=time_s)


def test_frames_due_counts_the_first_frame() -> None:
    assert frames_due(0.0, 30.0) == 1
    assert frames_due(1.0, 30.0) == 31


@pytest.mark.parametrize(
    ("camera_fps", "expected_written"),
    [(15.0, 31), (60.0, 31)],
    ids=["cámara más lenta: repite", "cámara más rápida: salta"],
)
def test_one_second_of_camera_is_one_second_of_video(
    tmp_path: Path, camera_fps: float, expected_written: int
) -> None:
    times_s = [index / camera_fps for index in range(int(camera_fps) + 1)]
    recording = VideoRecording(tmp_path / "v.avi", VIDEO_FPS, FOURCC, frame_at(times_s[0], color=False))
    for time_s in times_s[1:]:
        recording.write(frame_at(time_s, color=False))
    recording.close()

    assert recording.frames_written == expected_written
    assert recording.summary()["camera_fps"] == camera_fps
    assert recording.summary()["duration_s"] == 1.0


@pytest.mark.parametrize("color", [False, True], ids=["monocromática", "color"])
def test_the_video_file_plays_back_with_the_written_frames(tmp_path: Path, color: bool) -> None:
    recording = VideoRecording(tmp_path / "v.avi", VIDEO_FPS, FOURCC, frame_at(0.0, color=color))
    recording.write(frame_at(0.1, color=color))
    recording.close()

    video = cv2.VideoCapture(str(tmp_path / "v.avi"))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == recording.frames_written == 4
    ok, image = video.read()
    video.release()
    assert ok
    assert image.shape[:2] == (48, 64)

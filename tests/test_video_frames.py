"""video_frames: lee el video en orden y guarda la imagen vista en la carpeta junto al video."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
from gt1_tools.video_frames import VideoPlayer, frame_path

from gt1.domain.frame import Image
from gt1.hardware.camera.video import VideoFile
from gt1.runtime.console import OperatorConsole
from gt1.runtime.run_log import RunLog

from .fakes import write_video

VIDEO_FPS = 10.0
FRAME_COUNT = 3


def numbered_images() -> list[Image]:
    """La imagen ``i`` tiene brillo ``i * 80``, para reconocerla al leerla."""
    return [np.full((48, 64, 3), index * 80, dtype=np.uint8) for index in range(FRAME_COUNT)]


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    path = tmp_path / "videos" / "cam" / "clip.avi"
    path.parent.mkdir(parents=True)
    write_video(path, numbered_images(), VIDEO_FPS)
    return path


def test_frame_path_is_in_a_folder_next_to_the_video() -> None:
    path = frame_path(Path("videos/cam/clip.mp4"), "frames", 42)
    assert path == Path("videos/cam/frames/clip_000042.png")


def test_video_file_yields_every_frame_in_order_with_its_time(video_path: Path) -> None:
    with VideoFile(video_path) as video:
        frames = list(video.frames())
        assert video.fps == VIDEO_FPS
        assert video.frame_count == FRAME_COUNT

    assert [frame.index for frame in frames] == [0, 1, 2]
    assert [frame.time_s for frame in frames] == pytest.approx([0.0, 0.1, 0.2])
    assert [int(frame.image.mean()) // 40 for frame in frames] == [0, 2, 4]


def test_a_missing_video_names_the_config_field(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="video"):
        VideoFile(tmp_path / "no_existe.mp4")


def test_saving_writes_the_shown_frame_next_to_the_video_and_records_it(
    tmp_path: Path, video_path: Path
) -> None:
    run_log = RunLog.create(tmp_path / "runs", "video_frames")
    player = VideoPlayer(
        video_path=video_path, frames_dir="frames", console=OperatorConsole("test"), run_log=run_log
    )
    player.play_job()(threading.Event())
    player.save_shown()
    run_log.close()

    saved = video_path.parent / "frames" / "clip_000002.png"
    assert saved.is_file()
    record = json.loads((run_log.directory / "attempts.jsonl").read_text(encoding="utf-8"))
    assert record["frame_index"] == 2
    assert record["image"] == saved.as_posix()

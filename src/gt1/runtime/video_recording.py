"""Grabación de video en tiempo real a partir de la hora de captura de cada imagen.

El video dura lo mismo que lo capturado aunque las imágenes lleguen a otro
ritmo que ``video_fps``: si llegan menos, cada una se repite hasta la hora de
la siguiente; si llegan más, las que sobran se saltan.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2

from gt1.domain import Frame


def frames_due(elapsed_s: float, video_fps: float) -> int:
    """Imágenes que debe tener el video a ``elapsed_s`` de su inicio para ir en tiempo real."""
    return math.floor(elapsed_s * video_fps) + 1


class VideoRecording:
    """Un archivo de video abierto, que se llena en tiempo real según la hora de captura de cada imagen."""

    def __init__(self, path: Path, video_fps: float, fourcc: str, first_frame: Frame) -> None:
        width, height = first_frame.size_px
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._video_fps = video_fps
        self._writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter.fourcc(*fourcc),
            video_fps,
            (width, height),
            isColor=first_frame.image.ndim == 3,
        )
        if not self._writer.isOpened():
            raise RuntimeError(
                f'OpenCV no pudo crear {path.name} con el códec {fourcc}: probar fourcc = "MJPG" '
                "con la extensión avi en la configuración del programa."
            )
        self._started_at_s = first_frame.captured_at_s
        self._last_frame_at_s = first_frame.captured_at_s
        self.frames_captured = 0
        self.frames_written = 0
        self.write(first_frame)

    @property
    def duration_s(self) -> float:
        return self._last_frame_at_s - self._started_at_s

    def write(self, frame: Frame) -> None:
        self.frames_captured += 1
        self._last_frame_at_s = frame.captured_at_s
        due = frames_due(frame.captured_at_s - self._started_at_s, self._video_fps)
        while self.frames_written < due:
            self._writer.write(frame.image)
            self.frames_written += 1

    def close(self) -> None:
        self._writer.release()

    def summary(self) -> dict[str, str | int | float]:
        camera_fps = (self.frames_captured - 1) / self.duration_s if self.duration_s > 0 else 0.0
        return {
            "video": self.path.as_posix(),
            "duration_s": round(self.duration_s, 2),
            "frames_captured": self.frames_captured,
            "frames_written": self.frames_written,
            "camera_fps": round(camera_fps, 1),
            "video_fps": self._video_fps,
        }

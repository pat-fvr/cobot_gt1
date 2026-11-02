"""Videos grabados (p. ej. con record_video): lectura en orden y cámara para probar la percepción sin cámara.

``VideoFileCamera`` entrega todas las imágenes, en orden y sin saltarse
ninguna, a lo más a la velocidad del video: si quien las procesa es más lento,
el video avanza más lento en vez de saltar imágenes. Por eso ``captured_at_s``
es la hora de cada imagen en el video (inicio de la reproducción + su tiempo en
el video) y no la del reloj: una grabación del resultado dura lo mismo que el
video original. Al terminar, ``capture`` lanza ``NoMoreFramesError`` y la
captura siguiente vuelve al inicio.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from gt1 import paths
from gt1.config.schemas import VideoFileCameraConfig
from gt1.domain import Frame
from gt1.domain.frame import Image

from .base import Camera, NoMoreFramesError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class VideoFrame:
    image: Image
    index: int
    time_s: float


class VideoFile:
    """Un video abierto con OpenCV, que entrega sus imágenes en orden."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise RuntimeError(f"OpenCV no pudo abrir el video {path}: revisar que exista y su ruta.")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            self._capture.release()
            raise RuntimeError(f"{path.name} no informa sus imágenes por segundo: no se puede reproducir.")
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __enter__(self) -> VideoFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._capture.release()

    def frames(self) -> Iterator[VideoFrame]:
        index = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                return
            yield VideoFrame(image=np.asarray(image, dtype=np.uint8), index=index, time_s=index / self.fps)
            index += 1


class VideoFileCamera(Camera):
    def __init__(self, config: VideoFileCameraConfig) -> None:
        self._config = config
        self._video: VideoFile | None = None
        self._frames: Iterator[VideoFrame] = iter(())
        self._started_at_s = 0.0

    @property
    def camera_id(self) -> str:
        return self._config.camera_id

    @property
    def is_simulated(self) -> bool:
        return True

    @property
    def source_video_fps(self) -> float:
        return self._connected_video().fps

    def connect(self) -> None:
        self._video = VideoFile(paths.resolve(self._config.path))
        self._frames = self._video.frames()
        logger.info(
            "Cámara de video: %s, %d imágenes a %.1f img/s.",
            self._video.path,
            self._video.frame_count,
            self._video.fps,
        )

    def disconnect(self) -> None:
        if self._video is not None:
            self._video.close()
            self._video = None

    def capture(self) -> Frame:
        video = self._connected_video()
        frame = next(self._frames, None)
        if frame is None:
            self.disconnect()
            self.connect()
            raise NoMoreFramesError(f"Terminó el video {video.path.name}.")
        if frame.index == 0:
            self._started_at_s = time.time()
        captured_at_s = self._started_at_s + frame.time_s
        time.sleep(max(0.0, captured_at_s - time.time()))
        return Frame(image=frame.image, captured_at_s=captured_at_s)

    def _connected_video(self) -> VideoFile:
        if self._video is None:
            raise RuntimeError("La cámara de video no está conectada.")
        return self._video

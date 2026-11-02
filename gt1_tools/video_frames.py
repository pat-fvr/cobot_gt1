"""video_frames: reproducir un video grabado y guardar imágenes de él.

Reproduce ``video`` (``config/tools/video_frames.toml``) a su velocidad real;
``s`` guarda la imagen que se está viendo en la carpeta ``frames_dir`` junto al
video, como ``<nombre del video>_<número de imagen>.png``. Cada imagen guardada
se anota en el registro de la corrida.

Las imágenes guardadas sirven para probar la percepción sin cámara, con una
cámara de carpeta (``FolderCamera``) que apunte a esa carpeta.

Teclas:
    s        guardar la imagen que se está viendo
    espacio  pausar / seguir
    v        reproducir desde el inicio (al terminar el video o después de detenerlo con 'x')
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
from pydantic import Field

from gt1 import paths
from gt1.config.schemas import StrictModel
from gt1.hardware.camera.video import VideoFile, VideoFrame
from gt1.runtime.console import OperatorConsole
from gt1.runtime.program import Program, run_program
from gt1.runtime.run_log import RunLog
from gt1.runtime.runner import Job

PROGRAM = "video_frames"
# Cada cuánto revisa la pausa si el operador la quitó o detuvo el video.
PAUSE_POLL_S = 0.05


class Settings(StrictModel):
    video: str = Field(description="Video a reproducir (ruta relativa a la carpeta del proyecto).")
    frames_dir: str = Field(description="Carpeta de las imágenes guardadas, junto al video.")


class VideoFramesConfig(StrictModel):
    settings: Settings


def frame_path(video_path: Path, frames_dir: str, index: int) -> Path:
    """Archivo de la imagen ``index`` del video: guardar la misma imagen dos veces la reemplaza."""
    return video_path.parent / frames_dir / f"{video_path.stem}_{index:06d}.png"


class VideoPlayer:
    def __init__(
        self, *, video_path: Path, frames_dir: str, console: OperatorConsole, run_log: RunLog
    ) -> None:
        self._video_path = video_path
        self._frames_dir = frames_dir
        self._console = console
        self._run_log = run_log
        # Lo cambia la tecla 'espacio' (hilo de la consola); el hilo de trabajo espera mientras está activa.
        self._paused = threading.Event()
        self._lock = threading.Lock()
        self._shown: VideoFrame | None = None

    def play_job(self) -> Job:
        def run(stop: threading.Event) -> None:
            self._paused.clear()
            with VideoFile(self._video_path) as video:
                for frame in video.frames():
                    shown_at_s = time.perf_counter()
                    self._show(frame, video.frame_count)
                    if self._wait_next(stop, video.fps, shown_at_s):
                        return
            self._console.set_status(f"Fin de {self._video_path.name}. 'v' para verlo de nuevo.")

        return run

    def toggle_pause(self) -> None:
        if not self._console.job_running:
            self._console.set_status("El video no se está reproduciendo: 'v' para reproducirlo.")
            return
        if self._paused.is_set():
            self._paused.clear()
        else:
            self._paused.set()
            self._console.set_status(f"PAUSA en {self._position()} | espacio para seguir, 's' guarda")

    def save_shown(self) -> None:
        with self._lock:
            frame = self._shown
        if frame is None:
            self._console.set_status("Todavía no hay imagen que guardar.")
            return
        path = frame_path(self._video_path, self._frames_dir, frame.index)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame.image):
            raise RuntimeError(f"OpenCV no pudo escribir {path}.")
        self._run_log.record(
            {
                "video": self._video_path.as_posix(),
                "frame_index": frame.index,
                "time_s": round(frame.time_s, 3),
                "image": path.as_posix(),
            },
            {},
        )
        self._console.set_status(f"Guardada {path.name} en {path.parent}")

    def _show(self, frame: VideoFrame, frame_count: int) -> None:
        with self._lock:
            self._shown = frame
        self._console.show(frame.image)
        self._console.set_status(f"{self._position()} de {frame_count} | 's' guarda, espacio pausa")

    def _wait_next(self, stop: threading.Event, fps: float, shown_at_s: float) -> bool:
        """Espera la hora de la imagen siguiente (y mientras dure la pausa); devuelve si se pidió detener."""
        while self._paused.is_set():
            if stop.wait(PAUSE_POLL_S):
                return True
        remaining_s = 1.0 / fps - (time.perf_counter() - shown_at_s)
        return stop.wait(max(0.0, remaining_s))

    def _position(self) -> str:
        with self._lock:
            frame = self._shown
        if frame is None:
            return "inicio"
        return f"imagen {frame.index + 1} ({frame.time_s:.1f} s)"


def video_frames(program: Program[VideoFramesConfig]) -> None:
    settings = program.config.settings
    console = OperatorConsole(f"GT1 - {PROGRAM}")
    player = VideoPlayer(
        video_path=paths.resolve(settings.video),
        frames_dir=settings.frames_dir,
        console=console,
        run_log=program.run_log,
    )
    console.bind_action("s", "Guardar la imagen", player.save_shown)
    console.bind_action("space", "Pausar / seguir", player.toggle_pause)
    console.bind_job("v", "Reproducir desde el inicio", player.play_job())
    console.start_job("reproducir", player.play_job())
    console.run()


def main() -> None:
    run_program("tools", PROGRAM, VideoFramesConfig, video_frames)


if __name__ == "__main__":
    main()

"""record_video: grabar video con la cámara (cualquier cámara de config/cameras/).

Muestra la imagen en vivo; ``r`` empieza a grabar y ``r`` otra vez termina el
video, que queda en ``videos/<camera_id>/<fecha_hora>.<extensión>``. Cada video
se anota en el registro de la corrida (duración, imágenes, img/s de la cámara).

El video va en tiempo real aunque la cámara entregue menos imágenes por segundo
que ``video_fps`` (la Scout entrega ~17 img/s): cada imagen se repite hasta la
hora de captura de la siguiente. Si la cámara entrega más, las que sobran se saltan.

Teclas:
    r        empezar / terminar la grabación
    v        volver a la imagen en vivo (después de detenerla con 'x'; 'x' termina el video)
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from pydantic import Field

from gt1 import paths
from gt1.config.schemas import StrictModel
from gt1.domain import Frame
from gt1.hardware.camera import Camera
from gt1.runtime.console import OperatorConsole
from gt1.runtime.factories import build_camera
from gt1.runtime.program import Program, run_program
from gt1.runtime.run_log import RunLog
from gt1.runtime.runner import Job
from gt1.runtime.station import StationHardware
from gt1.runtime.video_recording import VideoRecording

PROGRAM = "record_video"
STATUS_PERIOD_S = 0.5
logger = logging.getLogger(__name__)


class Components(StrictModel):
    camera: str


class Settings(StrictModel):
    videos_dir: str = Field(
        description="Carpeta de los videos; cada cámara graba en <videos_dir>/<camera_id>/."
    )
    video_fps: float = Field(gt=0, description="Imágenes por segundo del archivo de video.")
    fourcc: str = Field(
        min_length=4, max_length=4, description="Códec de OpenCV (p. ej. MJPG para .avi, mp4v para .mp4)."
    )
    file_extension: str = Field(description="Extensión del archivo, sin punto (avi, mp4).")


class RecordVideoConfig(StrictModel):
    components: Components
    settings: Settings


class VideoRecorder:
    def __init__(
        self,
        *,
        camera: Camera,
        console: OperatorConsole,
        run_log: RunLog,
        settings: Settings,
        videos_dir: Path,
    ) -> None:
        self._camera = camera
        self._console = console
        self._run_log = run_log
        self._settings = settings
        self._videos_dir = videos_dir
        # Lo cambia la tecla 'r' (hilo de la consola); el hilo de trabajo abre y cierra el video.
        self._recording_requested = threading.Event()

    def toggle_recording(self) -> None:
        if not self._console.job_running:
            self._console.set_status("Para grabar, volver a la imagen en vivo con 'v'.")
            return
        if self._recording_requested.is_set():
            self._recording_requested.clear()
        else:
            self._recording_requested.set()

    def live_job(self) -> Job:
        def run(stop: threading.Event) -> None:
            recording: VideoRecording | None = None
            try:
                frames, window_start = 0, time.perf_counter()
                while not stop.is_set():
                    frame = self._camera.capture()
                    recording = self._update_recording(recording, frame)
                    self._console.show(frame.image)
                    frames += 1
                    elapsed_s = time.perf_counter() - window_start
                    if elapsed_s >= STATUS_PERIOD_S:
                        self._console.set_status(self._status(frames / elapsed_s, frame, recording))
                        frames, window_start = 0, time.perf_counter()
            finally:
                self._recording_requested.clear()
                if recording is not None:
                    self._finish(recording)

        return run

    def _update_recording(self, recording: VideoRecording | None, frame: Frame) -> VideoRecording | None:
        """Abre, llena o cierra el video según lo pedido con 'r'."""
        if self._recording_requested.is_set():
            if recording is None:
                path = (
                    self._videos_dir
                    / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self._settings.file_extension}"
                )
                logger.info("Grabando %s", path)
                return VideoRecording(path, self._settings.video_fps, self._settings.fourcc, frame)
            recording.write(frame)
            return recording
        if recording is not None:
            self._finish(recording)
        return None

    def _finish(self, recording: VideoRecording) -> None:
        recording.close()
        summary = recording.summary()
        self._run_log.record(dict(summary), {})
        logger.info(
            "Video guardado: %s (%.1f s, cámara a %.1f img/s).",
            recording.path,
            summary["duration_s"],
            summary["camera_fps"],
        )
        self._console.set_status(f"Video guardado: {recording.path.name} ({summary['duration_s']} s)")

    @staticmethod
    def _status(fps: float, frame: Frame, recording: VideoRecording | None) -> str:
        width, height = frame.size_px
        live = f"{fps:.1f} img/s, {width} x {height} px"
        if recording is None:
            return f"{live} | 'r' para grabar"
        return f"GRABANDO {recording.path.name}: {recording.duration_s:.0f} s | {live}"


def record_video(program: Program[RecordVideoConfig]) -> None:
    components = program.config.components
    settings = program.config.settings
    camera_config = program.repository.camera(components.camera)
    videos_dir = paths.resolve(settings.videos_dir) / camera_config.camera_id

    with build_camera(camera_config) as camera:
        program.confirm_station(
            StationHardware(
                camera=camera,
                camera_name=components.camera,
                robot=None,
                robot_name=None,
                gripper=None,
                tray_name=None,
            )
        )
        console = OperatorConsole(f"GT1 - {PROGRAM}")
        recorder = VideoRecorder(
            camera=camera, console=console, run_log=program.run_log, settings=settings, videos_dir=videos_dir
        )
        console.bind_action("r", "Grabar / terminar el video", recorder.toggle_recording)
        console.bind_job("v", "Imagen en vivo", recorder.live_job())
        console.start_job("en vivo", recorder.live_job())
        console.run()


def main() -> None:
    run_program("tools", PROGRAM, RecordVideoConfig, record_video)


if __name__ == "__main__":
    main()

"""camera_viewer: imagen en vivo de la cámara y captura de imágenes.

Las capturas van a ``captures/<camera_id>/`` y sirven para probar la
percepción sin cámara (``config/cameras/folder_basler.toml``).

Teclas:
    espacio  guardar la imagen que se está viendo
    v        volver a la imagen en vivo (después de detenerla con 'x')
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from gt1 import paths
from gt1.config.schemas import StrictModel
from gt1.domain import Frame
from gt1.hardware.camera import Camera
from gt1.runtime.console import OperatorConsole
from gt1.runtime.factories import build_camera
from gt1.runtime.program import Program, run_program
from gt1.runtime.runner import Job
from gt1.runtime.station import StationHardware

PROGRAM = "camera_viewer"


class Components(StrictModel):
    camera: str


class Settings(StrictModel):
    captures_dir: str


class CameraViewerConfig(StrictModel):
    components: Components
    settings: Settings


class CameraViewer:
    def __init__(self, camera: Camera, console: OperatorConsole, captures_dir: Path) -> None:
        self._camera = camera
        self._console = console
        self._captures_dir = captures_dir
        self._lock = threading.Lock()
        self._latest: Frame | None = None

    def live_job(self) -> Job:
        def run(stop: threading.Event) -> None:
            frames, window_start = 0, time.perf_counter()
            while not stop.is_set():
                frame = self._camera.capture()
                with self._lock:
                    self._latest = frame
                self._console.show(frame.image)
                frames += 1
                elapsed_s = time.perf_counter() - window_start
                if elapsed_s >= 1.0:
                    width, height = frame.size_px
                    self._console.set_status(
                        f"En vivo: {frames / elapsed_s:.1f} imágenes/s, {width} x {height} px"
                    )
                    frames, window_start = 0, time.perf_counter()

        return run

    def save_latest(self) -> None:
        with self._lock:
            frame = self._latest
        if frame is None:
            self._console.set_status("Todavía no hay imagen que guardar.")
            return
        self._captures_dir.mkdir(parents=True, exist_ok=True)
        path = self._captures_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        cv2.imwrite(str(path), frame.image)
        self._console.set_status(f"Guardada {path.name} en {path.parent}")


def camera_viewer(program: Program[CameraViewerConfig]) -> None:
    components = program.config.components
    camera_config = program.repository.camera(components.camera)
    captures_dir = paths.resolve(program.config.settings.captures_dir) / camera_config.camera_id

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
        viewer = CameraViewer(camera, console, captures_dir)
        console.bind_action("space", "Guardar la imagen", viewer.save_latest)
        console.bind_job("v", "Imagen en vivo", viewer.live_job())
        console.start_job("en vivo", viewer.live_job())
        console.run()


def main() -> None:
    run_program("tools", PROGRAM, CameraViewerConfig, camera_viewer)


if __name__ == "__main__":
    main()

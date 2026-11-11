"""capture_empty_scene: imagen de la escena vacía para el detector de piezas por referencia.

Protocolo:
    1. Dejar la mesa como va a quedar en las pruebas (film, bandeja), sin piezas y con el
       robot fuera de la vista.
    2. 'espacio' promedia varias imágenes y las guarda en
       ``references/<camera_id>/empty_scene.png`` (reemplaza la anterior).
    3. Repetir si cambia la luz, el film, la exposición o la posición de la cámara: lo que
       cambie respecto de la referencia se toma por pieza.

Teclas:
    espacio  capturar y guardar la escena vacía
    v        volver a la imagen en vivo
"""

from __future__ import annotations

import logging
import threading

from pydantic import Field

from gt1 import paths
from gt1.config.schemas import StrictModel
from gt1.hardware.camera import Camera
from gt1.perception.pieces.empty_scene import EmptySceneStore
from gt1.runtime import captures
from gt1.runtime.console import OperatorConsole
from gt1.runtime.factories import build_camera
from gt1.runtime.program import Program, run_program
from gt1.runtime.run_log import RunLog
from gt1.runtime.runner import Job
from gt1.runtime.station import StationHardware

PROGRAM = "capture_empty_scene"
logger = logging.getLogger(__name__)


class Components(StrictModel):
    camera: str


class Settings(StrictModel):
    frames_to_average: int = Field(ge=1, description="Imágenes que se promedian para la referencia.")


class CaptureEmptySceneConfig(StrictModel):
    components: Components
    settings: Settings


class EmptySceneCapture:
    def __init__(
        self,
        *,
        camera: Camera,
        store: EmptySceneStore,
        console: OperatorConsole,
        run_log: RunLog,
        frames_to_average: int,
    ) -> None:
        self._camera = camera
        self._store = store
        self._console = console
        self._run_log = run_log
        self._frames_to_average = frames_to_average

    def live_job(self) -> Job:
        def run(stop: threading.Event) -> None:
            while not stop.is_set():
                self._console.show(self._camera.capture().image)
                self._console.set_status("Sin piezas y con el robot fuera de la vista: 'espacio' captura")

        return run

    def capture(self) -> None:
        """Detiene la imagen en vivo y captura en el hilo de trabajo."""
        self._console.stop_job(timeout_s=2.0)
        self._console.start_job("capturar escena vacia", self._capture_job)

    def _capture_job(self, stop: threading.Event) -> None:
        frames = self._frames_to_average
        empty_scene = captures.capture_empty_scene(
            self._camera, self._store, frames, stop, self._console.set_status
        )
        if empty_scene is None:
            self._console.set_status("Captura detenida: no se guardó nada. 'v' vuelve a la imagen en vivo.")
            return
        path = self._store.path
        self._run_log.record({"path": str(path), "frames_averaged": frames}, {"empty_scene": empty_scene})
        logger.info("Escena vacía guardada en %s (promedio de %d imágenes).", path, frames)
        self._console.show(empty_scene)
        self._console.set_status(f"Guardada {path.name} ({frames} imágenes). 'v' vuelve a la imagen en vivo.")


def capture_empty_scene(program: Program[CaptureEmptySceneConfig]) -> None:
    camera_config = program.repository.camera(program.config.components.camera)
    with build_camera(camera_config) as camera:
        program.confirm_station(
            StationHardware(
                camera=camera,
                camera_name=program.config.components.camera,
                robot=None,
                robot_name=None,
                gripper=None,
                tray_name=None,
            )
        )
        console = OperatorConsole(f"GT1 - {PROGRAM}")
        capture = EmptySceneCapture(
            camera=camera,
            store=EmptySceneStore(paths.REFERENCES_DIR, camera.camera_id),
            console=console,
            run_log=program.run_log,
            frames_to_average=program.config.settings.frames_to_average,
        )
        console.bind_action("space", "Capturar la escena vacia", capture.capture)
        console.bind_job("v", "Imagen en vivo", capture.live_job())
        console.start_job("en vivo", capture.live_job())
        console.run()


def main() -> None:
    run_program("tools", PROGRAM, CaptureEmptySceneConfig, capture_empty_scene)


if __name__ == "__main__":
    main()

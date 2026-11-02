"""face_preview: detección de piezas y de su cara en tiempo real, con cualquier cámara.

Muestra cada imagen con las piezas encontradas y su cara (verde = piel,
naranjo = hueso, amarillo = incierto). El contorno gris es una pieza no
tomable: las del borde muestran igual su cara; los grupos, solo su estado. En
la barra de estado, las imágenes por segundo, el tiempo de percepción y el
conteo de caras.

Sirve para ajustar el detector y el clasificador mirando el resultado: si la
cámara todavía no tiene referencias de cara, solo detecta las piezas (para
generarlas, ``gt1_tools/build_face_references.py``).

Si la cámara es un video (``kind = "video"``), se procesan todas sus imágenes y
lo que se muestra se graba en la carpeta de la corrida (``runs/face_preview/``),
con la duración del video original. 'x' termina la grabación antes del final.

Teclas:
    m        alternar la vista de la máscara (lo que el detector considera pieza)
    espacio  guardar la imagen sin dibujos en captures/<camera_id>/
    v        volver a la imagen en vivo (después de detenerla con 'x'); con un video, desde el inicio
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from pydantic import Field

from gt1 import paths
from gt1.config.schemas import StrictModel, ViewZoneName
from gt1.domain import Frame
from gt1.domain.frame import Image
from gt1.hardware.camera import Camera, NoMoreFramesError
from gt1.perception.face.references import ReferencesMissingError
from gt1.perception.scene import Scene, ScenePerception
from gt1.runtime.console import OperatorConsole
from gt1.runtime.factories import build_camera, build_face_classifier, build_scene_perception
from gt1.runtime.overlay import draw_scene, tint_mask
from gt1.runtime.program import Program, run_program
from gt1.runtime.run_log import RunLog
from gt1.runtime.runner import Job
from gt1.runtime.station import StationHardware
from gt1.runtime.video_recording import VideoRecording

PROGRAM = "face_preview"
STATUS_PERIOD_S = 0.5
logger = logging.getLogger(__name__)


class Components(StrictModel):
    camera: str
    piece_detector: str
    face_classifier: str


class Settings(StrictModel):
    zone: ViewZoneName
    captures_dir: str
    video_fourcc: str = Field(
        min_length=4,
        max_length=4,
        description="Códec del video del resultado (con cámara de video): MJPG para .avi, mp4v para .mp4.",
    )
    video_file_extension: str = Field(description="Extensión del video del resultado, sin punto (avi, mp4).")


class FacePreviewConfig(StrictModel):
    components: Components
    settings: Settings


class FacePreview:
    def __init__(
        self,
        *,
        camera: Camera,
        perception: ScenePerception,
        console: OperatorConsole,
        captures_dir: Path,
        run_log: RunLog,
        settings: Settings,
    ) -> None:
        self._camera = camera
        self._perception = perception
        self._console = console
        self._captures_dir = captures_dir
        self._run_log = run_log
        self._settings = settings
        self._show_mask = False
        self._lock = threading.Lock()
        self._latest: Frame | None = None

    def live_job(self) -> Job:
        def run(stop: threading.Event) -> None:
            recording: VideoRecording | None = None
            try:
                frames, perception_s, window_start = 0, 0.0, time.perf_counter()
                while not stop.is_set():
                    frame = self._camera.capture()
                    with self._lock:
                        self._latest = frame
                    scene = self._perception.perceive(frame)
                    perception_s += scene.perception_s
                    shown = Frame(self._render(scene), frame.captured_at_s)
                    self._console.show(shown.image)
                    recording = self._record(recording, shown)
                    frames += 1
                    elapsed_s = time.perf_counter() - window_start
                    if elapsed_s >= STATUS_PERIOD_S:
                        self._console.set_status(
                            self._status(frames / elapsed_s, perception_s / frames, scene)
                        )
                        frames, perception_s, window_start = 0, 0.0, time.perf_counter()
            except NoMoreFramesError as error:
                self._console.set_status(f"{error} 'v' para verlo de nuevo.")
            finally:
                if recording is not None:
                    self._finish(recording)

        return run

    def toggle_mask(self) -> None:
        self._show_mask = not self._show_mask

    def save_latest(self) -> None:
        with self._lock:
            frame = self._latest
        if frame is None:
            self._console.set_status("Todavía no hay imagen que guardar.")
            return
        self._captures_dir.mkdir(parents=True, exist_ok=True)
        path = self._captures_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        cv2.imwrite(str(path), frame.image)
        logger.info("Imagen guardada en %s", path)

    def _record(self, recording: VideoRecording | None, shown: Frame) -> VideoRecording | None:
        """Graba lo que se muestra si las imágenes vienen de un video; abre el archivo con la primera."""
        video_fps = self._camera.source_video_fps
        if video_fps is None:
            return None
        if recording is None:
            number = self._run_log.attempt_count + 1
            path = self._run_log.directory / f"resultado_{number:02d}.{self._settings.video_file_extension}"
            logger.info("Grabando el resultado en %s", path)
            return VideoRecording(path, video_fps, self._settings.video_fourcc, shown)
        recording.write(shown)
        return recording

    def _finish(self, recording: VideoRecording) -> None:
        recording.close()
        self._run_log.record(dict(recording.summary()), {})
        logger.info("Resultado guardado en %s (%.1f s).", recording.path, recording.duration_s)
        self._console.set_status(f"Resultado guardado en {recording.path}. 'v' para verlo de nuevo.")

    def _render(self, scene: Scene) -> Image:
        if self._show_mask:
            return tint_mask(scene.frame.image, self._perception.foreground_mask(scene.frame))
        return draw_scene(scene)

    def _status(self, fps: float, perception_s: float, scene: Scene) -> str:
        faces = Counter(
            detection.face.face.value for detection in scene.detections if detection.face is not None
        )
        not_pickable = len(scene.detections) - len(scene.pickable)
        summary = ", ".join(f"{name} {count}" for name, count in sorted(faces.items())) or "sin clasificar"
        mode = " | MÁSCARA" if self._show_mask else ""
        references = "" if self._perception.classifies_faces else " | sin referencias: solo detección"
        return (
            f"{fps:.1f} img/s, percepción {perception_s * 1000:.0f} ms | {len(scene.detections)} piezas: "
            f"{summary}, no tomables {not_pickable}{references}{mode}"
        )


def face_preview(program: Program[FacePreviewConfig]) -> None:
    components = program.config.components
    repository = program.repository
    camera_config = repository.camera(components.camera)
    try:
        classifier = build_face_classifier(
            repository.face_classifier(components.face_classifier), camera_config.camera_id
        )
    except ReferencesMissingError as error:
        logger.warning("%s Se muestra solo la detección de piezas.", error)
        classifier = None
    perception = build_scene_perception(
        repository, camera_config, components.piece_detector, classifier, program.config.settings.zone
    )
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
        preview = FacePreview(
            camera=camera,
            perception=perception,
            console=console,
            captures_dir=captures_dir,
            run_log=program.run_log,
            settings=program.config.settings,
        )
        console.bind_action("m", "Mascara del detector on/off", preview.toggle_mask)
        console.bind_action("space", "Guardar la imagen", preview.save_latest)
        console.bind_job("v", "Imagen en vivo", preview.live_job())
        console.start_job("en vivo", preview.live_job())
        console.run()


def main() -> None:
    run_program("tools", PROGRAM, FacePreviewConfig, face_preview)


if __name__ == "__main__":
    main()

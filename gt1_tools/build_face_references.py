"""build_face_references: muestras etiquetadas de piel y hueso, y referencias del clasificador por histograma.

Protocolo: poner varias pechugas en la zona de toma, todas con la misma cara
hacia arriba, y apretar la tecla de esa cara: se guarda un recorte de cada
pieza tomable. Repetir moviendo las piezas y con la otra cara. Con 'g' se
generan las referencias a partir de todas las muestras guardadas (también las
de sesiones anteriores) y se informa cuántas muestras clasificarían bien.

Teclas:
    c   ver las piezas detectadas (sin guardar)
    1   guardar las piezas como PIEL arriba
    2   guardar las piezas como HUESO arriba
    g   generar las referencias con todas las muestras

Las muestras y las referencias quedan en ``references/<camera_id>/``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from gt1 import paths
from gt1.config import ConfigError
from gt1.config.schemas import HistogramFaceClassifierConfig, StrictModel, ViewZoneName
from gt1.domain import Face
from gt1.geometry import FloatArray
from gt1.hardware.camera import Camera
from gt1.perception.face.histogram import build_references, compare, piece_histogram
from gt1.perception.face.references import ReferenceStore
from gt1.perception.image_ops import erode
from gt1.perception.scene import ScenePerception
from gt1.runtime.console import OperatorConsole
from gt1.runtime.factories import build_camera, build_scene_perception
from gt1.runtime.overlay import draw_scene
from gt1.runtime.program import Program, run_program
from gt1.runtime.runner import Job
from gt1.runtime.station import StationHardware

PROGRAM = "build_face_references"
logger = logging.getLogger(__name__)


class Components(StrictModel):
    camera: str
    piece_detector: str
    face_classifier: str


class Settings(StrictModel):
    zone: ViewZoneName


class BuildFaceReferencesConfig(StrictModel):
    components: Components
    settings: Settings


@dataclass
class ReferenceBuilder:
    camera: Camera
    perception: ScenePerception
    classifier_config: HistogramFaceClassifierConfig
    store: ReferenceStore
    console: OperatorConsole

    def capture_job(self, label: Face | None) -> Job:
        """Capturar y mostrar las piezas; con ``label``, guardarlas como muestras de esa cara."""

        def run(stop: threading.Event) -> None:
            scene = self.perception.perceive(self.camera.capture())
            self.console.show(draw_scene(scene))
            if label is not None:
                for detection in scene.pickable:
                    self.store.save_sample(scene.frame, detection.piece, label)
            self.console.set_status(
                f"{len(scene.pickable)} piezas tomables de {len(scene.detections)}"
                f"{f', guardadas como {label.value}' if label else ''} | {self._counts()}"
            )

        return run

    def generate_job(self, stop: threading.Event) -> None:
        config = self.classifier_config
        histograms = {face: self._histograms(face) for face in (Face.SKIN, Face.BONE)}
        references = build_references(
            histograms[Face.SKIN], histograms[Face.BONE], config.color_space, config.bins_per_channel
        )
        path = self.store.save_references(references)
        hits = {
            face: sum(compare(histogram, references, config.min_margin).face == face for histogram in samples)
            for face, samples in histograms.items()
        }
        summary = ", ".join(f"{face.value} {hits[face]}/{len(histograms[face])}" for face in histograms)
        logger.info("Referencias guardadas en %s. Muestras bien clasificadas: %s.", path, summary)
        self.console.set_status(f"Referencias generadas. Bien clasificadas: {summary}")

    def _histograms(self, face: Face) -> list[FloatArray]:
        config = self.classifier_config
        return [
            piece_histogram(
                image, erode(mask, config.mask_erosion_px), config.color_space, config.bins_per_channel
            )
            for image, mask in self.store.load_samples(face)
        ]

    def _counts(self) -> str:
        return (
            f"muestras: piel {self.store.sample_count(Face.SKIN)}, hueso {self.store.sample_count(Face.BONE)}"
        )


def build_face_references(program: Program[BuildFaceReferencesConfig]) -> None:
    components = program.config.components
    repository = program.repository
    camera_config = repository.camera(components.camera)
    classifier_config = repository.face_classifier(components.face_classifier)
    if not isinstance(classifier_config, HistogramFaceClassifierConfig):
        raise ConfigError(
            f"face_classifier = '{components.face_classifier}' es de tipo '{classifier_config.kind}': "
            "las referencias son solo para el clasificador 'histogram'."
        )
    # Sin clasificador: las referencias se están generando y las piezas se muestran sin cara.
    perception = build_scene_perception(
        repository, camera_config, components.piece_detector, None, program.config.settings.zone
    )
    store = ReferenceStore(paths.REFERENCES_DIR, camera_config.camera_id)

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
        builder = ReferenceBuilder(camera, perception, classifier_config, store, console)
        console.bind_job("c", "Ver piezas (sin guardar)", builder.capture_job(None))
        console.bind_job("1", "Guardar como PIEL arriba", builder.capture_job(Face.SKIN))
        console.bind_job("2", "Guardar como HUESO arriba", builder.capture_job(Face.BONE))
        console.bind_job("g", "Generar las referencias", builder.generate_job)
        console.start_job("ver", builder.capture_job(None))
        console.run()


def main() -> None:
    run_program("tools", PROGRAM, BuildFaceReferencesConfig, build_face_references)


if __name__ == "__main__":
    main()

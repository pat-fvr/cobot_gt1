"""face_test: ¿la cámara distingue la piel del hueso?

Protocolo: poner varias pechugas en la zona de toma, todas con la misma cara
hacia arriba, y apretar la tecla de esa cara. Cambiar las piezas de lugar y
repetir; luego darlas vuelta y repetir con la otra tecla. Cada captura queda
registrada con la cara esperada, la detectada y su margen, y la barra de
estado muestra el acierto acumulado.

Teclas:
    c        ver la clasificación (sin registrar)
    1        registrar: todas las piezas están con la PIEL arriba
    2        registrar: todas las piezas están con el HUESO arriba
    espacio  registrar sin etiqueta

Requiere la calibración de la cámara y las referencias de cara.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from gt1.config.schemas import StrictModel, ViewZoneName
from gt1.domain import Face, FaceResult, PieceStatus
from gt1.hardware.camera import Camera
from gt1.perception.scene import Detection, ScenePerception
from gt1.runtime.console import OperatorConsole
from gt1.runtime.factories import build_camera, build_face_classifier, build_scene_perception
from gt1.runtime.overlay import draw_scene
from gt1.runtime.program import Program, run_program
from gt1.runtime.run_log import JsonValue, RunLog
from gt1.runtime.runner import Job
from gt1.runtime.station import StationHardware

PROGRAM = "face_test"


class Components(StrictModel):
    camera: str
    piece_detector: str
    face_classifier: str


class Settings(StrictModel):
    zone: ViewZoneName


class FaceTestConfig(StrictModel):
    components: Components
    settings: Settings


@dataclass
class FaceTally:
    """Acierto acumulado sobre las piezas etiquetadas."""

    correct: int = 0
    wrong: int = 0
    uncertain: int = 0
    skin_labeled: int = 0
    bone_labeled: int = 0

    def add(self, label: Face, result: FaceResult) -> bool | None:
        """Suma una pieza; devuelve si acertó (None si quedó incierta)."""
        if label == Face.SKIN:
            self.skin_labeled += 1
        else:
            self.bone_labeled += 1
        if result.face == Face.UNCERTAIN:
            self.uncertain += 1
            return None
        hit = result.face == label
        if hit:
            self.correct += 1
        else:
            self.wrong += 1
        return hit

    def summary(self) -> str:
        decided = self.correct + self.wrong
        accuracy = f"{100.0 * self.correct / decided:.1f} %" if decided else "-"
        return (
            f"acierto {accuracy} ({self.correct}/{decided}), inciertas {self.uncertain} | "
            f"piel {self.skin_labeled}, hueso {self.bone_labeled}"
        )


@dataclass
class FaceTest:
    camera: Camera
    perception: ScenePerception
    console: OperatorConsole
    run_log: RunLog
    tally: FaceTally = field(default_factory=FaceTally)

    def job(self, label: Face | None, record: bool) -> Job:
        """Trabajo: capturar, clasificar, mostrar y (si ``record``) registrar con la etiqueta ``label``."""

        def run(stop: threading.Event) -> None:
            scene = self.perception.perceive(self.camera.capture())
            perception_ms = scene.perception_s * 1000.0
            overlay = draw_scene(scene)
            self.console.show(overlay)
            if not record:
                self.console.set_status(
                    f"{len(scene.detections)} piezas, percepción {perception_ms:.0f} ms (sin registrar)"
                )
                return
            pieces = [self._piece_record(detection, label) for detection in scene.detections]
            attempt = self.run_log.record(
                {
                    "label": None if label is None else label.value,
                    "perception_ms": perception_ms,
                    "pieces": pieces,
                },
                {"capture": scene.frame.image, "overlay": overlay},
            )
            self.console.set_status(
                f"#{attempt}: {len(scene.detections)} piezas, {perception_ms:.0f} ms | {self.tally.summary()}"
            )

        return run

    def _piece_record(self, detection: Detection, label: Face | None) -> JsonValue:
        piece, result = detection.piece, detection.face
        record: dict[str, JsonValue] = {
            "status": piece.status.value,
            "center_mm": [round(float(value), 1) for value in piece.center_mm],
            "axis_angle_deg": round(piece.axis_angle_deg, 1),
            "length_mm": round(piece.length_mm, 1),
            "width_mm": round(piece.width_mm, 1),
            "area_mm2": round(piece.area_mm2),
        }
        if result is not None:
            # El acierto (criterio R2) se mide sobre las piezas que el robot tomaría: las del borde se
            # registran con su cara, pero no cuentan.
            pickable = piece.status == PieceStatus.OK
            record |= {
                "face": result.face.value,
                "margin": round(result.margin, 4),
                "measurements": {name: round(value, 4) for name, value in result.measurements.items()},
                "correct": self.tally.add(label, result) if label is not None and pickable else None,
            }
        return record


def face_test(program: Program[FaceTestConfig]) -> None:
    components = program.config.components
    repository = program.repository
    camera_config = repository.camera(components.camera)
    classifier = build_face_classifier(
        repository.face_classifier(components.face_classifier), camera_config.camera_id
    )
    perception = build_scene_perception(
        repository, camera_config, components.piece_detector, classifier, program.config.settings.zone
    )

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
        test = FaceTest(camera, perception, console, program.run_log)
        console.bind_job("c", "Ver clasificacion (sin registrar)", test.job(None, record=False))
        console.bind_job("1", "Registrar: todas con PIEL arriba", test.job(Face.SKIN, record=True))
        console.bind_job("2", "Registrar: todas con HUESO arriba", test.job(Face.BONE, record=True))
        console.bind_job("space", "Registrar sin etiqueta", test.job(None, record=True))
        console.start_job("ver", test.job(None, record=False))
        console.run()


def main() -> None:
    run_program("tests", PROGRAM, FaceTestConfig, face_test)


if __name__ == "__main__":
    main()

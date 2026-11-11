"""pick_place_test: el robot toma una pechuga con la piel arriba y la deja en la bandeja fija.

Las piezas se buscan en la zona de toma (mitad superior de la escena) y la
bandeja está en un lugar fijo (``tray_placement`` en ``cell.toml``, mitad
inferior). Antes de cada captura el robot va a la pose fuera de la vista
(``pose`` de ``[settings.clear_view]``, enseñada con freedrive en
``robot_poses.toml``), para que ni él ni su sombra tapen las piezas. Las piezas con
el hueso arriba, inciertas, en el borde o en grupo no se toman: se cuentan en
el registro.

Protocolo:
    1. Bandeja vacía en su lugar y pechugas con la piel arriba en la zona de toma.
    2. 'c' muestra la pieza elegida y el lado largo del gripper sobre ella, sin tomarla.
    3. 'p' hace un intento: toma la pieza, la deja en la bandeja, sale de la vista y verifica
       con la cámara que quedó una pieza con la piel arriba en la bandeja.
    4. Sacar la pieza de la bandeja, devolverla a la zona de toma en otra posición y repetir.

Teclas:
    c   ver la pieza elegida (el robot solo va a la pose fuera de la vista)
    p   intento: tomar, dejar en la bandeja y verificar
    a   abrir el gripper (si un intento se detuvo con la pieza tomada)
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass

from pydantic import Field

from gt1.calibration import PixelProjector
from gt1.config.schemas import ClearViewSettings, StrictModel, WideSide, ZoneName
from gt1.domain import Face, GripperCommand, PieceStatus, Plan, TrayDetection, joined_plan
from gt1.domain.frame import Image
from gt1.perception.scene import Detection, Scene, ScenePerception
from gt1.planning import PickStrategy, PlaceStrategy
from gt1.runtime.clear_view import ClearViewCamera
from gt1.runtime.console import OperatorConsole
from gt1.runtime.executor import PlanExecutor
from gt1.runtime.factories import (
    build_camera,
    build_face_classifier,
    build_fixed_tray,
    build_gripper,
    build_pick_strategy,
    build_place_strategy,
    build_projector,
    build_robot,
    build_scene_perception,
    gripper_wide_side,
)
from gt1.runtime.overlay import draw_gripper_axis, draw_scene, draw_tray
from gt1.runtime.program import Program, run_program
from gt1.runtime.records import detection_record, execution_record
from gt1.runtime.run_log import JsonValue, RunLog
from gt1.runtime.station import StationHardware

PROGRAM = "pick_place_test"
logger = logging.getLogger(__name__)


class Components(StrictModel):
    camera: str
    robot: str
    gripper: str
    tray: str
    piece_detector: str
    face_classifier: str
    pick_strategy: str
    place_strategy: str


class Settings(StrictModel):
    pick_zone: ZoneName = Field(description="Zona donde se buscan las piezas.")
    tray_zone: ZoneName = Field(
        description="Zona que contiene la bandeja fija: ahí se verifica que quedó la pieza."
    )
    clear_view: ClearViewSettings


class PickPlaceTestConfig(StrictModel):
    components: Components
    settings: Settings


@dataclass(frozen=True)
class CellPerception:
    """Percepción de la zona de toma y de la de la bandeja."""

    pick: ScenePerception
    tray: ScenePerception


def choose_piece(scene: Scene) -> Detection | None:
    """La pieza tomable con la piel arriba de la que el clasificador está más seguro."""
    skin_up = [
        detection
        for detection in scene.pickable
        if detection.face is not None and detection.face.face == Face.SKIN
    ]
    return max(skin_up, key=lambda detection: detection.face.margin if detection.face else 0.0, default=None)


def skipped_pieces(scene: Scene, chosen: Detection | None) -> dict[str, int]:
    """Por qué no se tomó cada una de las otras piezas: su estado, o su cara si era tomable."""
    reasons = Counter(
        detection.piece.status.value
        if detection.piece.status != PieceStatus.OK or detection.face is None
        else detection.face.face.value
        for detection in scene.detections
        if detection is not chosen
    )
    return dict(reasons)


def placed_skin_up(tray_scene: Scene) -> bool:
    """Una sola pieza en la zona de la bandeja, entera dentro de ella y con la piel arriba."""
    if len(tray_scene.detections) != 1:
        return False
    detection = tray_scene.detections[0]
    face_up = detection.face.face if detection.face is not None else None
    return detection.piece.status == PieceStatus.OK and face_up == Face.SKIN


class PickPlaceTest:
    def __init__(
        self,
        *,
        clear_view: ClearViewCamera,
        executor: PlanExecutor,
        pick: PickStrategy,
        wide_side: WideSide,
        place: PlaceStrategy,
        tray: TrayDetection,
        perception: CellPerception,
        projector: PixelProjector,
        console: OperatorConsole,
        run_log: RunLog,
    ) -> None:
        self._clear_view = clear_view
        self._executor = executor
        self._pick = pick
        self._wide_side: WideSide = wide_side  # sin anotar, pyright lo ensancha a str
        self._place = place
        self._tray = tray
        self._perception = perception
        self._projector = projector
        self._console = console
        self._run_log = run_log
        self._attempts = Counter[str]()

    # ------------------------------------------------------------------ trabajos

    def preview(self, stop: threading.Event) -> None:
        """Captura desde fuera de la vista y muestra la pieza elegida, sin tomarla."""
        frame = self._clear_view.capture(stop)
        if frame is None:
            return
        scene = self._perception.pick.perceive(frame)
        chosen = choose_piece(scene)
        self._console.show(self._draw(scene, chosen))
        self._console.set_status(self._choice_status(scene, chosen))

    def attempt(self, stop: threading.Event) -> None:
        """Un intento: elegir, tomar, dejar en la bandeja, salir de la vista y verificar."""
        frame = self._clear_view.capture(stop)
        if frame is None:
            return
        scene = self._perception.pick.perceive(frame)
        chosen = choose_piece(scene)
        overlay = self._draw(scene, chosen)
        self._console.show(overlay)
        data: dict[str, JsonValue] = {
            "perception_ms": round(scene.perception_s * 1000.0, 1),
            "skipped": dict(skipped_pieces(scene, chosen)),
        }
        images = {"capture": frame.image, "overlay": overlay}
        if chosen is None:
            self._finish("sin_pieza", data, images, self._choice_status(scene, None))
            return
        data["piece"] = detection_record(chosen)
        # Un solo plan: la subida pasa en curva al viaje a la bandeja (blend_radius_mm de la toma).
        plans = (self._pick.plan(chosen.piece), self._place.plan(chosen.piece, self._tray))
        execution = self._executor.execute(joined_plan("tomar y dejar", plans), stop)
        data["execution"] = execution_record(execution)
        if not execution.completed:
            message = f"{execution.describe()}. Si quedó con la pieza, 'a' abre el gripper."
            self._finish(execution.outcome.value, data, images, message)
            return
        self._verify(data, images, stop)

    def open_gripper(self, stop: threading.Event) -> None:
        """Abre el gripper con el robot quieto donde esté."""
        plan = Plan(name="abrir el gripper", steps=(GripperCommand("release", "abrir el gripper"),))
        self._console.set_status(self._executor.execute(plan, stop).describe())

    # ------------------------------------------------------------------ pasos

    def _verify(self, data: dict[str, JsonValue], images: dict[str, Image], stop: threading.Event) -> None:
        """Desde fuera de la vista, revisa con la cámara que la pieza quedó en la bandeja."""
        after = self._clear_view.capture(stop)
        if after is None:
            self._finish(
                "sin_verificar", data, images, "Pieza dejada, pero no se pudo verificar con la camara."
            )
            return
        tray_scene = self._perception.tray.perceive(after)
        success = placed_skin_up(tray_scene)
        data["tray"] = [detection_record(detection) for detection in tray_scene.detections]
        images |= {"after": after.image, "after_overlay": self._draw(tray_scene, None)}
        self._console.show(images["after_overlay"])
        message = (
            "Pieza en la bandeja con la piel arriba." if success else "La bandeja no quedo como se esperaba."
        )
        self._finish("colocada" if success else "mal_colocada", data, images, message)

    def _finish(
        self, outcome: str, data: dict[str, JsonValue], images: dict[str, Image], message: str
    ) -> None:
        self._attempts[outcome] += 1
        attempt = self._run_log.record({"outcome": outcome, **data}, images)
        tally = ", ".join(f"{name} {count}" for name, count in sorted(self._attempts.items()))
        self._console.set_status(f"#{attempt} {outcome}: {message} | {tally}")

    # ------------------------------------------------------------------ dibujo

    def _choice_status(self, scene: Scene, chosen: Detection | None) -> str:
        skipped = ", ".join(f"{reason} {count}" for reason, count in skipped_pieces(scene, chosen).items())
        others = f" Sin tomar: {skipped}." if skipped else ""
        if chosen is None:
            return f"No hay pieza tomable con la piel arriba en la zona.{others}"
        piece = chosen.piece
        x_mm, y_mm = piece.center_mm[:2]
        return f"Elegida: X={x_mm:.0f} Y={y_mm:.0f} mm, {piece.axis_angle_deg:.0f} grados.{others}"

    def _draw(self, scene: Scene, chosen: Detection | None) -> Image:
        canvas = draw_scene(scene)
        draw_tray(canvas, self._tray, self._projector)
        if chosen is not None:
            draw_gripper_axis(canvas, chosen.piece, self._wide_side, self._projector)
        return canvas


def pick_place_test(program: Program[PickPlaceTestConfig]) -> None:
    components, settings = program.config.components, program.config.settings
    repository = program.repository
    cell = repository.cell()
    clear_pose = repository.robot_pose(settings.clear_view.pose)
    camera_config = repository.camera(components.camera)
    classifier = build_face_classifier(
        repository.face_classifier(components.face_classifier), camera_config.camera_id
    )

    def in_zone(zone: ZoneName) -> ScenePerception:
        return build_scene_perception(repository, camera_config, components.piece_detector, classifier, zone)

    perception = CellPerception(pick=in_zone(settings.pick_zone), tray=in_zone(settings.tray_zone))
    gripper_config = repository.gripper(components.gripper)

    with build_camera(camera_config) as camera, build_robot(repository.robot(components.robot)) as robot:
        gripper = build_gripper(components.gripper, gripper_config, robot)
        program.confirm_station(
            StationHardware(
                camera=camera,
                camera_name=components.camera,
                robot=robot,
                robot_name=components.robot,
                gripper=gripper,
                tray_name=components.tray,
            )
        )
        console = OperatorConsole(f"GT1 - {PROGRAM}")
        executor = PlanExecutor(robot, gripper)
        clear_view = ClearViewCamera(
            camera=camera,
            robot=robot,
            executor=executor,
            table_z_mm=cell.table_z_mm,
            clear_pose=clear_pose,
            settings=settings.clear_view,
            report=console.set_status,
        )
        test = PickPlaceTest(
            clear_view=clear_view,
            executor=executor,
            pick=build_pick_strategy(
                repository.pick_strategy(components.pick_strategy), cell, gripper_config
            ),
            wide_side=gripper_wide_side(gripper_config),
            place=build_place_strategy(repository.place_strategy(components.place_strategy), gripper_config),
            tray=build_fixed_tray(cell, repository.tray(components.tray)),
            perception=perception,
            projector=build_projector(camera_config),
            console=console,
            run_log=program.run_log,
        )
        console.bind_job("c", "Ver la pieza elegida (va a la pose fuera de la vista)", test.preview)
        console.bind_job("p", "Intento: tomar, dejar en la bandeja y verificar", test.attempt)
        console.bind_job("a", "Abrir el gripper", test.open_gripper)
        console.run()


def main() -> None:
    run_program("tests", PROGRAM, PickPlaceTestConfig, pick_place_test)


if __name__ == "__main__":
    main()

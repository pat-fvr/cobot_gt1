"""demo_static_tray: el robot carga, una tras otra, las pechugas que ve la cámara en la bandeja fija.

La bandeja llega por una cinta y la detiene un tope (``stopper`` de
``[settings.tray_station]``: 1 cerrado, 0 abierto); un sensor (``tray_sensor``)
dice que llegó. Al iniciar, el programa cierra el tope y verifica que la
salida quedó en 1, o no arranca. El robot suelta la pieza desde una pose fija
sobre ella (``pose`` de ``[settings.clear_view]``, enseñada con freedrive), que
queda fuera de la vista de la cámara: desde ahí también captura, sin moverse.
No hay detector de bandejas: la bandeja está siempre en el mismo lugar.

Cada carga:
    1. Captura desde la pose de soltar y elige la pechuga tomable de más a la izquierda en la
       imagen (``pick_from = "right"``: la de más a la derecha). La izquierda y la derecha se miden
       a lo largo de ``left_to_right_deg``, en la base.
    2. La toma y la lleva a la pose de soltar.
    3. Espera que el tope esté cerrado (que salió la bandeja anterior) y que el sensor vea la
       bandeja, y suelta la pieza.
    4. Abre el tope para que salga la bandeja cargada y vuelve a 1 sin esperarla: el tope se cierra
       en segundo plano (``TrayStopper``) ``stopper_close_delay_s`` después de que el sensor deja de
       verla, mientras el robot captura y va por la pieza siguiente.
Y así, hasta que no quedan pechugas tomables. Las pechugas se buscan en
``pick_zone``: una zona de cell.toml o toda la imagen (``full_view``). Las piezas
en el borde de la zona o en grupo no se toman.

Protocolo:
    1. Pechugas con la piel arriba, separadas, en la zona de toma; la cinta con bandejas vacías.
    2. 'c' muestra la pieza que se tomaría primero, sin tomarla.
    3. 'p' carga hasta que no quedan piezas. 'x' detiene, también mientras espera la bandeja.

Teclas:
    c   ver la pieza que se toma primero (el robot va a la pose de soltar)
    p   cargar las pechugas una tras otra hasta que no quede ninguna
    a   abrir el gripper (si una carga se detuvo con la pieza tomada)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from typing import Literal

from pydantic import Field

from gt1.calibration import PixelProjector
from gt1.config.schemas import (
    ClearViewSettings,
    StrictModel,
    TopDownPickConfig,
    TrayStationSettings,
    ViewZoneName,
    WideSide,
)
from gt1.domain import GripperCommand, Piece, Plan, joined_plan
from gt1.domain.frame import Image
from gt1.geometry import position_along_mm
from gt1.perception.scene import Scene, ScenePerception
from gt1.planning import PickStrategy, tray_station_drop_plan
from gt1.runtime.clear_view import ClearViewCamera
from gt1.runtime.console import OperatorConsole
from gt1.runtime.executor import PlanExecution, PlanExecutor
from gt1.runtime.factories import (
    build_camera,
    build_gripper,
    build_pick_strategy,
    build_projector,
    build_robot,
    build_scene_perception,
    gripper_wide_side,
)
from gt1.runtime.overlay import draw_gripper_axis, draw_scene
from gt1.runtime.program import Program, run_program
from gt1.runtime.records import execution_record, piece_record
from gt1.runtime.run_log import JsonValue, RunLog
from gt1.runtime.station import StationHardware
from gt1.runtime.tray_stopper import TrayStopper

PROGRAM = "demo_static_tray"
logger = logging.getLogger(__name__)

PickFrom = Literal["left", "right"]

# Resultados de una carga que no vienen del ejecutor.
LOADED = "cargada"
NO_PIECES = "sin_piezas"
NOT_CAPTURED = "sin_captura"


class Components(StrictModel):
    camera: str
    robot: str
    gripper: str
    piece_detector: str


class Settings(StrictModel):
    pick_zone: ViewZoneName = Field(
        description='Zona donde se buscan las pechugas: una de cell.toml, o "full_view" (toda la imagen).'
    )
    left_to_right_deg: float = Field(
        description="Dirección en la base (ángulo respecto del eje X) que en la imagen va de izquierda a "
        "derecha: a lo largo de ella se elige la pieza de más a la izquierda o a la derecha."
    )
    pick_from: PickFrom = Field(
        description='"left": se toma primero la pieza de más a la izquierda en la imagen; "right": la de '
        "más a la derecha."
    )
    pick: TopDownPickConfig = Field(
        description="La toma desde arriba: alturas y velocidades, en este archivo y no en config/strategies/ "
        "para verlas junto al resto de la prueba (sobre la plataforma, con la bandeja abajo)."
    )
    clear_view: ClearViewSettings = Field(
        description="Su pose es la de soltar sobre la bandeja fija: fuera de la vista, desde ella se captura."
    )
    tray_station: TrayStationSettings


class DemoStaticTrayConfig(StrictModel):
    components: Components
    settings: Settings


def first_piece(scene: Scene, left_to_right_deg: float, pick_from: PickFrom) -> Piece | None:
    """La pieza tomable de más a la izquierda en la imagen (o de más a la derecha); None si no hay."""
    toward_deg = left_to_right_deg if pick_from == "left" else left_to_right_deg + 180.0
    pickable = [detection.piece for detection in scene.pickable]
    return min(pickable, key=lambda piece: position_along_mm(piece.center_mm, toward_deg), default=None)


class StaticTrayDemo:
    def __init__(
        self,
        *,
        clear_view: ClearViewCamera,
        executor: PlanExecutor,
        pick: PickStrategy,
        wide_side: WideSide,
        drop: Plan,
        stopper: TrayStopper,
        perception: ScenePerception,
        projector: PixelProjector,
        settings: Settings,
        console: OperatorConsole,
        run_log: RunLog,
    ) -> None:
        """``drop`` es el plan de soltar en la bandeja con el robot ya en la pose de soltar."""
        self._clear_view = clear_view
        self._executor = executor
        self._pick = pick
        self._wide_side: WideSide = wide_side  # sin anotar, pyright lo ensancha a str
        self._drop = drop
        self._stopper = stopper
        self._perception = perception
        self._projector = projector
        self._settings = settings
        self._console = console
        self._run_log = run_log
        self._loads = Counter[str]()

    # ------------------------------------------------------------------ trabajos

    def preview(self, stop: threading.Event) -> None:
        """Captura desde la pose de soltar y muestra la pieza que se tomaría primero, sin tomarla."""
        frame = self._clear_view.capture(stop)
        if frame is None:
            return
        scene = self._perception.perceive(frame)
        piece = self._choose(scene)
        self._console.show(self._draw(scene, piece))
        if piece is None:
            self._console.set_status("No hay pechugas tomables en la zona.")
            return
        x_mm, y_mm = piece.center_mm[:2]
        others = len(scene.pickable) - 1
        self._console.set_status(f"Primera: X={x_mm:.0f} Y={y_mm:.0f} mm. Después quedan {others}.")

    def run(self, stop: threading.Event) -> None:
        """Carga una pechuga tras otra hasta que no quede ninguna, se detenga o falle."""
        loaded = 0
        while (outcome := self._load_next(stop, loaded + 1)) == LOADED:
            loaded += 1
        if outcome == NO_PIECES:
            self._report(f"No quedan pechugas tomables: {loaded} cargadas en esta corrida.")

    def open_gripper(self, stop: threading.Event) -> None:
        """Abre el gripper con el robot quieto donde esté."""
        plan = Plan(name="abrir el gripper", steps=(GripperCommand("release", "abrir el gripper"),))
        self._console.set_status(self._executor.execute(plan, stop).describe())

    # ------------------------------------------------------------------ una carga

    def _load_next(self, stop: threading.Event, number: int) -> str:
        """Captura, elige, toma y suelta en la bandeja; el resultado (``LOADED`` si hay que seguir)."""
        started_s = time.perf_counter()
        frame = self._clear_view.capture(stop)
        if frame is None:
            return NOT_CAPTURED  # ClearViewCamera ya informó por qué
        scene = self._perception.perceive(frame)
        piece = self._choose(scene)
        overlay = self._draw(scene, piece)
        self._console.show(overlay)
        if piece is None:
            return NO_PIECES
        self._console.set_status(f"Pieza {number}: tomando ({len(scene.pickable)} tomables).")
        pick = self._executor.execute(self._pick.plan(piece), stop)
        drop = self._drop_in_tray(stop, number) if pick.completed else None
        last = drop if drop is not None else pick
        outcome = LOADED if last.completed else last.outcome.value
        data: dict[str, JsonValue] = {
            "outcome": outcome,
            "perception_ms": round(scene.perception_s * 1000.0, 1),
            "pickable_pieces": len(scene.pickable),
            "pick_from": self._settings.pick_from,
            "piece": piece_record(piece),
            "pick": execution_record(pick),
            "drop": None if drop is None else execution_record(drop),
            "cycle_s": round(time.perf_counter() - started_s, 2),
        }
        attempt = self._run_log.record(data, {"capture": frame.image, "overlay": overlay})
        self._loads[outcome] += 1
        self._report(self._load_message(attempt, last))
        return outcome

    def _drop_in_tray(self, stop: threading.Event, number: int) -> PlanExecution:
        """Con la pieza tomada: a la pose de soltar, esperar la bandeja, soltar y dejarla pasar."""
        sensor = self._settings.tray_station.tray_sensor.digital_input().describe()
        self._console.set_status(f"Pieza {number}: a la bandeja; espera la bandeja en {sensor}.")
        plan = joined_plan("llevar y soltar", (self._clear_view.plan_from_here(), self._drop))
        execution = self._executor.execute(plan, stop)
        if execution.completed:
            self._stopper.let_loaded_tray_out()  # sale mientras el robot captura y va por la siguiente
        return execution

    def _choose(self, scene: Scene) -> Piece | None:
        return first_piece(scene, self._settings.left_to_right_deg, self._settings.pick_from)

    # ------------------------------------------------------------------ consola

    @staticmethod
    def _load_message(attempt: int, last: PlanExecution) -> str:
        if last.completed:
            return f"#{attempt} cargada."
        return f"#{attempt} {last.describe()}. Si quedó con la pieza, 'a' abre el gripper."

    def _report(self, message: str) -> None:
        tally = ", ".join(f"{name} {count}" for name, count in sorted(self._loads.items()))
        self._console.set_status(f"{message} | {tally}" if tally else message)

    def _draw(self, scene: Scene, piece: Piece | None) -> Image:
        canvas = draw_scene(scene)
        if piece is not None:
            draw_gripper_axis(canvas, piece, self._wide_side, self._projector)
        return canvas


def demo_static_tray(program: Program[DemoStaticTrayConfig]) -> None:
    components, settings = program.config.components, program.config.settings
    repository = program.repository
    cell = repository.cell()
    drop_pose = repository.robot_pose(settings.clear_view.pose)
    camera_config = repository.camera(components.camera)
    gripper_config = repository.gripper(components.gripper)
    perception = build_scene_perception(
        repository, camera_config, components.piece_detector, None, settings.pick_zone
    )

    with (
        build_camera(camera_config) as camera,
        build_robot(repository.robot(components.robot)) as robot,
        TrayStopper(robot, settings.tray_station) as stopper,
    ):
        gripper = build_gripper(components.gripper, gripper_config, robot)
        program.confirm_station(
            StationHardware(
                camera=camera,
                camera_name=components.camera,
                robot=robot,
                robot_name=components.robot,
                gripper=gripper,
                tray_name=None,
            )
        )
        stopper.close_and_verify()
        console = OperatorConsole(f"GT1 - {PROGRAM}")
        executor = PlanExecutor(robot, gripper)
        demo = StaticTrayDemo(
            clear_view=ClearViewCamera(
                camera=camera,
                robot=robot,
                executor=executor,
                table_z_mm=cell.table_z_mm,
                clear_pose=drop_pose,
                settings=settings.clear_view,
                report=console.set_status,
            ),
            executor=executor,
            pick=build_pick_strategy(settings.pick, cell, gripper_config),
            wide_side=gripper_wide_side(gripper_config),
            drop=tray_station_drop_plan(settings.tray_station),
            stopper=stopper,
            perception=perception,
            projector=build_projector(camera_config),
            settings=settings,
            console=console,
            run_log=program.run_log,
        )
        console.bind_job("c", "Ver la pieza que se toma primero (va a la pose de soltar)", demo.preview)
        console.bind_job("p", "Cargar las pechugas una tras otra hasta que no quede ninguna", demo.run)
        console.bind_job("a", "Abrir el gripper", demo.open_gripper)
        console.run()


def main() -> None:
    run_program("tests", PROGRAM, DemoStaticTrayConfig, demo_static_tray)


if __name__ == "__main__":
    main()

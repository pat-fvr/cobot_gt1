"""tray_loading_test: el robot carga en las bandejas detectadas las pechugas con la piel arriba.

Una sola cámara ve toda la escena: las pechugas se buscan en la zona de toma
(mitad superior de la imagen) y las bandejas en la zona de bandejas (mitad
inferior), sin escena vacía. Cada pechuga se toma por su centro, a lo ancho,
y se deja en el centro de una bandeja libre con su línea central a lo largo del
lado largo de la bandeja y los lóbulos hacia ``lobes_toward_deg`` de la
estrategia de colocación.

'c' decide todo con una sola imagen, tomada con el robot fuera de la vista, y
'p' ejecuta lo que decidió 'c', de inmediato y sin otra captura. Las pechugas
tomables y las bandejas libres se ordenan a lo largo de
``load_order_toward_deg`` (de izquierda a derecha en la imagen) y se cargan en
ese orden: la primera pechuga en la primera bandeja, la segunda en la segunda.
Si hay más pechugas que bandejas libres, las últimas se quedan.
Una bandeja con una pieza adentro no está libre. Con todas las cargas
asignadas y planificadas, el robot las hace una tras otra sin volver fuera de
la vista: después de soltar una pieza va directo a la siguiente. Por eso las
pechugas tienen que estar separadas: si al tomar una se mueve otra, se busca
donde estaba. Tomar y dejar son un solo plan por carga, para que el robot pase
en curva de la subida al viaje a la bandeja (``blend_radius_mm`` de la toma).

Al terminar, el robot vuelve fuera de la vista y, con ``verify_loads``,
captura y verifica cada carga: una pechuga con la piel arriba dentro de la
bandeja, con los lóbulos hacia donde se pidió. Si el robot no completa una
carga, el ciclo se corta ahí con el robot detenido donde paró.

Protocolo:
    1. Bandejas vacías en la mitad inferior de la mesa; pechugas con la piel arriba, separadas, en la
       superior.
    2. 'c' captura desde fuera de la vista y calcula las cargas: muestra las piezas, las bandejas y
       qué pieza va a cuál. Sin mover nada en la mesa hasta apretar 'p'.
    3. 'p' ejecuta esas cargas seguidas y vuelve fuera de la vista. Lo calculado se usa una sola vez:
       el ciclo siguiente empieza con otra 'c'.
    4. Vaciar las bandejas, devolver las piezas en otras posiciones y repetir.

Teclas:
    c   capturar y calcular las cargas (el robot va a la pose fuera de la vista)
    p   ejecutar las cargas calculadas con 'c', y la verificación si está activada
    a   abrir el gripper (si una carga se detuvo con la pieza tomada)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from pydantic import Field

from gt1.calibration import PixelProjector
from gt1.config.schemas import ClearViewSettings, StrictModel, WideSide, ZoneName
from gt1.domain import Frame, GripperCommand, Piece, Plan, TrayDetection, Zone, joined_plan
from gt1.domain.frame import Image
from gt1.geometry import axis_angle_nearest_deg, wrap_angle_deg
from gt1.perception.scene import Scene, ScenePerception
from gt1.perception.tray import TrayDetector, tray_contains
from gt1.planning import PickStrategy, PlaceStrategy, pair_pieces_with_trays
from gt1.runtime.clear_view import ClearViewCamera
from gt1.runtime.console import OperatorConsole
from gt1.runtime.executor import PlanExecution, PlanExecutor
from gt1.runtime.factories import (
    build_camera,
    build_gripper,
    build_pick_strategy,
    build_place_strategy,
    build_projector,
    build_robot,
    build_scene_perception,
    build_tray_detector,
    build_zone,
    gripper_wide_side,
)
from gt1.runtime.overlay import (
    draw_direction,
    draw_gripper_axis,
    draw_lobes_direction,
    draw_scene,
    draw_tray,
    draw_zone,
)
from gt1.runtime.program import Program, run_program
from gt1.runtime.records import detection_record, execution_record, piece_record, tray_record
from gt1.runtime.run_log import JsonValue, RunLog
from gt1.runtime.station import StationHardware

PROGRAM = "tray_loading_test"
logger = logging.getLogger(__name__)

# Dibujo: la línea que une cada pieza con su bandeja y el largo de la flecha del sentido pedido.
PAIRING_COLOR = (255, 255, 255)
TARGET_ARROW_MM = 90.0


class Components(StrictModel):
    camera: str
    robot: str
    gripper: str
    tray: str
    piece_detector: str
    tray_detector: str
    pick_strategy: str
    place_strategy: str


class Settings(StrictModel):
    pick_zone: ZoneName = Field(description="Zona donde se buscan las pechugas.")
    tray_zone: ZoneName = Field(description="Zona donde se buscan las bandejas.")
    load_order_toward_deg: float = Field(
        description="Dirección en la base (ángulo respecto del eje X) en que avanzan las cargas: las piezas "
        "y las bandejas libres se ordenan a lo largo de ella y la primera pieza va a la primera bandeja."
    )
    verify_loads: bool = Field(
        description="Al terminar el ciclo, capturar desde fuera de la vista y verificar cada carga."
    )
    max_lobes_error_deg: float = Field(
        gt=0,
        le=180,
        description="Con verify_loads: diferencia máxima entre hacia dónde quedaron los lóbulos en la "
        "bandeja y hacia dónde se pidieron, para dar la carga por buena.",
    )
    clear_view: ClearViewSettings


class TrayLoadingTestConfig(StrictModel):
    components: Components
    settings: Settings


@dataclass(frozen=True, eq=False)
class Loading:
    """Lo que vio la cámara: piezas por tomar, bandejas y lo que ya hay en ellas."""

    frame: Frame
    pieces: Scene  # zona de toma
    trays: tuple[TrayDetection, ...]
    tray_zone: Scene  # piezas en la zona de bandejas
    perception_s: float


@dataclass(frozen=True)
class LoadingPerception:
    """Piezas en la zona de toma, bandejas en la de bandejas y piezas dentro de ellas."""

    pieces: ScenePerception
    tray_contents: ScenePerception
    trays: TrayDetector
    tray_zone: Zone

    def perceive(self, frame: Frame) -> Loading:
        pieces = self.pieces.perceive(frame)
        tray_contents = self.tray_contents.perceive(frame)
        started = time.perf_counter()
        trays = tuple(self.trays.detect(frame, self.tray_zone))
        trays_s = time.perf_counter() - started
        perception_s = pieces.perception_s + tray_contents.perception_s + trays_s
        return Loading(frame, pieces, trays, tray_contents, perception_s)


@dataclass(frozen=True, eq=False)
class Load:
    """Una carga del ciclo, con su plan (tomar y dejar) calculado antes de mover el robot."""

    piece: Piece
    tray: TrayDetection
    target_lobes_deg: float
    plan: Plan


@dataclass(frozen=True, eq=False)
class PlannedCycle:
    """Lo que calculó 'c' para que 'p' lo ejecute: lo que vio la cámara, su dibujo y las cargas en orden."""

    loading: Loading
    overlay: Image
    loads: tuple[Load, ...]


def free_trays(loading: Loading) -> list[TrayDetection]:
    """Las bandejas sin una pieza adentro."""
    occupied = [detection.piece.center_mm for detection in loading.tray_zone.detections]
    return [tray for tray in loading.trays if not any(tray_contains(tray, point) for point in occupied)]


def assign_loads(loading: Loading, order_toward_deg: float) -> list[tuple[Piece, TrayDetection]]:
    """Cada pieza tomable con su bandeja libre, en el orden de carga; las piezas que sobran se quedan."""
    pickable = [detection.piece for detection in loading.pieces.pickable]
    return pair_pieces_with_trays(pickable, free_trays(loading), order_toward_deg)


def lobes_error_deg(piece: Piece, target_lobes_deg: float) -> float | None:
    """Cuánto se apartan los lóbulos de la pieza del sentido pedido; None si no se midieron."""
    if piece.lobes_direction_deg is None:
        return None
    return abs(wrap_angle_deg(piece.lobes_direction_deg - target_lobes_deg))


def verified_outcome(
    after: Loading, load: Load, max_lobes_error_deg: float
) -> tuple[str, dict[str, JsonValue]]:
    """Resultado de una carga según la captura de después, y lo medido para el registro."""
    inside = [
        detection
        for detection in after.tray_zone.detections
        if tray_contains(load.tray, detection.piece.center_mm)
    ]
    measured: dict[str, JsonValue] = {"in_tray": [detection_record(detection) for detection in inside]}
    if len(inside) != 1:
        return "sin_pieza_en_bandeja", measured
    error_deg = lobes_error_deg(inside[0].piece, load.target_lobes_deg)
    measured["lobes_error_deg"] = None if error_deg is None else round(error_deg, 1)
    if error_deg is None or error_deg > max_lobes_error_deg:
        return "mal_orientada", measured
    return "cargada", measured


class TrayLoadingTest:
    def __init__(
        self,
        *,
        clear_view: ClearViewCamera,
        executor: PlanExecutor,
        pick: PickStrategy,
        wide_side: WideSide,
        place: PlaceStrategy,
        perception: LoadingPerception,
        lobes_toward_deg: float,
        projector: PixelProjector,
        settings: Settings,
        console: OperatorConsole,
        run_log: RunLog,
    ) -> None:
        self._clear_view = clear_view
        self._executor = executor
        self._pick = pick
        self._wide_side: WideSide = wide_side  # sin anotar, pyright lo ensancha a str
        self._place = place
        self._perception = perception
        self._lobes_toward_deg = lobes_toward_deg
        self._projector = projector
        self._settings = settings
        self._console = console
        self._run_log = run_log
        self._loads = Counter[str]()
        # Lo único que cambia entre trabajos: se escribe con 'c' y se consume con 'p', los dos en el hilo de
        # trabajo, que corre de a un trabajo por vez.
        self._planned: PlannedCycle | None = None

    # ------------------------------------------------------------------ trabajos

    def plan_cycle(self, stop: threading.Event) -> None:
        """Captura desde fuera de la vista y calcula las cargas, en orden, para que 'p' las ejecute."""
        self._planned = None
        frame = self._clear_view.capture(stop)
        if frame is None:
            return
        loading = self._perception.perceive(frame)
        loads = tuple(self._plan_loads(loading))
        overlay = self._draw(loading, [(load.piece, load.tray) for load in loads])
        self._planned = PlannedCycle(loading, overlay, loads)
        self._console.show(overlay)
        next_step = "'p' las ejecuta." if loads else "Nada que cargar."
        self._console.set_status(f"{self._assignment_summary(loading, len(loads))} {next_step}")

    def run_cycle(self, stop: threading.Event) -> None:
        """Ejecuta las cargas calculadas con 'c', vuelve fuera de la vista y verifica si está activado."""
        planned, self._planned = self._planned, None  # al mover piezas, lo calculado deja de valer
        if planned is None:
            self._console.set_status("No hay cargas calculadas: 'c' captura y las calcula.")
            return
        self._console.show(planned.overlay)
        images = {"capture": planned.loading.frame.image, "overlay": planned.overlay}
        results = self._execute(planned.loads, stop)
        after = self._leave_view(results, stop) if results else None
        if after is not None:
            images |= {"after": after.frame.image, "after_overlay": self._draw(after, [])}
            self._console.show(images["after_overlay"])
        self._record_cycle(planned.loading, results, after, images)

    def open_gripper(self, stop: threading.Event) -> None:
        """Abre el gripper con el robot quieto donde esté."""
        plan = Plan(name="abrir el gripper", steps=(GripperCommand("release", "abrir el gripper"),))
        self._console.set_status(self._executor.execute(plan, stop).describe())

    # ------------------------------------------------------------------ el ciclo

    def _plan_loads(self, loading: Loading) -> list[Load]:
        """Todas las cargas de la captura con sus planes: si una no se puede planificar, no se mueve nada."""
        return [
            Load(
                piece=piece,
                tray=tray,
                target_lobes_deg=axis_angle_nearest_deg(tray.axis_angle_deg, self._lobes_toward_deg),
                plan=joined_plan("cargar", (self._pick.plan(piece), self._place.plan(piece, tray))),
            )
            for piece, tray in assign_loads(loading, self._settings.load_order_toward_deg)
        ]

    def _execute(self, loads: Sequence[Load], stop: threading.Event) -> list[tuple[Load, PlanExecution]]:
        """Las cargas una tras otra, sin volver fuera de la vista; se corta en la primera que no termina."""
        results: list[tuple[Load, PlanExecution]] = []
        for load in loads:
            execution = self._executor.execute(load.plan, stop)
            results.append((load, execution))
            if not execution.completed:
                break
        return results

    def _leave_view(
        self, results: Sequence[tuple[Load, PlanExecution]], stop: threading.Event
    ) -> Loading | None:
        """Con todas las cargas hechas, vuelve fuera de la vista; con ``verify_loads``, lo que ve desde ahí.

        Si una carga se cortó, el robot se queda donde paró: moverlo sin revisar podría empeorar un choque.
        """
        if not all(execution.completed for _, execution in results):
            return None
        if not self._settings.verify_loads:
            self._clear_view.move_clear(stop)
            return None
        after = self._clear_view.capture(stop)
        return None if after is None else self._perception.perceive(after)

    # ------------------------------------------------------------------ registro

    def _record_cycle(
        self,
        loading: Loading,
        results: Sequence[tuple[Load, PlanExecution]],
        after: Loading | None,
        images: dict[str, Image],
    ) -> None:
        outcomes = [self._load_outcome(execution, load, after) for load, execution in results]
        load_records: list[JsonValue] = [
            {"outcome": outcome, **self._load_record(load, execution), **measured}
            for (load, execution), (outcome, measured) in zip(results, outcomes, strict=True)
        ]
        attempt = self._run_log.record(
            {
                "perception_ms": round(loading.perception_s * 1000.0, 1),
                "pickable_pieces": len(loading.pieces.pickable),
                "trays_seen": len(loading.trays),
                "free_trays": len(free_trays(loading)),
                "verified": self._settings.verify_loads,
                "loads": load_records,
            },
            images,
        )
        cycle_tally = Counter(outcome for outcome, _ in outcomes)
        self._loads.update(cycle_tally)
        self._report(f"Ciclo #{attempt}: {self._cycle_message(loading, results, cycle_tally)}")

    def _load_outcome(
        self, execution: PlanExecution, load: Load, after: Loading | None
    ) -> tuple[str, dict[str, JsonValue]]:
        if not execution.completed:
            return execution.outcome.value, {}
        if not self._settings.verify_loads:
            return "dejada", {}
        if after is None:
            return "sin_verificar", {}
        return verified_outcome(after, load, self._settings.max_lobes_error_deg)

    @staticmethod
    def _load_record(load: Load, execution: PlanExecution) -> dict[str, JsonValue]:
        return {
            "piece": piece_record(load.piece),
            "tray": tray_record(load.tray),
            "target_lobes_deg": round(load.target_lobes_deg, 1),
            "execution": execution_record(execution),
        }

    def _cycle_message(
        self,
        loading: Loading,
        results: Sequence[tuple[Load, PlanExecution]],
        cycle_tally: Counter[str],
    ) -> str:
        if not results:
            return f"nada que cargar. {self._assignment_summary(loading, 0)}"
        outcomes = ", ".join(f"{name} {count}" for name, count in sorted(cycle_tally.items()))
        _, last = results[-1]
        if not last.completed:
            return f"{outcomes}. {last.describe()}. Si quedó con la pieza, 'a' abre el gripper."
        return f"{len(results)} cargas: {outcomes}."

    @staticmethod
    def _assignment_summary(loading: Loading, loads: int) -> str:
        pickable = len(loading.pieces.pickable)
        return (
            f"{pickable} piezas para tomar, {len(free_trays(loading))} de {len(loading.trays)} bandejas "
            f"libres: {loads} cargas."
        )

    def _report(self, message: str) -> None:
        tally = ", ".join(f"{name} {count}" for name, count in sorted(self._loads.items()))
        self._console.set_status(f"{message} | {tally}" if tally else message)

    # ------------------------------------------------------------------ dibujo

    def _draw(self, loading: Loading, pairs: Sequence[tuple[Piece, TrayDetection]]) -> Image:
        """Piezas con su línea central, bandejas con el sentido pedido y cada pieza unida a su bandeja."""
        canvas = draw_scene(loading.pieces)
        draw_zone(canvas, loading.tray_zone.zone_polygon_px, loading.tray_zone.zone.name)
        for detection in (*loading.pieces.detections, *loading.tray_zone.detections):
            draw_lobes_direction(canvas, detection.piece, self._projector)
        for tray in loading.trays:
            draw_tray(canvas, tray, self._projector, f"{tray.measurements.get('line_contrast_pct', 0):.0f}%")
        for number, (piece, tray) in enumerate(pairs, start=1):
            draw_gripper_axis(canvas, piece, self._wide_side, self._projector)
            target_deg = axis_angle_nearest_deg(tray.axis_angle_deg, self._lobes_toward_deg)
            draw_direction(canvas, tray.center_mm, target_deg, TARGET_ARROW_MM, self._projector)
            self._draw_pairing(canvas, piece, tray, number)
        return canvas

    def _draw_pairing(self, canvas: Image, piece: Piece, tray: TrayDetection, number: int) -> None:
        (piece_x, piece_y), (tray_x, tray_y) = self._projector.base_to_pixels(
            np.stack([piece.center_mm, tray.center_mm])
        )
        start, end = (round(piece_x), round(piece_y)), (round(tray_x), round(tray_y))
        cv2.line(canvas, start, end, PAIRING_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(number), start, cv2.FONT_HERSHEY_SIMPLEX, 1.0, PAIRING_COLOR, 2, cv2.LINE_AA)


def build_loading_perception(
    program: Program[TrayLoadingTestConfig],
) -> tuple[LoadingPerception, PixelProjector]:
    """La percepción de la carga y el proyector de la cámara con que mide."""
    components, settings = program.config.components, program.config.settings
    repository = program.repository
    camera_config = repository.camera(components.camera)
    projector = build_projector(camera_config)

    def in_zone(zone: ZoneName) -> ScenePerception:
        return build_scene_perception(repository, camera_config, components.piece_detector, None, zone)

    trays = build_tray_detector(
        repository.tray_detector(components.tray_detector), repository.tray(components.tray), projector
    )
    perception = LoadingPerception(
        pieces=in_zone(settings.pick_zone),
        tray_contents=in_zone(settings.tray_zone),
        trays=trays,
        tray_zone=build_zone(repository.cell(), settings.tray_zone),
    )
    return perception, projector


def tray_loading_test(program: Program[TrayLoadingTestConfig]) -> None:
    components, settings = program.config.components, program.config.settings
    repository = program.repository
    cell = repository.cell()
    clear_pose = repository.robot_pose(settings.clear_view.pose)
    perception, projector = build_loading_perception(program)
    place_config = repository.place_strategy(components.place_strategy)
    gripper_config = repository.gripper(components.gripper)
    camera_config = repository.camera(components.camera)
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
        test = TrayLoadingTest(
            clear_view=ClearViewCamera(
                camera=camera,
                robot=robot,
                executor=executor,
                table_z_mm=cell.table_z_mm,
                clear_pose=clear_pose,
                settings=settings.clear_view,
                report=console.set_status,
            ),
            executor=executor,
            pick=build_pick_strategy(
                repository.pick_strategy(components.pick_strategy), cell, gripper_config
            ),
            wide_side=gripper_wide_side(gripper_config),
            place=build_place_strategy(place_config, gripper_config),
            perception=perception,
            lobes_toward_deg=place_config.lobes_toward_deg,
            projector=projector,
            settings=settings,
            console=console,
            run_log=program.run_log,
        )
        console.bind_job(
            "c", "Capturar y calcular las cargas (va a la pose fuera de la vista)", test.plan_cycle
        )
        console.bind_job("p", "Ejecutar las cargas calculadas con 'c'", test.run_cycle)
        console.bind_job("a", "Abrir el gripper", test.open_gripper)
        console.run()


def main() -> None:
    run_program("tests", PROGRAM, TrayLoadingTestConfig, tray_loading_test)


if __name__ == "__main__":
    main()

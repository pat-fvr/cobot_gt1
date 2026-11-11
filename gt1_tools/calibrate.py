"""calibrate: modelo de la lente y relación cámara-robot.

Al iniciar, un menú en la terminal:

    0  generar la imagen del tablero para imprimir
    1  lente (intrínsecos): mostrar el tablero a la cámara en muchas posiciones
    2  cámara-robot: tablero sobre la mesa y cinco esquinas tocadas con el pincho

Paso 1, teclas:
    espacio  agregar la vista actual del tablero
    c        calcular y guardar el modelo de la lente
    v        volver a la imagen en vivo

Paso 2, teclas:
    b        fijar el tablero (con el robot fuera de la vista)
    f        activar / desactivar freedrive
    t        registrar el toque de la esquina marcada
    u        deshacer el último toque
    c        calcular y guardar
    v        volver a la imagen en vivo (para mover el tablero y fijarlo de nuevo)

Los archivos quedan en ``calibration/<camera_id>/``. El paso 2 usa el gripper
"pincho" (su TCP es la punta): se carga en el robot al iniciar. El tablero
(ChArUco o de ajedrez) se elige en ``board`` de ``config/tools/calibrate.toml``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import cv2
import numpy as np

from gt1 import paths
from gt1.calibration import CalibrationStore
from gt1.calibration.boards import BoardDetection, CalibrationBoard, touch_corner_ids
from gt1.calibration.extrinsics import TouchPoint, fit_camera_pose, fit_homography, locate_board
from gt1.calibration.intrinsics import MIN_CORNERS_PER_VIEW, Intrinsics, calibrate_intrinsics
from gt1.config.schemas import CalibrationMethod, StrictModel
from gt1.domain import Frame
from gt1.domain.frame import Image
from gt1.geometry import transform_points
from gt1.hardware.camera import Camera
from gt1.hardware.robot import Robot
from gt1.runtime.console import OperatorConsole
from gt1.runtime.factories import build_board, build_camera, build_gripper, build_robot
from gt1.runtime.overlay import to_color
from gt1.runtime.program import Program, run_program
from gt1.runtime.runner import Job
from gt1.runtime.station import StationHardware
from gt1.runtime.terminal import ask_choice

PROGRAM = "calibrate"
logger = logging.getLogger(__name__)

EXPLANATION = """
CALIBRACIÓN DE LA CÁMARA
========================
Responde: este píxel de la imagen, ¿a qué punto del robot corresponde?

Paso 1 - Lente (intrínsecos). Corrige la distorsión del lente. Lo usan los dos métodos del paso 2.
  Se hace una vez por cámara (y de nuevo si se cambia el lente, el foco o la resolución).

Paso 2 - Cámara-robot. Con el tablero sobre la mesa, se tocan cinco esquinas con el pincho.
  Se calculan uno o los dos métodos:

  homography    Homografía plana: una tabla de conversión píxel -> (X, Y) ajustada sobre la mesa.
                Simple, pero exacta solo a la altura del tablero: la parte de arriba de una pechuga
                (~7 cm más alta) se ve desplazada, más cuanto más lejos del centro de la imagen.

  camera_model  Modelo 3D de la cámara: dónde está y hacia dónde apunta respecto del robot.
                Cada píxel es un rayo y se corta a la altura que se pida (mesa, pieza, bandeja),
                sin ese desplazamiento. Recomendado.

Cuál se usa lo decide [calibration] method en config/cameras/<cámara>.toml.
"""


class Components(StrictModel):
    camera: str
    robot: str
    gripper: str


class Settings(StrictModel):
    board: str
    min_lens_views: int
    board_print_pixels_per_mm: float


class CalibrateConfig(StrictModel):
    components: Components
    settings: Settings


def draw_corners(image: Image, detection: BoardDetection | None) -> Image:
    canvas = to_color(image)
    if detection is not None:
        for x, y in detection.pixels_px:
            cv2.circle(canvas, (round(x), round(y)), 5, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas


class LiveBoardView:
    """Imagen en vivo con el tablero detectado; guarda la última imagen y su detección."""

    def __init__(self, camera: Camera, board: CalibrationBoard, console: OperatorConsole) -> None:
        self._camera = camera
        self._board = board
        self._console = console
        self._lock = threading.Lock()
        self._latest: tuple[Frame, BoardDetection | None] | None = None

    def job(self, describe: Callable[[], str]) -> Job:
        """Trabajo de imagen en vivo; ``describe`` da el texto de la barra de estado en cada imagen."""

        def run(stop: threading.Event) -> None:
            while not stop.is_set():
                frame = self._camera.capture()
                detection = self._board.detect(frame.image)
                with self._lock:
                    self._latest = (frame, detection)
                self._console.show(draw_corners(frame.image, detection))
                corners = 0 if detection is None else len(detection.corner_ids)
                self._console.set_status(
                    f"{describe()} | esquinas visibles: {corners}/{self._board.corner_count}"
                )

        return run

    def latest(self) -> tuple[Frame, BoardDetection | None] | None:
        with self._lock:
            return self._latest


class LensCalibration:
    def __init__(
        self, view: LiveBoardView, console: OperatorConsole, store: CalibrationStore, min_views: int
    ) -> None:
        self._view = view
        self._console = console
        self._store = store
        self._min_views = min_views
        self._views: list[BoardDetection] = []
        self._image_size_px: tuple[int, int] | None = None

    def live_job(self) -> Job:
        return self._view.job(
            lambda: f"Lente: {len(self._views)}/{self._min_views} vistas ('espacio' agrega)"
        )

    def add_view(self) -> None:
        latest = self._view.latest()
        detection = None if latest is None else latest[1]
        if latest is None or detection is None or len(detection.corner_ids) < MIN_CORNERS_PER_VIEW:
            self._console.set_status(
                f"No se agregó: se necesitan {MIN_CORNERS_PER_VIEW} esquinas visibles o más."
            )
            return
        self._views.append(detection)
        self._image_size_px = latest[0].size_px

    def compute(self) -> None:
        if len(self._views) < self._min_views or self._image_size_px is None:
            self._console.set_status(f"Faltan vistas: {len(self._views)}/{self._min_views}.")
            return
        self._console.stop_job(timeout_s=2.0)
        self._console.start_job("calcular lente", self._compute_job)

    def _compute_job(self, stop: threading.Event) -> None:
        if self._image_size_px is None:
            return
        self._console.set_status("Calculando el modelo de la lente…")
        intrinsics = calibrate_intrinsics(self._views, self._image_size_px)
        path = self._store.save_intrinsics(intrinsics, len(self._views))
        logger.info(
            "Lente calibrada: error %.3f px con %d vistas -> %s", intrinsics.rms_px, len(self._views), path
        )
        self._console.set_status(
            f"Lente guardada: error {intrinsics.rms_px:.3f} px ({len(self._views)} vistas)."
        )


class RobotCalibration:
    def __init__(
        self,
        *,
        view: LiveBoardView,
        robot: Robot,
        board: CalibrationBoard,
        intrinsics: Intrinsics,
        console: OperatorConsole,
        store: CalibrationStore,
        methods: tuple[CalibrationMethod, ...],
    ) -> None:
        self._view = view
        self._robot = robot
        self._board = board
        self._intrinsics = intrinsics
        self._console = console
        self._store = store
        self._methods = methods
        self._targets = np.zeros(0, dtype=np.int32)  # esquinas a tocar; se eligen al fijar el tablero
        self._fixed: tuple[Frame, BoardDetection] | None = None
        self._touches: list[TouchPoint] = []
        self._freedrive = False

    def live_job(self) -> Job:
        return self._view.job(lambda: "Tablero sobre la mesa y robot fuera de la vista: 'b' para fijarlo")

    def fix_board(self) -> None:
        latest = self._view.latest()
        detection = None if latest is None else latest[1]
        if latest is None or detection is None:
            self._console.set_status("No se ve el tablero.")
            return
        targets = touch_corner_ids(detection)
        if len(set(targets.tolist())) < len(targets):
            self._console.set_status("Se ve muy poco del tablero para elegir las esquinas a tocar.")
            return
        self._console.stop_job(timeout_s=2.0)
        self._fixed = (latest[0], detection)
        self._targets = targets
        self._touches = []
        self._show_targets()

    def toggle_freedrive_job(self, stop: threading.Event) -> None:
        self._robot.set_freedrive(not self._freedrive)
        self._freedrive = not self._freedrive
        self._show_targets()

    def touch_job(self, stop: threading.Event) -> None:
        if self._fixed is None:
            self._console.set_status("Primero fije el tablero con 'b'.")
            return
        _, detection = self._fixed
        if len(self._touches) == len(self._targets):
            self._console.set_status("Ya están todos los toques: 'c' para calcular o 'u' para deshacer.")
            return
        corner_id = int(self._targets[len(self._touches)])
        tip = self._robot.tcp_pose().position_mm
        board_point = detection.board_points_mm[detection.corner_ids.tolist().index(corner_id)]
        self._touches.append(TouchPoint(corner_id=corner_id, board_point_mm=board_point, base_point_mm=tip))
        logger.info("Esquina %d tocada en X=%.1f Y=%.1f Z=%.1f mm.", corner_id, *tip)
        self._show_targets()

    def undo_touch(self) -> None:
        if self._touches:
            self._touches.pop()
        self._show_targets()

    def compute_job(self, stop: threading.Event) -> None:
        if self._fixed is None or len(self._touches) < len(self._targets):
            self._console.set_status(f"Faltan toques: {len(self._touches)}/{len(self._targets)}.")
            return
        if self._freedrive:
            self._robot.set_freedrive(False)
            self._freedrive = False
        _, detection = self._fixed
        board_in_base = locate_board(self._touches)
        results = [f"toques: error {board_in_base.touch_rms_mm:.2f} mm"]
        if "homography" in self._methods:
            homography = fit_homography(self._intrinsics, detection, board_in_base)
            self._store.save_homography(homography)
            results.append(f"homografía: ajuste {homography.fit_rms_mm:.2f} mm")
        if "camera_model" in self._methods:
            camera_model = fit_camera_pose(self._intrinsics, detection, board_in_base)
            self._store.save_camera_model(camera_model)
            results.append(f"modelo de cámara: reproyección {camera_model.reprojection_rms_px:.2f} px")
        board_z_mm = float(
            transform_points(board_in_base.base_from_board, detection.board_points_mm)[:, 2].mean()
        )
        table_z_mm = board_z_mm - self._board.thickness_mm
        logger.info("Calibración cámara-robot guardada en %s: %s.", self._store.directory, "; ".join(results))
        logger.info(
            "Superficie del tablero en Z=%.1f mm -> mesa en Z=%.1f mm (table_z_mm en cell.toml).",
            board_z_mm,
            table_z_mm,
        )
        self._console.set_status(f"Guardado. {' | '.join(results)} | mesa Z={table_z_mm:.1f} mm")

    def _show_targets(self) -> None:
        if self._fixed is None:
            return
        frame, detection = self._fixed
        canvas = draw_corners(frame.image, detection)
        for index, corner_id in enumerate(self._targets.tolist()):
            x, y = detection.pixels_px[detection.corner_ids.tolist().index(corner_id)]
            done, current = index < len(self._touches), index == len(self._touches)
            color = (80, 200, 80) if done else (0, 220, 255) if current else (0, 140, 255)
            cv2.circle(canvas, (round(x), round(y)), 18 if current else 12, color, 3, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(index + 1),
                (round(x) + 16, round(y) - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )
        self._console.show(canvas)
        freedrive = "freedrive ACTIVO" if self._freedrive else "freedrive apagado ('f')"
        if len(self._touches) < len(self._targets):
            next_step = f"toque la esquina {len(self._touches) + 1} con el pincho y apriete 't'"
        else:
            next_step = "'c' para calcular"
        self._console.set_status(
            f"Toques {len(self._touches)}/{len(self._targets)}: {next_step} | {freedrive}"
        )


def calibrate(program: Program[CalibrateConfig]) -> None:
    print(EXPLANATION)
    step = ask_choice(
        "¿Qué paso quiere hacer?",
        {
            "0": "Generar la imagen del tablero para imprimir",
            "1": "Lente (intrínsecos)",
            "2": "Cámara-robot",
        },
    )
    settings = program.config.settings
    board = build_board(program.repository.board(settings.board))
    if step == "0":
        print_board(board, settings)
    elif step == "1":
        calibrate_lens(program, board)
    else:
        options: dict[str, tuple[CalibrationMethod, ...]] = {
            "1": ("homography",),
            "2": ("camera_model",),
            "3": ("homography", "camera_model"),
        }
        choice = ask_choice("¿Qué método calcular?", {"1": "homography", "2": "camera_model", "3": "ambos"})
        methods = options[choice]
        calibrate_camera_to_robot(program, board, methods)


def print_board(board: CalibrationBoard, settings: Settings) -> None:
    path = paths.CALIBRATION_DIR / f"tablero_{settings.board}.png"
    cv2.imwrite(str(path), board.render(settings.board_print_pixels_per_mm))
    dpi = settings.board_print_pixels_per_mm * 25.4
    logger.info(
        "Tablero guardado en %s. Imprimir al 100 %% a %.0f ppp y medir un cuadro con regla.", path, dpi
    )


def calibrate_lens(program: Program[CalibrateConfig], board: CalibrationBoard) -> None:
    components = program.config.components
    camera_config = program.repository.camera(components.camera)
    store = CalibrationStore(paths.CALIBRATION_DIR, camera_config.camera_id)
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
        console = OperatorConsole(f"GT1 - {PROGRAM} - lente")
        lens = LensCalibration(
            LiveBoardView(camera, board, console), console, store, program.config.settings.min_lens_views
        )
        console.bind_action("space", "Agregar la vista actual", lens.add_view)
        console.bind_action("c", "Calcular y guardar", lens.compute)
        console.bind_job("v", "Imagen en vivo", lens.live_job())
        console.start_job("en vivo", lens.live_job())
        console.run()


def calibrate_camera_to_robot(
    program: Program[CalibrateConfig], board: CalibrationBoard, methods: tuple[CalibrationMethod, ...]
) -> None:
    components = program.config.components
    repository = program.repository
    camera_config = repository.camera(components.camera)
    store = CalibrationStore(paths.CALIBRATION_DIR, camera_config.camera_id)
    intrinsics = store.load_intrinsics()
    with build_camera(camera_config) as camera, build_robot(repository.robot(components.robot)) as robot:
        pointer = build_gripper(components.gripper, repository.gripper(components.gripper), robot)
        program.confirm_station(
            StationHardware(
                camera=camera,
                camera_name=components.camera,
                robot=robot,
                robot_name=components.robot,
                gripper=pointer,
                tray_name=None,
            )
        )
        console = OperatorConsole(f"GT1 - {PROGRAM} - camara-robot")
        session = RobotCalibration(
            view=LiveBoardView(camera, board, console),
            robot=robot,
            board=board,
            intrinsics=intrinsics,
            console=console,
            store=store,
            methods=methods,
        )
        console.bind_action("b", "Fijar el tablero", session.fix_board)
        console.bind_job("f", "Freedrive on/off", session.toggle_freedrive_job)
        console.bind_job("t", "Registrar el toque", session.touch_job)
        console.bind_action("u", "Deshacer el ultimo toque", session.undo_touch)
        console.bind_job("c", "Calcular y guardar", session.compute_job)
        console.bind_job("v", "Imagen en vivo", session.live_job())
        console.start_job("en vivo", session.live_job())
        console.run()


def main() -> None:
    run_program("tools", PROGRAM, CalibrateConfig, calibrate)


if __name__ == "__main__":
    main()

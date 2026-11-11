"""freedrive: mover el robot a mano viendo la cámara.

Imagen en vivo con la punta del TCP marcada según la calibración de la cámara
(círculo amarillo) y, unida por una línea, el punto de la mesa justo debajo
(círculo gris). Sirve para sacar el robot de la vista, llevarlo a una pose y
leer su posición, enseñar las poses con nombre de ``robot_poses.toml`` (como
``home``, desde la que capturan las pruebas), leer valores de ``cell.toml`` ('t'
registra la pose del TCP y las articulaciones), y revisar a ojo la
calibración: la marca amarilla debe caer sobre la punta de la herramienta.

El robot debe estar en modo remoto: el freedrive se activa por RTDE. Al
detener la imagen en vivo ('x', 'q', cerrar la ventana o un error) el
freedrive se apaga.

Teclas:
    f   activar / desactivar el freedrive
    t   registrar la pose actual (TCP y articulaciones) en el registro de la corrida
    1…  guardar las articulaciones actuales en robot_poses.toml con el nombre de
        ``poses_to_teach`` (1 la primera, 2 la segunda…); reemplaza la anterior,
        que queda en el registro de la corrida
    v   volver a la imagen en vivo (después de detenerla con 'x')
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from datetime import datetime

import cv2
import numpy as np
from pydantic import Field

from gt1.calibration import PixelProjector
from gt1.config import ConfigRepository, save_robot_pose, taught_pose
from gt1.config.schemas import StrictModel
from gt1.domain import Pose
from gt1.domain.frame import Image
from gt1.hardware.camera import Camera
from gt1.hardware.robot import Robot
from gt1.runtime.console import OperatorConsole
from gt1.runtime.factories import build_camera, build_gripper, build_projector, build_robot
from gt1.runtime.overlay import to_color
from gt1.runtime.program import Program, run_program
from gt1.runtime.run_log import JsonValue, RunLog
from gt1.runtime.runner import Job
from gt1.runtime.station import StationHardware

PROGRAM = "freedrive"
logger = logging.getLogger(__name__)

TIP_COLOR = (0, 220, 255)
TABLE_COLOR = (160, 160, 160)

# Lo que pide una tecla y hace el trabajo en vivo con la pose del TCP y la imagen de esa vuelta.
PoseRequest = Callable[[Pose, Image], None]


class Components(StrictModel):
    camera: str
    robot: str
    gripper: str


class Settings(StrictModel):
    poses_to_teach: tuple[str, ...] = Field(
        min_length=1,
        max_length=9,
        description="Nombres de las poses de robot_poses.toml que se enseñan aquí: la tecla 1 guarda la "
        "primera, la 2 la segunda...",
    )


class FreedriveConfig(StrictModel):
    components: Components
    settings: Settings


class FreedriveView:
    """Imagen en vivo con la punta del TCP; el freedrive lo cambia el trabajo en vivo, no la tecla.

    Las teclas solo piden el cambio o la lectura: así el robot se usa siempre desde el hilo de trabajo.
    """

    def __init__(
        self,
        *,
        camera: Camera,
        robot: Robot,
        projector: PixelProjector,
        table_z_mm: float,
        repository: ConfigRepository,
        console: OperatorConsole,
        run_log: RunLog,
    ) -> None:
        self._camera = camera
        self._robot = robot
        self._projector = projector
        self._table_z_mm = table_z_mm
        self._repository = repository
        self._console = console
        self._run_log = run_log
        self._freedrive_requested = threading.Event()
        self._freedrive_on = False
        self._pose_requests = queue.SimpleQueue[PoseRequest]()
        self._last_recorded = ""

    def toggle_freedrive(self) -> None:
        if self._freedrive_requested.is_set():
            self._freedrive_requested.clear()
        else:
            self._freedrive_requested.set()

    def request_pose_record(self) -> None:
        """Pide registrar la pose; la lee el trabajo en vivo, que es el que usa el robot."""
        self._pose_requests.put(self._record_pose)

    def pose_save_action(self, name: str) -> Callable[[], None]:
        """Acción de la tecla que guarda la pose actual como ``name`` en robot_poses.toml."""

        def request(tcp: Pose, canvas: Image) -> None:
            self._save_pose(name, tcp, canvas)

        return lambda: self._pose_requests.put(request)

    def live_job(self) -> Job:
        def run(stop: threading.Event) -> None:
            try:
                while not stop.is_set():
                    self._apply_requested_freedrive()
                    frame = self._camera.capture()
                    tcp = self._robot.tcp_pose()
                    canvas = self._draw_tip(frame.image, tcp)
                    self._serve_pose_requests(tcp, canvas)
                    self._console.show(canvas)
                    self._console.set_status(self._status(tcp))
            finally:
                self._release_freedrive()

        return run

    def _serve_pose_requests(self, tcp: Pose, canvas: Image) -> None:
        while not self._pose_requests.empty():
            self._pose_requests.get_nowait()(tcp, canvas)

    def _save_pose(self, name: str, tcp: Pose, canvas: Image) -> None:
        pose = taught_pose(self._robot.joints_deg(), tcp.position_mm.tolist(), datetime.now())
        replaced = save_robot_pose(self._repository, name, pose)
        attempt = self._run_log.record(
            {
                "saved_pose": name,
                "joints_deg": list(pose.joints_deg),
                "tcp_mm": list(pose.tcp_mm),
                "replaced_joints_deg": list(replaced.joints_deg) if replaced is not None else None,
            },
            {"view": canvas},
        )
        logger.info("Pose '%s' guardada en robot_poses.toml (#%d): %s", name, attempt, pose.joints_deg)
        self._last_recorded = f" | pose '{name}' guardada en robot_poses.toml"

    def _record_pose(self, tcp: Pose, canvas: Image) -> None:
        joints_deg: list[JsonValue] = [round(value, 2) for value in self._robot.joints_deg()]
        attempt = self._run_log.record(
            {
                "tcp_mm": [round(float(value), 1) for value in tcp.position_mm],
                "tcp_rotation_vector_deg": [round(float(value), 2) for value in tcp.rotation_vector_deg()],
                "joints_deg": joints_deg,
            },
            {"view": canvas},
        )
        logger.info("Pose #%d: TCP %s | articulaciones %s", attempt, tcp.describe(), joints_deg)
        self._last_recorded = f" | pose #{attempt} registrada"

    def _apply_requested_freedrive(self) -> None:
        wanted = self._freedrive_requested.is_set()
        if wanted != self._freedrive_on:
            self._robot.set_freedrive(wanted)
            self._freedrive_on = wanted

    def _release_freedrive(self) -> None:
        """Estado seguro al detener la imagen en vivo: el robot vuelve a sostenerse solo."""
        self._freedrive_requested.clear()
        if self._freedrive_on:
            self._robot.set_freedrive(False)
            self._freedrive_on = False

    def _draw_tip(self, image: Image, tcp: Pose) -> Image:
        canvas = to_color(image)
        tip_mm = tcp.position_mm
        below_on_table_mm = tip_mm.copy()
        below_on_table_mm[2] = self._table_z_mm
        tip_px, table_px = self._projector.base_to_pixels(np.stack([tip_mm, below_on_table_mm]))
        tip, table = (round(tip_px[0]), round(tip_px[1])), (round(table_px[0]), round(table_px[1]))
        cv2.line(canvas, tip, table, TABLE_COLOR, 1, cv2.LINE_AA)
        cv2.circle(canvas, table, 6, TABLE_COLOR, 2, cv2.LINE_AA)
        cv2.circle(canvas, tip, 10, TIP_COLOR, 2, cv2.LINE_AA)
        return canvas

    def _status(self, tcp: Pose) -> str:
        x, y, z = tcp.position_mm
        freedrive = "freedrive ACTIVO ('f' apaga)" if self._freedrive_on else "freedrive apagado ('f')"
        return f"TCP X={x:.1f} Y={y:.1f} Z={z:.1f} mm | {freedrive}{self._last_recorded}"


def freedrive(program: Program[FreedriveConfig]) -> None:
    components, settings = program.config.components, program.config.settings
    repository = program.repository
    camera_config = repository.camera(components.camera)
    projector = build_projector(camera_config)
    with build_camera(camera_config) as camera, build_robot(repository.robot(components.robot)) as robot:
        gripper = build_gripper(components.gripper, repository.gripper(components.gripper), robot)
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
        console = OperatorConsole(f"GT1 - {PROGRAM}")
        view = FreedriveView(
            camera=camera,
            robot=robot,
            projector=projector,
            table_z_mm=repository.cell().table_z_mm,
            repository=repository,
            console=console,
            run_log=program.run_log,
        )
        console.bind_action("f", "Freedrive on/off", view.toggle_freedrive)
        console.bind_action("t", "Registrar la pose (TCP y articulaciones)", view.request_pose_record)
        for key, name in enumerate(settings.poses_to_teach, start=1):
            console.bind_action(str(key), f"Guardar la pose como '{name}'", view.pose_save_action(name))
        console.bind_job("v", "Imagen en vivo", view.live_job())
        console.start_job("en vivo", view.live_job())
        console.run()


def main() -> None:
    run_program("tools", PROGRAM, FreedriveConfig, freedrive)


if __name__ == "__main__":
    main()

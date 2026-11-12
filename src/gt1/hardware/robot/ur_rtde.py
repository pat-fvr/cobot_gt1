"""UR10e (u otro e-Series) controlado por RTDE con ur_rtde.

Es el único módulo que habla con el robot real. ur_rtde trabaja en metros y
radianes; la conversión a mm y grados se hace solo aquí.

Los movimientos no consultan antes ``isPoseWithinSafetyLimits`` ni
``isJointsWithinSafetyLimits``: en el banco (2026-11-10) rechazaban cargas sin
mover el robot. Los límites los sigue aplicando la configuración de seguridad
del controlador: si un movimiento los pasa, el robot se detiene (parada de
protección) y ``_wait_motion`` lo informa.

Conexiones RTDE:
    receive   al conectar. Solo lectura (también de las entradas digitales): funciona también con la
              consola en modo local.
    control   con el primer movimiento, cambio de herramienta o freedrive. Requiere el robot en
              modo remoto y toma el control: la consola no puede mover el robot hasta desconectar.
    io        con el primer cambio de una salida digital.
"""

# pyright: basic
# ur_rtde es una extensión compilada sin tipos: este archivo es el único que la toca.

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from gt1.config.schemas import UrRtdeRobotConfig
from gt1.domain import DigitalInput, DigitalOutput, InputPort, MoveLinear, OutputPort, Pose, ToolSettings
from gt1.geometry import rotation_from_vector, vector_from_rotation

from .base import Joints, MotionResult, Robot

logger = logging.getLogger(__name__)

# Primer bit de cada grupo de salidas en getDigitalOutState y de entradas en getDigitalInState.
_OUTPUT_FIRST_BIT = {OutputPort.STANDARD: 0, OutputPort.CONFIGURABLE: 8, OutputPort.TOOL: 16}
_INPUT_FIRST_BIT = {InputPort.STANDARD: 0, InputPort.CONFIGURABLE: 8, InputPort.TOOL: 16}
_OUTPUT_SETTER = {
    OutputPort.STANDARD: "setStandardDigitalOut",
    OutputPort.CONFIGURABLE: "setConfigurableDigitalOut",
    OutputPort.TOOL: "setToolDigitalOut",
}


class UrRtdeRobot(Robot):
    def __init__(self, config: UrRtdeRobotConfig) -> None:
        self._config = config
        self._receive: Any = None
        self._control: Any = None
        self._io: Any = None
        self._last_motion_was_linear = True
        self._freedrive = False
        # Las E/S se usan también desde el hilo del tope (TrayStopper): una a la vez.
        self._io_lock = threading.Lock()

    @property
    def is_simulated(self) -> bool:
        return False

    def connect(self) -> None:
        if self._receive is not None:
            return
        import rtde_receive  # noqa: PLC0415

        try:
            self._receive = rtde_receive.RTDEReceiveInterface(self._config.ip)
        except RuntimeError as error:
            raise RuntimeError(f"No se pudo conectar por RTDE al robot {self._config.ip}: {error}") from error
        logger.info("Robot %s conectado (lectura).", self._config.ip)

    def disconnect(self) -> None:
        if self._control is not None:
            if self._freedrive:
                self._control.endTeachMode()
                self._freedrive = False
            self._stop()
            self._control.stopScript()
            self._control.disconnect()
            self._control = None
            logger.info("Control del robot %s liberado.", self._config.ip)
        if self._io is not None:
            self._io.disconnect()
            self._io = None
        if self._receive is not None:
            self._receive.disconnect()
            self._receive = None

    def tcp_pose(self) -> Pose:
        return _pose_from_ur(self._require_receive().getActualTCPPose())

    def joints_deg(self) -> Joints:
        q = np.degrees(np.asarray(self._require_receive().getActualQ(), dtype=np.float64))
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]), float(q[5]))

    def move_linear(self, target: Pose, speed_mm_s: float, stop: threading.Event) -> MotionResult:
        control = self._require_control()
        pose = _pose_to_ur(target)
        self._last_motion_was_linear = True
        if not control.moveL(pose, speed_mm_s / 1000.0, self._config.linear_accel_mm_s2 / 1000.0, True):
            raise RuntimeError(f"El robot rechazó el movimiento a {target.describe()}.")
        return self._wait_motion(stop)

    def move_linear_path(self, moves: Sequence[MoveLinear], stop: threading.Event) -> MotionResult:
        """Una sola trayectoria (moveL con una lista de puntos): el robot pasa en curva por cada punto con
        radio sin detenerse. El controlador exige radio cero en el último punto."""
        control = self._require_control()
        accel_m_s2 = self._config.linear_accel_mm_s2 / 1000.0
        path: list[list[float]] = []
        for index, move in enumerate(moves):
            pose = _pose_to_ur(move.target)
            is_last = index == len(moves) - 1
            blend_m = 0.0 if is_last else move.blend_radius_mm / 1000.0
            path.append([*pose, move.speed_mm_s / 1000.0, accel_m_s2, blend_m])
        self._last_motion_was_linear = True
        if not control.moveL(path, True):
            targets = ", ".join(move.target.describe() for move in moves)
            raise RuntimeError(f"El robot rechazó la trayectoria por {targets}.")
        return self._wait_motion(stop)

    def move_joints(self, joints_deg: Joints, speed_deg_s: float, stop: threading.Event) -> MotionResult:
        control = self._require_control()
        q = np.radians(np.asarray(joints_deg, dtype=np.float64)).tolist()
        self._last_motion_was_linear = False
        accel = np.radians(self._config.joint_accel_deg_s2)
        if not control.moveJ(q, np.radians(speed_deg_s), accel, True):
            raise RuntimeError(f"El robot rechazó el movimiento articular a {joints_deg}.")
        return self._wait_motion(stop)

    def set_digital_output(self, output: DigitalOutput, value: bool) -> None:
        setter = getattr(self._require_io(), _OUTPUT_SETTER[output.port])
        with self._io_lock:
            accepted = setter(output.number, value)
        if not accepted:
            raise RuntimeError(f"El robot no aceptó cambiar la salida {output.describe()}.")
        logger.info("%s -> %s", output.describe(), "1" if value else "0")

    def digital_output(self, output: DigitalOutput) -> bool:
        bit = _OUTPUT_FIRST_BIT[output.port] + output.number
        with self._io_lock:
            return bool(self._require_receive().getDigitalOutState(bit))

    def digital_input(self, sensor: DigitalInput) -> bool:
        bit = _INPUT_FIRST_BIT[sensor.port] + sensor.number
        with self._io_lock:
            return bool(self._require_receive().getDigitalInState(bit))

    def apply_tool(self, tool: ToolSettings) -> None:
        control = self._require_control()
        if not control.setTcp(_pose_to_ur(tool.tcp)):
            raise RuntimeError("El robot no aceptó el TCP de la herramienta.")
        center_of_gravity_m = (tool.center_of_gravity_mm / 1000.0).tolist()
        if not control.setPayload(tool.payload_kg, center_of_gravity_m):
            raise RuntimeError("El robot no aceptó la carga (payload) de la herramienta.")
        self._check_tool(tool)
        logger.info("Herramienta cargada en el robot: TCP %s, %.2f kg.", tool.tcp.describe(), tool.payload_kg)

    def _check_tool(self, requested: ToolSettings) -> None:
        """Lee de vuelta el TCP y la carga: el controlador puede aceptar el comando y no aplicarlo."""
        loaded = self.tool()
        tcp_ok = loaded.tcp.is_close(
            requested.tcp, self._config.tool_position_tol_mm, self._config.tool_rotation_tol_deg
        )
        payload_ok = abs(loaded.payload_kg - requested.payload_kg) <= self._config.payload_tol_kg
        if not (tcp_ok and payload_ok):
            raise RuntimeError(
                f"El robot no quedó con la herramienta pedida. Pedido: TCP {requested.tcp.describe()}, "
                f"{requested.payload_kg:.2f} kg. "
                f"Robot: TCP {loaded.tcp.describe()}, {loaded.payload_kg:.2f} kg."
            )

    def tool(self) -> ToolSettings:
        receive = self._require_receive()
        return ToolSettings(
            tcp=_pose_from_ur(self._require_control().getTCPOffset()),
            payload_kg=float(receive.getPayload()),
            center_of_gravity_mm=np.asarray(receive.getPayloadCog(), dtype=np.float64) * 1000.0,
        )

    def set_freedrive(self, enabled: bool) -> None:
        control = self._require_control()
        if enabled and not self._freedrive:
            if not control.teachMode():
                raise RuntimeError("El robot no aceptó entrar en freedrive.")
        elif not enabled and self._freedrive and not control.endTeachMode():
            raise RuntimeError("El robot no aceptó salir de freedrive.")
        self._freedrive = enabled
        logger.info("Freedrive %s.", "activado" if enabled else "desactivado")

    def _wait_motion(self, stop: threading.Event) -> MotionResult:
        control = self._require_control()
        while control.getAsyncOperationProgress() >= 0:
            if stop.is_set():
                self._stop()
                return MotionResult.STOPPED
            halt = self._halt_reason()
            if halt:
                raise RuntimeError(
                    f"El robot se detuvo antes de llegar: {halt}. Revisar el robot y su entorno, liberar "
                    "la parada en la consola y volver a iniciar el programa."
                )
            time.sleep(self._config.poll_period_s)
        return MotionResult.COMPLETED

    def _halt_reason(self) -> str:
        """Por qué el robot dejó de moverse sin que se lo pidieran, o "" si sigue en marcha.

        Con una parada de protección o de emergencia el controlador corta el programa de ur_rtde y el
        progreso del movimiento queda congelado: sin esta revisión la espera no terminaría nunca.
        """
        receive = self._require_receive()
        if receive.isProtectiveStopped():
            return "parada de protección (choque o límite de seguridad)"
        if receive.isEmergencyStopped():
            return "parada de emergencia"
        if not self._require_control().isProgramRunning():
            return "el programa de control del robot se detuvo (¿se paró desde la consola?)"
        return ""

    def _stop(self) -> None:
        if self._last_motion_was_linear:
            self._control.stopL(self._config.stop_decel_mm_s2 / 1000.0)
        else:
            self._control.stopJ(np.radians(self._config.stop_decel_deg_s2))

    def _require_receive(self) -> Any:
        if self._receive is None:
            raise RuntimeError("El robot no está conectado.")
        return self._receive

    def _require_control(self) -> Any:
        if self._control is None:
            self._require_receive()
            import rtde_control  # noqa: PLC0415

            try:
                self._control = rtde_control.RTDEControlInterface(self._config.ip)
            except RuntimeError as error:
                raise RuntimeError(
                    f"No se pudo tomar el control del robot ({error}). ¿Está la consola en modo remoto?"
                ) from error
            logger.info("Control del robot %s tomado.", self._config.ip)
        return self._control

    def _require_io(self) -> Any:
        if self._io is None:
            self._require_receive()
            import rtde_io  # noqa: PLC0415

            self._io = rtde_io.RTDEIOInterface(self._config.ip)
        return self._io


def _pose_from_ur(pose: list[float]) -> Pose:
    """[x, y, z (m), rx, ry, rz (rad)] de ur_rtde -> Pose en mm."""
    values = np.asarray(pose, dtype=np.float64)
    return Pose.from_position(values[:3] * 1000.0, rotation_from_vector(values[3:]))


def _pose_to_ur(pose: Pose) -> list[float]:
    return [*(pose.position_mm / 1000.0).tolist(), *vector_from_rotation(pose.rotation).tolist()]

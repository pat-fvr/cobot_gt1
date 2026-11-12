"""Tope de la bandeja fija: cerrado al iniciar y, después de soltar, abierto en segundo plano hasta que sale
la bandeja cargada, mientras el robot va por la pieza siguiente.

El hilo del tope es la única excepción a "un hilo a la vez" (AGENTS.md, sección 5): solo usa las E/S del
robot (la salida del tope y la entrada del sensor), nunca lo mueve, y el driver serializa las E/S. La parada
del operador ('x') no lo corta: cerrar el tope cuando salió la bandeja deja la estación en su estado seguro.
El plan de soltar espera que el tope esté cerrado antes de mirar el sensor (``tray_station_drop_plan``).
"""

from __future__ import annotations

import logging
import threading
import time
from types import TracebackType

from gt1.config.schemas import TrayStationSettings
from gt1.hardware.robot import Robot
from gt1.planning import STOPPER_CLOSED, TRAY_PRESENT

logger = logging.getLogger(__name__)

# Hasta cuánto esperar a leer de vuelta la salida del tope al iniciar: el estado de las salidas llega por
# RTDE unos milisegundos después de la orden.
READBACK_TIMEOUT_S = 0.5
# Cada cuánto se lee el sensor o la salida mientras se espera: muy por debajo de lo que tarda una bandeja.
POLL_S = 0.02


class TrayStopper:
    def __init__(self, robot: Robot, station: TrayStationSettings) -> None:
        self._robot = robot
        self._station = station
        self._stopper = station.stopper.digital_output()
        self._sensor = station.tray_sensor.digital_input()
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> TrayStopper:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close_and_verify(self) -> None:
        """Cierra el tope y verifica que su salida quedó en 1: abierto, la bandeja no se detiene bajo la
        pieza."""
        self._robot.set_digital_output(self._stopper, STOPPER_CLOSED)
        deadline_s = time.monotonic() + READBACK_TIMEOUT_S
        while self._robot.digital_output(self._stopper) != STOPPER_CLOSED:
            if time.monotonic() > deadline_s:
                raise RuntimeError(
                    f"El tope ({self._stopper.describe()}) no quedó cerrado al iniciar: el robot no informa "
                    f"la salida en 1 después de {READBACK_TIMEOUT_S} s. Revisar que el tope esté en esa "
                    "salida (stopper en [settings.tray_station]) y su cableado."
                )
            time.sleep(POLL_S)
        logger.info("Tope (%s) cerrado al iniciar.", self._stopper.describe())

    def let_loaded_tray_out(self) -> None:
        """Abre el tope (ya, en este hilo) y lo cierra en segundo plano ``stopper_close_delay_s`` después de
        que el sensor deja de ver la bandeja cargada."""
        self._join()
        self._robot.set_digital_output(self._stopper, not STOPPER_CLOSED)
        self._thread = threading.Thread(target=self._close_once_tray_left, name="tope", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Termina el hilo del tope; si la bandeja no había salido, el tope queda abierto."""
        self._halt.set()
        self._join()

    def _join(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _close_once_tray_left(self) -> None:
        try:
            while self._robot.digital_input(self._sensor) == TRAY_PRESENT:
                if self._halt.wait(POLL_S):
                    return
            if self._halt.wait(self._station.stopper_close_delay_s):
                return
            self._robot.set_digital_output(self._stopper, STOPPER_CLOSED)
        except RuntimeError as error:
            # El plan de soltar espera el tope cerrado: el robot se queda esperándolo y 'x' lo detiene.
            logger.error(
                "No se pudo cerrar el tope (%s) después de que salió la bandeja: %s. El robot va a esperar "
                "el tope cerrado para soltar: revisar el tope y reiniciar el programa.",
                self._stopper.describe(),
                error,
            )

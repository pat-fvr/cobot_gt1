"""Hilo de trabajo: corre un trabajo a la vez (capturar, mover el robot…) sin bloquear la consola.

El trabajo es una función que recibe la señal de parada (``threading.Event``)
y la revisa entre pasos; los movimientos del robot también la reciben y frenan
solos. Así el hardware solo se toca desde este hilo y la tecla de parada
responde siempre.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

Job = Callable[[threading.Event], None]


class JobRunner:
    def __init__(self, on_status: Callable[[str], None]) -> None:
        self._on_status = on_status
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, name: str, job: Job) -> bool:
        """Inicia ``job`` si no hay otro en curso; devuelve si se inició."""
        if self.busy:
            self._on_status(f"Ocupado: espere a que termine o detenga con 'x' antes de '{name}'.")
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(name, job), name=f"trabajo:{name}", daemon=True
        )
        self._thread.start()
        return True

    def request_stop(self) -> None:
        """Pide al trabajo en curso que se detenga (no espera)."""
        if self.busy:
            self._stop.set()
            self._on_status("Deteniendo…")

    def wait(self, timeout_s: float) -> bool:
        """Espera a que termine el trabajo en curso; devuelve si terminó dentro del plazo."""
        if self._thread is not None:
            self._thread.join(timeout_s)
        return not self.busy

    def _run(self, name: str, job: Job) -> None:
        try:
            job(self._stop)
        except Exception as error:
            logger.exception("Falló '%s'.", name)
            self._on_status(f"Error en '{name}': {error}")
        else:
            if self._stop.is_set():
                self._on_status(f"'{name}' detenido.")

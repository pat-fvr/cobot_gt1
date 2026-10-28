"""Consola de operación: ventana con la imagen, barra de estado, ayuda y teclado.

Cada programa registra sus teclas; las reservadas funcionan en todos:

    x   detener el trabajo en curso (siempre responde)
    h   mostrar u ocultar la ayuda
    q   salir (detiene el trabajo en curso y espera a que termine)

Dos tipos de tecla:

    bind_job      inicia un trabajo en el hilo de trabajo (capturar, mover el robot…):
                  la ventana sigue respondiendo mientras corre.
    bind_action   llama a una función rápida en el hilo de la ventana (cambiar un
                  modo, guardar la última imagen).

La consola no tiene lógica del programa: muestra lo que le mandan (``show``,
``set_status``, desde cualquier hilo) y ejecuta lo registrado.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from gt1.domain.frame import Image

from .overlay import ascii_text, to_color
from .runner import Job, JobRunner

logger = logging.getLogger(__name__)

RESERVED_KEYS = {"x": "Detener", "h": "Ayuda", "q": "Salir"}
KEY_CODES = {"space": 32, "enter": 13}
WINDOW_POLL_MS = 30
DEFAULT_CANVAS_SIZE_PX = (960, 540)
MAX_WINDOW_WIDTH_PX = 1400
JOB_STOP_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class KeyBinding:
    key: str
    description: str
    action: Callable[[], None]


class OperatorConsole:
    def __init__(self, title: str) -> None:
        self._title = title
        self._runner = JobRunner(self.set_status)
        self._bindings: dict[int, KeyBinding] = {}
        self._lock = threading.Lock()
        self._image: Image | None = None
        self._status = "Listo. 'h' muestra la ayuda."
        self._show_help = True

    def bind_job(self, key: str, description: str, job: Job) -> None:
        """Tecla que inicia ``job`` en el hilo de trabajo (si no hay otro en curso)."""

        def start() -> None:
            self.start_job(description, job)

        self.bind_action(key, description, start)

    def bind_action(self, key: str, description: str, action: Callable[[], None]) -> None:
        """Tecla que llama a ``action`` en el hilo de la ventana: debe volver rápido.

        ``key`` es un carácter ('c', '1') o un nombre ('space', 'enter').
        """
        if key in RESERVED_KEYS:
            raise ValueError(f"La tecla '{key}' está reservada para '{RESERVED_KEYS[key]}'.")
        code = KEY_CODES.get(key, ord(key) if len(key) == 1 else -1)
        if code < 0:
            raise ValueError(f"Tecla desconocida: {key!r}.")
        if code in self._bindings:
            raise ValueError(f"La tecla '{key}' ya está asignada a '{self._bindings[code].description}'.")
        self._bindings[code] = KeyBinding(key, description, action)

    def start_job(self, name: str, job: Job) -> bool:
        """Inicia un trabajo sin tecla (p. ej. la imagen en vivo al abrir); devuelve si se inició."""
        return self._runner.submit(name, job)

    def stop_job(self, timeout_s: float) -> bool:
        """Detiene el trabajo en curso y espera; devuelve si terminó dentro del plazo."""
        self._runner.request_stop()
        return self._runner.wait(timeout_s)

    @property
    def job_running(self) -> bool:
        return self._runner.busy

    def show(self, image: Image) -> None:
        with self._lock:
            self._image = image

    def set_status(self, text: str) -> None:
        with self._lock:
            self._status = text

    def run(self) -> None:
        """Atiende la ventana y el teclado hasta 'q' o hasta cerrar la ventana."""
        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        try:
            while True:
                cv2.imshow(self._title, self._render())
                key = cv2.waitKey(WINDOW_POLL_MS) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(self._title, cv2.WND_PROP_VISIBLE) < 1:
                    return
                self._handle_key(key)
        finally:
            cv2.destroyWindow(self._title)
            if not self.stop_job(JOB_STOP_TIMEOUT_S):
                logger.error("El trabajo en curso no se detuvo en %.0f s al salir.", JOB_STOP_TIMEOUT_S)

    def _handle_key(self, key: int) -> None:
        if key == ord("x"):
            self._runner.request_stop()
        elif key == ord("h"):
            self._show_help = not self._show_help
        elif key in self._bindings:
            binding = self._bindings[key]
            try:
                binding.action()
            except Exception as error:
                # Una tecla que falla no cierra la consola: el operador ve el error y puede detener o salir.
                logger.exception("Falló la tecla '%s' (%s).", binding.key, binding.description)
                self.set_status(f"Error en '{binding.description}': {error}")

    def _render(self) -> Image:
        with self._lock:
            image, status = self._image, self._status
        if image is None:
            width, height = DEFAULT_CANVAS_SIZE_PX
            canvas = np.full((height, width, 3), 40, dtype=np.uint8)
        else:
            canvas = to_color(image)
        if canvas.shape[1] > MAX_WINDOW_WIDTH_PX:
            scale = MAX_WINDOW_WIDTH_PX / canvas.shape[1]
            resized = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            canvas = np.asarray(resized, dtype=np.uint8)
        if self._show_help:
            self._draw_help(canvas)
        return self._with_status_bar(canvas, status)

    def _help_lines(self) -> list[str]:
        lines = [f"{binding.key:>6}  {binding.description}" for binding in self._bindings.values()]
        return [*lines, *(f"{key:>6}  {text}" for key, text in RESERVED_KEYS.items())]

    def _draw_help(self, canvas: Image) -> None:
        lines = self._help_lines()
        line_height = 26
        box_height = line_height * len(lines) + 16
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (420, 8 + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, dst=canvas)
        for index, line in enumerate(lines):
            position = (18, 8 + 24 + index * line_height)
            cv2.putText(
                canvas,
                ascii_text(line),
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _with_status_bar(canvas: Image, status: str) -> Image:
        bar = np.full((36, canvas.shape[1], 3), 25, dtype=np.uint8)
        cv2.putText(
            bar, ascii_text(status), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA
        )
        return np.vstack([canvas, bar])

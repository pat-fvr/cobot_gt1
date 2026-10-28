"""Interfaz de las cámaras."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from gt1.domain import Frame


class Camera(ABC):
    """Fuente de imágenes. Se usa como contexto: ``with camera: frame = camera.capture()``."""

    @property
    @abstractmethod
    def camera_id(self) -> str:
        """Cámara física cuyos datos (calibración, referencias) se usan con estas imágenes."""

    @property
    @abstractmethod
    def is_simulated(self) -> bool:
        """Las imágenes no vienen de una cámara conectada (p. ej. de una carpeta)."""

    @abstractmethod
    def connect(self) -> None:
        """Conecta y, si la cámara informa su modelo y número de serie, verifica que sean los esperados."""

    @abstractmethod
    def disconnect(self) -> None:
        """Libera la cámara. Se puede llamar aunque no esté conectada."""

    @abstractmethod
    def capture(self) -> Frame:
        """La imagen siguiente, tomada después de la llamada."""

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.disconnect()

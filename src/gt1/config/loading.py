"""Lectura y validación de los archivos de ``config/``.

Cada componente se nombra por su archivo: ``camera = "basler_scA1390"`` en la
configuración de una prueba se refiere a ``config/cameras/basler_scA1390.toml``.
El repositorio recuerda qué archivos leyó, para guardar una copia en el
registro de cada corrida.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from .schemas import (
    CameraConfig,
    CellConfig,
    CharucoBoardConfig,
    FaceClassifierConfig,
    GripperConfig,
    PieceDetectorConfig,
    RobotConfig,
    StationConfig,
    TrayConfig,
)

ProgramKind = Literal["tests", "tools"]

_ROBOT = TypeAdapter[RobotConfig](RobotConfig)
_CAMERA = TypeAdapter[CameraConfig](CameraConfig)
_GRIPPER = TypeAdapter[GripperConfig](GripperConfig)
_TRAY = TypeAdapter(TrayConfig)
_PIECE_DETECTOR = TypeAdapter[PieceDetectorConfig](PieceDetectorConfig)
_FACE_CLASSIFIER = TypeAdapter[FaceClassifierConfig](FaceClassifierConfig)
_BOARD = TypeAdapter(CharucoBoardConfig)
_CELL = TypeAdapter(CellConfig)
_STATION = TypeAdapter(StationConfig)


class ConfigError(Exception):
    """Archivo de configuración que falta o que no cumple su esquema."""


class ConfigRepository:
    """Acceso tipado a los archivos de una carpeta ``config/``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._loaded: list[Path] = []

    @property
    def loaded_files(self) -> tuple[Path, ...]:
        """Archivos leídos hasta ahora, sin repetir, en el orden en que se leyeron."""
        return tuple(dict.fromkeys(self._loaded))

    def station_path(self) -> Path:
        return self.root / "station.toml"

    def program_file[T](self, path: Path, schema: type[T]) -> T:
        """Configuración de una prueba o herramienta (``tests/<nombre>.toml`` o ``tools/<nombre>.toml``)."""
        return self._load(path, TypeAdapter(schema))

    def robot(self, name: str) -> RobotConfig:
        return self._load(self._named_file("robots", name), _ROBOT)

    def camera(self, name: str) -> CameraConfig:
        return self._load(self._named_file("cameras", name), _CAMERA)

    def gripper(self, name: str) -> GripperConfig:
        return self._load(self._named_file("grippers", name), _GRIPPER)

    def tray(self, name: str) -> TrayConfig:
        return self._load(self._named_file("trays", name), _TRAY)

    def piece_detector(self, name: str) -> PieceDetectorConfig:
        return self._load(self._named_file("perception", name), _PIECE_DETECTOR)

    def face_classifier(self, name: str) -> FaceClassifierConfig:
        return self._load(self._named_file("perception", name), _FACE_CLASSIFIER)

    def board(self, name: str) -> CharucoBoardConfig:
        return self._load(self._named_file("boards", name), _BOARD)

    def cell(self) -> CellConfig:
        return self._load(self.root / "cell.toml", _CELL)

    def station(self) -> StationConfig:
        return self._load(self.station_path(), _STATION)

    def available(self, folder: str) -> list[str]:
        """Nombres de los archivos de una carpeta (sin la extensión)."""
        return sorted(path.stem for path in (self.root / folder).glob("*.toml"))

    def _named_file(self, folder: str, name: str) -> Path:
        path = self.root / folder / f"{name}.toml"
        if not path.is_file():
            options = ", ".join(self.available(folder)) or "(ninguno)"
            raise ConfigError(
                f"No existe {path.relative_to(self.root.parent)}. Disponibles en {folder}/: {options}."
            )
        return path

    def _load[T](self, path: Path, adapter: TypeAdapter[T]) -> T:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ConfigError(f"No existe el archivo de configuración {path}.") from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path}: el TOML no es válido: {error}") from error
        try:
            # Se valida como JSON: el modo estricto acepta listas de TOML para las tuplas.
            value = adapter.validate_json(json.dumps(data), strict=True)
        except ValidationError as error:
            raise ConfigError(f"{path}:\n{_describe(error)}") from error
        self._loaded.append(path)
        return value


def _describe(error: ValidationError) -> str:
    lines: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"])
        lines.append(f"  - {location or '(archivo)'}: {detail['msg']}")
    return "\n".join(lines)

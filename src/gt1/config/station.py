"""Comparación entre el hardware que pide un programa y el que figura montado en ``station.toml``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from .schemas import StationConfig

STATION_HEADER = (
    "# Hardware montado en el banco según la última configuración confirmada.\n"
    "# Lo actualizan los programas cuando el operador confirma un cambio; no hace falta editarlo a mano.\n"
)


@dataclass(frozen=True)
class HardwareChange:
    component: str  # robot, camera, gripper o tray
    mounted: str  # lo que dice station.toml
    requested: str  # lo que pide el programa


def station_changes(station: StationConfig, requested: Mapping[str, str]) -> list[HardwareChange]:
    """Componentes cuyo nombre pedido difiere del montado. Solo se comparan los que pide el programa."""
    mounted = station.model_dump()
    unknown = requested.keys() - mounted.keys()
    if unknown:
        raise ValueError(f"Componentes que no son hardware de la estación: {', '.join(sorted(unknown))}.")
    return [
        HardwareChange(component, mounted[component], name)
        for component, name in requested.items()
        if mounted[component] != name
    ]


def updated_station(station: StationConfig, changes: list[HardwareChange]) -> StationConfig:
    return station.model_copy(update={change.component: change.requested for change in changes})


def write_station(path: Path, station: StationConfig) -> None:
    path.write_text(STATION_HEADER + tomli_w.dumps(station.model_dump()), encoding="utf-8")

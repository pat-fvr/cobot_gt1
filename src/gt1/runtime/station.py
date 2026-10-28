"""Confirmación del hardware al iniciar un programa.

1. Se compara el hardware que pide el programa con ``station.toml`` y se
   muestran las diferencias.
2. Las verificaciones automáticas ya ocurrieron al conectar: la cámara revisa
   su modelo y número de serie, y el robot responde en su IP.
3. Si hay diferencias, el operador confirma lo que no se puede leer (qué
   gripper y qué bandeja están montados). Si no confirma, el programa no sigue.
4. Se carga en el robot el TCP y la carga del gripper, y el robot los lee de
   vuelta para verificarlos.
5. Se actualiza ``station.toml``.

Lo simulado (robot simulado, cámara de carpeta) no se compara ni se escribe:
``station.toml`` describe el hardware real del banco.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from gt1.config import ConfigRepository, station_changes, updated_station, write_station
from gt1.hardware.camera import Camera
from gt1.hardware.gripper import Gripper
from gt1.hardware.robot import Robot

logger = logging.getLogger(__name__)


class StationNotConfirmedError(Exception):
    """El operador no confirmó el hardware montado."""


@dataclass(frozen=True)
class StationHardware:
    """Hardware que usa un programa, ya conectado. Lo que el programa no usa queda en None."""

    camera: Camera | None
    camera_name: str | None
    robot: Robot | None
    robot_name: str | None
    gripper: Gripper | None
    tray_name: str | None


def confirm_station(
    repository: ConfigRepository, hardware: StationHardware, ask: Callable[[str], bool]
) -> None:
    requested = _real_hardware(hardware)
    station = repository.station()
    changes = station_changes(station, requested)
    if changes:
        lines = [f"  {change.component}: {change.mounted} -> {change.requested}" for change in changes]
        logger.warning("La configuración difiere del hardware montado (station.toml):\n%s", "\n".join(lines))
        question = (
            "Diferencias con el hardware montado:\n" + "\n".join(lines) + "\n"
            "Revise físicamente el gripper y la bandeja montados. ¿Es correcto el hardware para esta prueba?"
        )
        if not ask(question):
            raise StationNotConfirmedError("El operador no confirmó el hardware: no se inicia el programa.")
    if hardware.robot is not None and hardware.gripper is not None:
        hardware.robot.apply_tool(hardware.gripper.tool_settings)
    if changes:
        write_station(repository.station_path(), updated_station(station, changes))
        logger.info("station.toml actualizado.")


def _real_hardware(hardware: StationHardware) -> dict[str, str]:
    """Nombres del hardware real que pide el programa (sin lo simulado)."""
    requested: dict[str, str] = {}
    if hardware.camera is not None and hardware.camera_name is not None and not hardware.camera.is_simulated:
        requested["camera"] = hardware.camera_name
    robot_is_real = hardware.robot is not None and not hardware.robot.is_simulated
    if robot_is_real and hardware.robot_name is not None:
        requested["robot"] = hardware.robot_name
        if hardware.gripper is not None:
            requested["gripper"] = hardware.gripper.name
    if hardware.tray_name is not None:
        requested["tray"] = hardware.tray_name
    return requested

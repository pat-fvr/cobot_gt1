"""Soltar en la bandeja fija con tope: esperar que llegue, soltar la pieza y dejarla pasar.

El robot ya está quieto sobre la bandeja con la pieza tomada. El tope pasa
cerrado (detiene la bandeja bajo la pieza); se abre solo después de soltar,
durante ``stopper_open_s``, para que salga la bandeja cargada, y se cierra
para la siguiente. No se espera a que el sensor deje de verla: el tiempo
abierto es el de la configuración.
"""

from __future__ import annotations

from gt1.config.schemas import TrayStationSettings
from gt1.domain import GripperCommand, Plan, SetOutput, Wait, WaitForInput

STOPPER_CLOSED = True
TRAY_PRESENT = True


def tray_station_drop_plan(station: TrayStationSettings) -> Plan:
    stopper = station.stopper.digital_output()
    sensor = station.tray_sensor.digital_input()
    return Plan(
        name="soltar en la bandeja",
        steps=(
            # Ya debería estar cerrado: al iniciar el programa no se sabe cómo quedó.
            SetOutput(stopper, STOPPER_CLOSED, f"tope cerrado ({stopper.describe()})"),
            WaitForInput(sensor, TRAY_PRESENT, f"esperar la bandeja ({sensor.describe()})"),
            GripperCommand("release", "soltar la pieza en la bandeja"),
            SetOutput(stopper, not STOPPER_CLOSED, f"abrir el tope ({stopper.describe()})"),
            Wait(station.stopper_open_s, "dejar pasar la bandeja cargada"),
            SetOutput(stopper, STOPPER_CLOSED, f"cerrar el tope ({stopper.describe()}) para la siguiente"),
        ),
    )

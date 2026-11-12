"""Soltar en la bandeja fija con tope: esperar que llegue y soltar la pieza.

El robot ya está quieto sobre la bandeja con la pieza tomada. El tope lo
maneja ``TrayStopper`` (``gt1.runtime.tray_stopper``): después de soltar lo
abre para que salga la bandeja cargada y lo cierra en segundo plano, mientras
el robot va por la pieza siguiente. Por eso este plan primero espera que el
tope esté cerrado: si no, el sensor podría estar viendo todavía la bandeja
anterior, ya cargada.
"""

from __future__ import annotations

from gt1.config.schemas import TrayStationSettings
from gt1.domain import GripperCommand, Plan, WaitForInput, WaitForOutput

STOPPER_CLOSED = True
TRAY_PRESENT = True


def tray_station_drop_plan(station: TrayStationSettings) -> Plan:
    stopper = station.stopper.digital_output()
    sensor = station.tray_sensor.digital_input()
    return Plan(
        name="soltar en la bandeja",
        steps=(
            WaitForOutput(
                stopper, STOPPER_CLOSED, f"esperar que salga la bandeja anterior ({stopper.describe()})"
            ),
            WaitForInput(sensor, TRAY_PRESENT, f"esperar la bandeja ({sensor.describe()})"),
            GripperCommand("release", "soltar la pieza en la bandeja"),
        ),
    )

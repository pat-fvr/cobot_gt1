"""TrayStopper: el tope de la bandeja fija, cerrado al iniciar y abierto en segundo plano para que salga la
bandeja cargada."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from gt1.config.schemas import InputChannel, OutputChannel, SimRobotConfig, TrayStationSettings
from gt1.domain import DigitalOutput, InputPort, OutputPort
from gt1.hardware.robot.sim import SimRobot
from gt1.runtime.tray_stopper import TrayStopper

STATION = TrayStationSettings(
    stopper=OutputChannel(port=OutputPort.STANDARD, number=2),
    tray_sensor=InputChannel(port=InputPort.STANDARD, number=1),
    stopper_close_delay_s=0.05,
)
STOPPER = STATION.stopper.digital_output()
SENSOR = STATION.tray_sensor.digital_input()
SIM = SimRobotConfig(
    kind="sim", initial_tcp_mm=(0.0, 0.0, 300.0), initial_joints_deg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
)


class StuckOutputRobot(SimRobot):
    """Robot simulado cuyas salidas no cambian: acepta la orden, pero la salida sigue en 0."""

    def digital_output(self, output: DigitalOutput) -> bool:
        return False


def becomes_true(condition: Callable[[], bool], timeout_s: float) -> bool:
    deadline_s = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline_s:
            return False
        time.sleep(0.01)
    return True


def test_at_start_the_stopper_is_closed_and_read_back() -> None:
    """En el banco (2026-11-12) el programa empezó con el tope abierto."""
    with SimRobot(SIM) as robot, TrayStopper(robot, STATION) as stopper:
        stopper.close_and_verify()
        assert robot.digital_output(STOPPER)


def test_a_stopper_that_does_not_close_at_start_stops_the_program() -> None:
    with (
        StuckOutputRobot(SIM) as robot,
        TrayStopper(robot, STATION) as stopper,
        pytest.raises(RuntimeError, match="DO2"),
    ):
        stopper.close_and_verify()


def test_the_loaded_tray_goes_out_and_the_stopper_closes_after_the_sensor_stops_seeing_it() -> None:
    with SimRobot(SIM) as robot, TrayStopper(robot, STATION) as stopper:
        stopper.close_and_verify()
        robot.set_digital_input(SENSOR, True)
        stopper.let_loaded_tray_out()
        assert not robot.digital_output(STOPPER)  # abierto apenas vuelve: el robot ya puede irse
        assert not becomes_true(lambda: robot.digital_output(STOPPER), 0.2)  # la bandeja sigue ahí
        robot.set_digital_input(SENSOR, False)
        assert becomes_true(lambda: robot.digital_output(STOPPER), 2.0)


def test_closing_the_stopper_object_ends_the_wait_for_a_tray_that_never_leaves() -> None:
    with SimRobot(SIM) as robot:
        robot.set_digital_input(SENSOR, True)
        stopper = TrayStopper(robot, STATION)
        stopper.let_loaded_tray_out()
        stopper.close()  # vuelve aunque la bandeja siga en el sensor
        assert not robot.digital_output(STOPPER)  # el programa terminó con la bandeja todavía en el tope

"""Gripper, confirmación de la estación, hilo de trabajo y registro de corridas."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from gt1 import paths
from gt1.config import ConfigRepository
from gt1.config.schemas import (
    DigitalOutputGripperConfig,
    GripperActionConfig,
    OutputSetting,
    PayloadConfig,
    SimRobotConfig,
    TcpConfig,
)
from gt1.domain import DigitalOutput, OutputPort
from gt1.hardware.gripper.digital_outputs import DigitalOutputGripper
from gt1.hardware.robot.sim import SimRobot
from gt1.runtime.run_log import RunLog
from gt1.runtime.runner import JobRunner
from gt1.runtime.station import StationHardware, StationNotConfirmedError, confirm_station

from .fakes import StaticCamera

DO1 = DigitalOutput(OutputPort.STANDARD, 1)
DO2 = DigitalOutput(OutputPort.STANDARD, 2)


def two_valve_gripper(robot: SimRobot) -> DigitalOutputGripper:
    """Gripper de presión (DO1) y vacío (DO2): cada acción fija las dos salidas."""

    def action(description: str, pressure: bool) -> GripperActionConfig:
        return GripperActionConfig(
            description=description,
            outputs=(
                OutputSetting(port=OutputPort.STANDARD, number=1, value=pressure),
                OutputSetting(port=OutputPort.STANDARD, number=2, value=not pressure),
            ),
            settle_s=0.0,
        )

    config = DigitalOutputGripperConfig(
        kind="digital_outputs",
        description="presión y vacío",
        tcp=TcpConfig(position_mm=(0.0, 0.0, 150.0), rotation_vector_deg=(0.0, 0.0, 0.0)),
        payload=PayloadConfig(mass_kg=1.2, center_of_gravity_mm=(0.0, 0.0, 60.0)),
        actions={"grip": action("cerrar", True), "release": action("abrir", False)},
    )
    return DigitalOutputGripper("presion_vacio", config, robot)


@pytest.fixture
def robot() -> SimRobot:
    sim = SimRobot(
        SimRobotConfig(kind="sim", initial_tcp_mm=(0.0, 0.0, 0.0), initial_joints_deg=(0, 0, 0, 0, 0, 0))
    )
    sim.connect()
    return sim


def test_gripper_actions_set_their_outputs(robot: SimRobot) -> None:
    gripper = two_valve_gripper(robot)
    gripper.perform("grip")
    assert (robot.digital_output(DO1), robot.digital_output(DO2)) == (True, False)
    gripper.perform("release")
    assert (robot.digital_output(DO1), robot.digital_output(DO2)) == (False, True)
    with pytest.raises(ValueError, match="no tiene la acción"):
        gripper.perform("girar")


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    shutil.copy(paths.CONFIG_DIR / "station.toml", tmp_path / "station.toml")
    return tmp_path


def station_text(config_dir: Path) -> str:
    return (config_dir / "station.toml").read_text(encoding="utf-8")


def camera_hardware(name: str, simulated: bool) -> StationHardware:
    camera = StaticCamera(np.zeros((10, 10), dtype=np.uint8), simulated=simulated)
    return StationHardware(
        camera=camera, camera_name=name, robot=None, robot_name=None, gripper=None, tray_name=None
    )


def never_asked(question: str) -> bool:
    raise AssertionError(f"No debía preguntar: {question}")


def test_same_hardware_asks_nothing(config_dir: Path) -> None:
    before = station_text(config_dir)
    confirm_station(
        ConfigRepository(config_dir), camera_hardware("basler_scA1390", simulated=False), never_asked
    )
    assert station_text(config_dir) == before


def test_confirmed_change_updates_the_station(config_dir: Path) -> None:
    questions: list[str] = []
    confirm_station(
        ConfigRepository(config_dir),
        camera_hardware("logitech_c270", simulated=False),
        lambda question: questions.append(question) is None,
    )
    assert "camera: basler_scA1390 -> logitech_c270" in questions[0]
    assert ConfigRepository(config_dir).station().camera == "logitech_c270"


def test_rejected_change_stops_and_keeps_the_station(config_dir: Path) -> None:
    before = station_text(config_dir)
    with pytest.raises(StationNotConfirmedError):
        confirm_station(
            ConfigRepository(config_dir), camera_hardware("logitech_c270", simulated=False), lambda _: False
        )
    assert station_text(config_dir) == before


def test_simulated_hardware_is_not_compared(config_dir: Path) -> None:
    confirm_station(
        ConfigRepository(config_dir), camera_hardware("folder_basler", simulated=True), never_asked
    )


def test_gripper_tool_is_loaded_into_the_robot(config_dir: Path, robot: SimRobot) -> None:
    gripper = two_valve_gripper(robot)
    hardware = StationHardware(
        camera=None, camera_name=None, robot=robot, robot_name="sim", gripper=gripper, tray_name=None
    )
    confirm_station(ConfigRepository(config_dir), hardware, never_asked)
    assert robot.tool().payload_kg == pytest.approx(1.2)
    assert robot.tool().tcp.position_mm == pytest.approx([0.0, 0.0, 150.0])


def test_runner_stops_a_job_and_reports_errors() -> None:
    statuses: list[str] = []
    runner = JobRunner(statuses.append)
    started = threading.Event()

    def wait_for_stop(stop: threading.Event) -> None:
        started.set()
        stop.wait(timeout=5.0)

    assert runner.submit("esperar", wait_for_stop)
    started.wait(timeout=5.0)
    assert not runner.submit("otro", wait_for_stop)
    runner.request_stop()
    assert runner.wait(timeout_s=5.0)

    def fail(stop: threading.Event) -> None:
        raise RuntimeError("se cayó")

    runner.submit("fallar", fail)
    runner.wait(timeout_s=5.0)
    assert any("Error en 'fallar': se cayó" in status for status in statuses)


def test_run_log_writes_attempts_images_and_config(tmp_path: Path) -> None:
    run_log = RunLog.create(tmp_path / "runs", "prueba")
    try:
        logging.getLogger("prueba").warning("algo pasó")
        number = run_log.record(
            {"label": "piel", "pieces": [{"face": "piel"}]}, {"capture": np.zeros((4, 4), np.uint8)}
        )
        run_log.snapshot_config([paths.CONFIG_DIR / "cell.toml"], paths.CONFIG_DIR)
    finally:
        run_log.close()
    line = json.loads((run_log.directory / "attempts.jsonl").read_text(encoding="utf-8"))
    assert number == 1
    assert line["label"] == "piel"
    assert (run_log.directory / line["images"]["capture"]).is_file()
    assert (run_log.directory / "config" / "cell.toml").is_file()
    assert "algo pasó" in (run_log.directory / "run.log").read_text(encoding="utf-8")


def test_run_log_copies_config_files_from_outside_the_config_folder(tmp_path: Path) -> None:
    external = tmp_path / "variante_webcam.toml"
    external.write_text("[components]\n", encoding="utf-8")
    run_log = RunLog.create(tmp_path / "runs", "prueba")
    try:
        run_log.snapshot_config([external, paths.CONFIG_DIR / "cell.toml"], paths.CONFIG_DIR)
    finally:
        run_log.close()
    assert (run_log.directory / "config" / "variante_webcam.toml").is_file()
    assert (run_log.directory / "config" / "cell.toml").is_file()

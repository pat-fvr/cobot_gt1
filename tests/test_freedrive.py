"""freedrive: las teclas piden, el trabajo en vivo usa el robot; al detenerse apaga el freedrive.

Las poses con nombre se guardan en el ``robot_poses.toml`` de una carpeta de configuración temporal.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from gt1_tools.freedrive import FreedriveView

from gt1.calibration import NominalScaleProjector
from gt1.config import ConfigRepository, write_robot_poses
from gt1.config.schemas import RobotPoseConfig, SimRobotConfig
from gt1.hardware.robot.sim import SimRobot
from gt1.runtime.console import OperatorConsole
from gt1.runtime.run_log import RunLog

from .fakes import StaticCamera

WAIT_S = 5.0
SIM_JOINTS_DEG = (0.0, -90.0, 90.0, 0.0, 90.0, 0.0)


def wait_until(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def run_log(tmp_path: Path) -> Iterator[RunLog]:
    log = RunLog.create(tmp_path, "freedrive")
    yield log
    log.close()


@pytest.fixture
def repository(tmp_path: Path) -> ConfigRepository:
    root = tmp_path / "config"
    root.mkdir()
    write_robot_poses(root / "robot_poses.toml", {})
    return ConfigRepository(root)


def sim_robot() -> SimRobot:
    return SimRobot(
        SimRobotConfig(
            kind="sim",
            initial_tcp_mm=(100.0, 50.0, 200.0),
            initial_joints_deg=SIM_JOINTS_DEG,
        )
    )


def freedrive_view(robot: SimRobot, run_log: RunLog, repository: ConfigRepository) -> FreedriveView:
    return FreedriveView(
        camera=StaticCamera(np.zeros((240, 320), dtype=np.uint8), simulated=True),
        robot=robot,
        projector=NominalScaleProjector(mm_per_px=1.0),
        table_z_mm=-110.0,
        repository=repository,
        console=OperatorConsole("prueba"),
        run_log=run_log,
    )


def test_freedrive_follows_the_key_and_turns_off_when_the_live_view_stops(
    run_log: RunLog, repository: ConfigRepository
) -> None:
    with sim_robot() as robot:
        view = freedrive_view(robot, run_log, repository)
        stop = threading.Event()
        live = threading.Thread(target=view.live_job(), args=(stop,))
        live.start()

        view.toggle_freedrive()
        assert wait_until(lambda: robot.freedrive)
        view.toggle_freedrive()
        assert wait_until(lambda: not robot.freedrive)

        view.toggle_freedrive()
        assert wait_until(lambda: robot.freedrive)
        stop.set()
        live.join(WAIT_S)
        assert not robot.freedrive


def test_the_pose_key_records_the_tcp_and_the_joints(run_log: RunLog, repository: ConfigRepository) -> None:
    with sim_robot() as robot:
        view = freedrive_view(robot, run_log, repository)
        stop = threading.Event()
        live = threading.Thread(target=view.live_job(), args=(stop,))
        live.start()
        view.request_pose_record()
        assert wait_until(lambda: run_log.attempt_count == 1)
        stop.set()
        live.join(WAIT_S)
    record = json.loads((run_log.directory / "attempts.jsonl").read_text(encoding="utf-8"))
    assert record["tcp_mm"] == [100.0, 50.0, 200.0]
    assert record["joints_deg"] == list(SIM_JOINTS_DEG)


def test_a_named_pose_key_saves_the_joints_and_keeps_the_other_poses(
    run_log: RunLog, repository: ConfigRepository
) -> None:
    other = RobotPoseConfig(
        joints_deg=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0), tcp_mm=(1.0, 2.0, 3.0), taught_at="2026-11-01 10:00"
    )
    old_home = RobotPoseConfig(
        joints_deg=(20.0, 20.0, 20.0, 20.0, 20.0, 20.0), tcp_mm=(4.0, 5.0, 6.0), taught_at="2026-11-02 10:00"
    )
    write_robot_poses(repository.robot_poses_path(), {"other": other, "home": old_home})
    with sim_robot() as robot:
        view = freedrive_view(robot, run_log, repository)
        stop = threading.Event()
        live = threading.Thread(target=view.live_job(), args=(stop,))
        live.start()
        view.pose_save_action("home")()
        assert wait_until(lambda: run_log.attempt_count == 1)
        stop.set()
        live.join(WAIT_S)

    poses = repository.robot_poses()
    assert list(poses) == ["other", "home"]
    assert poses["other"] == other
    assert poses["home"].joints_deg == SIM_JOINTS_DEG
    assert poses["home"].tcp_mm == (100.0, 50.0, 200.0)
    record = json.loads((run_log.directory / "attempts.jsonl").read_text(encoding="utf-8"))
    assert record["saved_pose"] == "home"
    assert record["replaced_joints_deg"] == [20.0] * 6

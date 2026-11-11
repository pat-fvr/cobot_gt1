"""Escritura de ``robot_poses.toml``: las poses del robot con nombre que enseña freedrive.

Se guardan en articulaciones: una pose en articulaciones es única, mientras que
la misma posición del TCP puede alcanzarse con varias configuraciones del brazo
(codo arriba o abajo) y un movimiento articular hacia ella podría pasar por
cualquiera.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import tomli_w

from .loading import ConfigRepository
from .schemas import RobotPoseConfig, RobotPoses

ROBOT_POSES_HEADER = (
    "# Poses del robot con nombre, en articulaciones (grados). Las escribe gt1_tools/freedrive.py: cada\n"
    "# nombre de poses_to_teach (config/tools/freedrive.toml) tiene una tecla que guarda la pose actual.\n"
    '# Los programas eligen la suya por nombre (p. ej. pose = "home" en [settings.clear_view]).\n'
)
# Para que el archivo se lea: 0,001° mueve la punta menos de 0,02 mm a 1 m de la base.
JOINTS_DECIMALS = 3
TCP_DECIMALS = 1


def taught_pose(joints_deg: Sequence[float], tcp_mm: Sequence[float], taught_at: datetime) -> RobotPoseConfig:
    """La pose leída del robot, redondeada para guardarla."""
    return RobotPoseConfig.model_validate(
        {
            "joints_deg": tuple(round(value, JOINTS_DECIMALS) for value in joints_deg),
            "tcp_mm": tuple(round(value, TCP_DECIMALS) for value in tcp_mm),
            "taught_at": taught_at.strftime("%Y-%m-%d %H:%M"),
        }
    )


def save_robot_pose(repository: ConfigRepository, name: str, pose: RobotPoseConfig) -> RobotPoseConfig | None:
    """Guarda ``pose`` como ``name`` sin tocar las demás; devuelve la que reemplazó, si había."""
    poses = repository.robot_poses()
    write_robot_poses(repository.robot_poses_path(), {**poses, name: pose})
    return poses.get(name)


def write_robot_poses(path: Path, poses: RobotPoses) -> None:
    content = {name: pose.model_dump(mode="json") for name, pose in poses.items()}
    path.write_text(ROBOT_POSES_HEADER + tomli_w.dumps(content), encoding="utf-8")

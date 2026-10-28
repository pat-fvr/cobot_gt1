"""Conversión de la sección ``tcp`` / ``payload`` de un archivo de gripper a ``ToolSettings``."""

from __future__ import annotations

import numpy as np

from gt1.config.schemas import DigitalOutputGripperConfig, FixedToolConfig
from gt1.domain import Pose, ToolSettings


def tool_settings_from_config(config: DigitalOutputGripperConfig | FixedToolConfig) -> ToolSettings:
    return ToolSettings(
        tcp=Pose.from_rotation_vector_deg(config.tcp.position_mm, config.tcp.rotation_vector_deg),
        payload_kg=config.payload.mass_kg,
        center_of_gravity_mm=np.asarray(config.payload.center_of_gravity_mm, dtype=np.float64),
    )

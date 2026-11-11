"""Captura con el robot fuera de la vista: lo lleva a una pose enseñada y captura con él quieto.

Se decide siempre con imágenes en que el robot no tapa la escena ni la mueve:
la imagen es posterior a la llegada y a la espera de ``capture_settle_s``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from gt1.config.schemas import ClearViewSettings, RobotPoseConfig
from gt1.domain import Frame, Plan
from gt1.hardware.camera import Camera
from gt1.hardware.robot import Robot
from gt1.planning import clear_view_plan

from .captures import capture_still
from .executor import PlanExecutor


class ClearViewCamera:
    def __init__(
        self,
        *,
        camera: Camera,
        robot: Robot,
        executor: PlanExecutor,
        table_z_mm: float,
        clear_pose: RobotPoseConfig,
        settings: ClearViewSettings,
        report: Callable[[str], None],
    ) -> None:
        """``clear_pose`` es la pose ``settings.pose`` ya leída; ``report`` recibe el motivo cuando no se
        pudo capturar (para la consola)."""
        self._camera = camera
        self._robot = robot
        self._executor = executor
        self._table_z_mm = table_z_mm
        self._clear_pose = clear_pose
        self._settings = settings
        self._report = report

    def move_clear(self, stop: threading.Event) -> bool:
        """Lleva el robot fuera de la vista, sin capturar; False si no llegó."""
        execution = self._executor.execute(self.plan_from_here(), stop)
        if not execution.completed:
            self._report(execution.describe())
        return execution.completed

    def capture(self, stop: threading.Event) -> Frame | None:
        """Lleva el robot fuera de la vista y captura con él quieto; None si no llegó o se detuvo."""
        if not self.move_clear(stop):
            return None
        frame = capture_still(self._camera, self._settings.capture_settle_s, stop)
        if frame is None:
            self._report("Captura detenida por el operador.")
        return frame

    def plan_from_here(self) -> Plan:
        """El plan para llegar a la pose fuera de la vista desde donde está el robot ahora (sin capturar)."""
        settings = self._settings
        return clear_view_plan(
            self._robot.tcp_pose(),
            self._robot.joints_deg(),
            safe_z_mm=self._table_z_mm + settings.safe_height_mm,
            clear_joints_deg=self._clear_pose.joints_deg,
            pose_name=settings.pose,
            rise_speed_mm_s=settings.rise_speed_mm_s,
            joint_speed_deg_s=settings.clear_speed_deg_s,
        )

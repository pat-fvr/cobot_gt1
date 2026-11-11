"""Salir de la vista de la cámara: a la pose fija desde la que se captura.

Si la punta está baja (junto a una pieza o en la bandeja), primero sube en
vertical: el movimiento articular a la pose de captura no recorre una línea
recta y, desde abajo, podría arrastrar la pieza o chocar la bandeja. Si ya
está en la pose, no se mueve: aunque la pose quede bajo la altura segura (p. ej.
la de soltar en la bandeja fija), no sube y vuelve a bajar antes de capturar.
"""

from __future__ import annotations

import numpy as np

from gt1.config.schemas import Joints6
from gt1.domain import MoveJoints, MoveLinear, Plan, PlanStep, Pose

# Diferencia en cada articulación bajo la cual el robot ya está en la pose: muy por encima del ruido de
# lectura de los encoders y muy por debajo de cualquier movimiento que valga la pena hacer.
AT_POSE_TOL_DEG = 0.1


def clear_view_plan(
    current_tcp: Pose,
    current_joints_deg: Joints6,
    *,
    safe_z_mm: float,
    clear_joints_deg: Joints6,
    pose_name: str,
    rise_speed_mm_s: float,
    joint_speed_deg_s: float,
) -> Plan:
    """``pose_name`` es el nombre de la pose de captura, para el registro (p. ej. ``home``)."""
    steps: list[PlanStep] = []
    if is_at_pose(current_joints_deg, clear_joints_deg):
        return Plan(name="salir de la vista", steps=())
    if current_tcp.position_mm[2] < safe_z_mm:
        raised_mm = current_tcp.position_mm.copy()
        raised_mm[2] = safe_z_mm
        steps.append(MoveLinear(Pose(raised_mm, current_tcp.rotation), rise_speed_mm_s, "subir en vertical"))
    steps.append(MoveJoints(clear_joints_deg, joint_speed_deg_s, f"a la pose '{pose_name}'"))
    return Plan(name="salir de la vista", steps=tuple(steps))


def is_at_pose(joints_deg: Joints6, pose_joints_deg: Joints6) -> bool:
    difference_deg = np.abs(np.subtract(joints_deg, pose_joints_deg))
    return bool(np.all(difference_deg <= AT_POSE_TOL_DEG))

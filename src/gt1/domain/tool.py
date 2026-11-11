"""Herramienta montada en la brida del robot: lo que el controlador necesita saber de ella."""

from __future__ import annotations

from dataclasses import dataclass

from gt1.geometry import FloatArray

from .pose import Pose


@dataclass(frozen=True, eq=False)
class ToolSettings:
    """Lo que el controlador necesita saber de la herramienta: punta (TCP) y carga."""

    tcp: Pose  # punta respecto de la brida
    payload_kg: float
    center_of_gravity_mm: FloatArray  # (3,) respecto de la brida

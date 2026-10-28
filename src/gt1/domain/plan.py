"""Planes: la lista de pasos simples que una estrategia entrega y el ejecutor lleva a cabo.

Las estrategias (``gt1.planning``) no tocan hardware: solo devuelven un
``Plan``. Así una forma nueva de tomar, voltear o dejar una pieza es una
clase con un método ``plan()``, y se prueba sin robot.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pose import Pose


@dataclass(frozen=True, eq=False)
class MoveLinear:
    """Movimiento recto de la punta de la herramienta hasta ``target`` (en la base)."""

    target: Pose
    speed_mm_s: float
    description: str


@dataclass(frozen=True)
class MoveJoints:
    """Movimiento articular a ``joints_deg`` (base, hombro, codo, muñeca 1, 2, 3)."""

    joints_deg: tuple[float, float, float, float, float, float]
    speed_deg_s: float
    description: str


@dataclass(frozen=True)
class GripperCommand:
    """Ejecutar una acción del gripper (las definidas en su archivo de configuración)."""

    action: str
    description: str


@dataclass(frozen=True)
class Wait:
    duration_s: float
    description: str


PlanStep = MoveLinear | MoveJoints | GripperCommand | Wait


@dataclass(frozen=True)
class Plan:
    """Pasos en orden y para qué sirve el plan (se muestra y se registra)."""

    name: str
    steps: tuple[PlanStep, ...]

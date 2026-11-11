"""Planes: la lista de pasos simples que una estrategia entrega y el ejecutor lleva a cabo.

Las estrategias (``gt1.planning``) no tocan hardware: solo devuelven un
``Plan``. Así una forma nueva de tomar, voltear o dejar una pieza es una
clase con un método ``plan()``, y se prueba sin robot.
"""

from __future__ import annotations

from dataclasses import dataclass

from .io import DigitalInput, DigitalOutput
from .pose import Pose


@dataclass(frozen=True, eq=False)
class MoveLinear:
    """Movimiento recto de la punta de la herramienta hasta ``target`` (en la base).

    Con ``blend_radius_mm`` mayor que cero, el robot no se detiene en ``target``: lo pasa en curva, a esa
    distancia, y sigue con el movimiento recto siguiente del plan. Con cero (lo normal) se detiene ahí.
    """

    target: Pose
    speed_mm_s: float
    description: str
    blend_radius_mm: float = 0.0

    @property
    def blends_into_next(self) -> bool:
        return self.blend_radius_mm > 0.0


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


@dataclass(frozen=True)
class SetOutput:
    """Fijar una salida digital del robot que no es del gripper (p. ej. el tope de la bandeja)."""

    output: DigitalOutput
    value: bool
    description: str


@dataclass(frozen=True)
class WaitForInput:
    """Esperar, con el robot quieto, a que una entrada digital valga ``value`` (p. ej. el sensor de bandeja).

    No tiene tiempo máximo: espera hasta que llegue o se pida la parada.
    """

    sensor: DigitalInput
    value: bool
    description: str


PlanStep = MoveLinear | MoveJoints | GripperCommand | Wait | SetOutput | WaitForInput


@dataclass(frozen=True)
class Plan:
    """Pasos en orden y para qué sirve el plan (se muestra y se registra)."""

    name: str
    steps: tuple[PlanStep, ...]


def joined_plan(name: str, plans: tuple[Plan, ...]) -> Plan:
    """Un solo plan con los pasos de ``plans``: el último movimiento de uno pasa en curva al siguiente."""
    return Plan(name=name, steps=tuple(step for plan in plans for step in plan.steps))

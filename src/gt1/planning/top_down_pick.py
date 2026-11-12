"""Toma desde arriba: el gripper baja vertical sobre el centro de la pieza y la toma.

Primero va en vertical, desde donde esté, a la altura de llegada
(``approach_height_mm``), y a esa altura viaja hasta quedar sobre la pieza: no
cruza en diagonal sobre la bandeja ni sobre otras piezas. Ahí se detiene y
recién baja. Después de cerrar puede bajar un poco más
(``after_grip_descent_mm``): los dedos de un soft gripper se recogen al cerrar,
y bajar los mete bajo la pieza. Solo el punto de arriba de la subida se pasa en
curva (``blend_radius_mm``), hacia el viaje a la bandeja.

El eje Y de la herramienta se gira a la línea central de la pieza, con la
parte ancha de un gripper asimétrico hacia los lóbulos (``grasp.py``), para
que los dedos cierren a lo ancho de ella. La altura se mide desde la mesa, no
desde la pieza: la cámara ve el contorno, no el espesor.
"""

from __future__ import annotations

from gt1.config.schemas import TopDownPickConfig, WideSide
from gt1.domain import GripperCommand, MoveLinear, MoveVertical, Piece, Plan, PlanStep, Pose
from gt1.geometry import tool_pointing_down

from .base import PickStrategy
from .grasp import grasp_tool_y_deg


class TopDownPick(PickStrategy):
    def __init__(self, config: TopDownPickConfig, table_z_mm: float, wide_side: WideSide) -> None:
        self._config = config
        self._table_z_mm = table_z_mm
        self._wide_side: WideSide = wide_side  # sin anotar, pyright lo ensancha a str

    def plan(self, piece: Piece) -> Plan:
        config = self._config
        steps: list[PlanStep] = [
            GripperCommand("release", "abrir el gripper antes de bajar"),
            MoveVertical(
                self._table_z_mm + config.approach_height_mm,
                config.descend_speed_mm_s,
                "en vertical a la altura de llegada",
            ),
            MoveLinear(
                self._above(piece, config.approach_height_mm), config.travel_speed_mm_s, "sobre la pieza"
            ),
            MoveLinear(
                self._above(piece, config.grasp_height_mm), config.descend_speed_mm_s, "bajar a la pieza"
            ),
            GripperCommand("grip", "tomar la pieza"),
        ]
        if config.after_grip_descent_mm > 0:
            settled_height_mm = config.grasp_height_mm - config.after_grip_descent_mm
            steps.append(
                MoveLinear(
                    self._above(piece, settled_height_mm),
                    config.descend_speed_mm_s,
                    "bajar cerrado: los dedos bajo la pieza",
                )
            )
        steps.append(
            MoveLinear(
                self._above(piece, config.lift_height_mm),
                config.descend_speed_mm_s,
                "levantar la pieza",
                blend_radius_mm=config.blend_radius_mm,
            )
        )
        return Plan(name="tomar desde arriba", steps=tuple(steps))

    def _above(self, piece: Piece, height_mm: float) -> Pose:
        """La punta sobre el centro de la pieza, ``height_mm`` sobre la mesa, alineada con la pieza."""
        x_mm, y_mm = piece.center_mm[:2]
        return Pose.from_position(
            (x_mm, y_mm, self._table_z_mm + height_mm),
            tool_pointing_down(grasp_tool_y_deg(piece, self._wide_side)),
        )

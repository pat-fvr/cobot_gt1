"""Cómo queda el gripper sobre la pieza al tomarla: su eje Y a lo largo de la línea central de la pieza.

Un gripper asimétrico tiene una parte ancha y una angosta a lo largo del eje Y
de su TCP (``wide_side`` en su archivo). La parte ancha va hacia los lóbulos
(la mitad ancha de la pechuga) y la angosta hacia la punta: así sostiene cada
mitad con los dedos que le corresponden. Si el detector no midió hacia dónde
están los lóbulos, o el gripper es simétrico, solo se alinea el eje, con el
sentido que menos gira la muñeca.

La pieza queda tomada con su línea central fija respecto de la herramienta:
girar la herramienta gira la pieza lo mismo (``TopDownPlace``).
"""

from __future__ import annotations

from gt1.config.schemas import WideSide
from gt1.domain import Piece
from gt1.geometry import wrap_angle_deg


def grasp_tool_y_deg(piece: Piece, wide_side: WideSide) -> float:
    """Ángulo del eje Y de la herramienta (respecto del eje X de la base) al tomar la pieza."""
    if wide_side == "symmetric" or piece.lobes_direction_deg is None:
        return piece.axis_angle_deg
    toward_lobes_deg = piece.lobes_direction_deg
    return wrap_angle_deg(toward_lobes_deg if wide_side == "+y" else toward_lobes_deg + 180.0)

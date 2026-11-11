"""Colocación sobre la bandeja: el gripper llega directo sobre su centro y suelta la pieza.

Llega en línea recta a ``release_height_mm`` sobre el fondo, sin bajar ni subir
después: así puede seguir directo a la pieza siguiente. Como llega desde el
costado, esa altura no puede ser menor que la profundidad de la bandeja.

La pieza va tomada con su línea central a lo largo del eje Y de la
herramienta (``grasp.py``): girar la herramienta gira la pieza lo mismo. La
línea central de la pieza queda a lo largo del lado largo de la bandeja.

- Si el detector midió hacia dónde están los lóbulos, la pieza se gira para
  que queden hacia ``lobes_toward_deg`` (el sentido del lado largo de la
  bandeja más cercano a ese ángulo): todas las bandejas quedan iguales. El
  giro puede llegar a media vuelta.
- Si no, de los dos sentidos del lado largo se usa el que menos gira la
  muñeca desde la toma.
"""

from __future__ import annotations

from gt1.config.schemas import TopDownPlaceConfig, WideSide
from gt1.domain import GripperCommand, MoveLinear, Piece, Plan, Pose, TrayDetection
from gt1.geometry import axis_angle_nearest_deg, tool_pointing_down, wrap_angle_deg

from .base import PlaceStrategy
from .grasp import grasp_tool_y_deg


class TopDownPlace(PlaceStrategy):
    def __init__(self, config: TopDownPlaceConfig, wide_side: WideSide) -> None:
        self._config = config
        self._wide_side: WideSide = wide_side  # sin anotar, pyright lo ensancha a str

    def plan(self, piece: Piece, tray: TrayDetection) -> Plan:
        config = self._config
        self._check_clears_the_rim(tray)
        tool_y_angle_deg = self._release_tool_angle_deg(piece, tray)

        x_mm, y_mm, floor_z_mm = tray.center_mm
        release = Pose.from_position(
            (x_mm, y_mm, floor_z_mm + config.release_height_mm), tool_pointing_down(tool_y_angle_deg)
        )
        return Plan(
            name="dejar en la bandeja",
            steps=(
                MoveLinear(release, config.travel_speed_mm_s, "sobre la bandeja"),
                GripperCommand("release", "soltar la pieza"),
            ),
        )

    def _release_tool_angle_deg(self, piece: Piece, tray: TrayDetection) -> float:
        """Ángulo del eje Y de la herramienta al soltar la pieza en la bandeja."""
        grasp_deg = grasp_tool_y_deg(piece, self._wide_side)
        if piece.lobes_direction_deg is None:
            return axis_angle_nearest_deg(tray.axis_angle_deg, grasp_deg)
        target_lobes_deg = self._target_lobes_direction_deg(tray)
        turn_deg = wrap_angle_deg(target_lobes_deg - piece.lobes_direction_deg)
        return grasp_deg + turn_deg

    def _target_lobes_direction_deg(self, tray: TrayDetection) -> float:
        """Hacia dónde van los lóbulos en esta bandeja: el sentido de su lado largo más cercano al pedido."""
        return axis_angle_nearest_deg(tray.axis_angle_deg, self._config.lobes_toward_deg)

    def _check_clears_the_rim(self, tray: TrayDetection) -> None:
        depth_mm = float(tray.corners_mm[:, 2].max() - tray.center_mm[2])
        if self._config.release_height_mm < depth_mm:
            raise ValueError(
                f"release_height_mm ({self._config.release_height_mm:.0f} mm) es menor que la profundidad "
                f"de la bandeja ({depth_mm:.0f} mm): el gripper llega desde el costado a esa altura y "
                "chocaría el borde. Subirla en el archivo de la estrategia de colocación."
            )

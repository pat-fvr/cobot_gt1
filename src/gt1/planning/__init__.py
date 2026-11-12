"""Estrategias puras de toma, volteo y colocación, qué pieza va a qué bandeja, la salida de la vista y la
bandeja fija con tope."""

from .assignment import pair_pieces_with_trays
from .base import FlipStrategy, PickStrategy, PlaceStrategy
from .clear_view import clear_view_plan
from .grasp import grasp_tool_y_deg
from .top_down_pick import TopDownPick
from .top_down_place import TopDownPlace
from .tray_station import STOPPER_CLOSED, TRAY_PRESENT, tray_station_drop_plan

__all__ = [
    "STOPPER_CLOSED",
    "TRAY_PRESENT",
    "FlipStrategy",
    "PickStrategy",
    "PlaceStrategy",
    "TopDownPick",
    "TopDownPlace",
    "clear_view_plan",
    "grasp_tool_y_deg",
    "pair_pieces_with_trays",
    "tray_station_drop_plan",
]

"""Tipos del problema, sin dependencias de hardware ni de configuración.

Todo en mm, grados y segundos. Las posiciones están en el sistema de la base
del robot salvo que el nombre diga otra cosa (``_px`` = píxeles de la imagen).
"""

from .frame import Frame
from .pieces import Face, FaceResult, Piece, PieceStatus
from .plan import GripperCommand, MoveJoints, MoveLinear, Plan, PlanStep, Wait
from .pose import Pose
from .tool import DigitalOutput, OutputPort, ToolSettings
from .tray import TrayDetection
from .zone import Zone

__all__ = [
    "DigitalOutput",
    "Face",
    "FaceResult",
    "Frame",
    "GripperCommand",
    "MoveJoints",
    "MoveLinear",
    "OutputPort",
    "Piece",
    "PieceStatus",
    "Plan",
    "PlanStep",
    "Pose",
    "ToolSettings",
    "TrayDetection",
    "Wait",
    "Zone",
]

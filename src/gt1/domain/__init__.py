"""Tipos del problema, sin dependencias de hardware ni de configuración.

Todo en mm, grados y segundos. Las posiciones están en el sistema de la base
del robot salvo que el nombre diga otra cosa (``_px`` = píxeles de la imagen).
"""

from .frame import Frame
from .io import DigitalInput, DigitalOutput, InputPort, OutputPort
from .pieces import Face, FaceResult, Piece, PieceStatus
from .plan import (
    GripperCommand,
    MoveJoints,
    MoveLinear,
    MoveVertical,
    Plan,
    PlanStep,
    SetOutput,
    Wait,
    WaitForInput,
    WaitForOutput,
    joined_plan,
)
from .pose import Pose
from .tool import ToolSettings
from .tray import TrayDetection
from .zone import Zone

__all__ = [
    "DigitalInput",
    "DigitalOutput",
    "Face",
    "FaceResult",
    "Frame",
    "GripperCommand",
    "InputPort",
    "MoveJoints",
    "MoveLinear",
    "MoveVertical",
    "OutputPort",
    "Piece",
    "PieceStatus",
    "Plan",
    "PlanStep",
    "Pose",
    "SetOutput",
    "ToolSettings",
    "TrayDetection",
    "Wait",
    "WaitForInput",
    "WaitForOutput",
    "Zone",
    "joined_plan",
]

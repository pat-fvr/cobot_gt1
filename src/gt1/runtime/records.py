"""Lo que se anota en el registro de una corrida sobre piezas, bandejas y planes ejecutados."""

from __future__ import annotations

from gt1.domain import Piece, TrayDetection
from gt1.perception.scene import Detection

from .executor import PlanExecution
from .run_log import JsonValue


def detection_record(detection: Detection) -> dict[str, JsonValue]:
    """Pose, medidas, estado y cara de una pieza detectada."""
    record = piece_record(detection.piece)
    if detection.face is not None:
        record |= {"face": detection.face.face.value, "margin": round(detection.face.margin, 3)}
    return record


def piece_record(piece: Piece) -> dict[str, JsonValue]:
    """Pose, medidas y estado de una pieza."""
    record: dict[str, JsonValue] = {
        "center_mm": [round(float(value), 1) for value in piece.center_mm],
        "axis_angle_deg": round(piece.axis_angle_deg, 1),
        "length_mm": round(piece.length_mm, 1),
        "width_mm": round(piece.width_mm, 1),
        "area_mm2": round(piece.area_mm2),
        "status": piece.status.value,
    }
    if piece.lobes_direction_deg is not None:
        record["lobes_direction_deg"] = round(piece.lobes_direction_deg, 1)
    return record


def tray_record(tray: TrayDetection) -> dict[str, JsonValue]:
    """Pose de una bandeja y las medidas con que la aceptó su detector."""
    return {
        "center_mm": [round(float(value), 1) for value in tray.center_mm],
        "axis_angle_deg": round(tray.axis_angle_deg, 1),
        **{name: round(value, 3) for name, value in tray.measurements.items()},
    }


def execution_record(execution: PlanExecution) -> dict[str, JsonValue]:
    """Hasta dónde llegó un plan y por qué se cortó, si se cortó."""
    return {
        "plan": execution.plan.name,
        "outcome": execution.outcome.value,
        "steps_completed": execution.steps_completed,
        "steps": [step.description for step in execution.plan.steps],
        "reason": execution.reason,
    }

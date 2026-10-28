"""Pose (posición + orientación) de un sistema de referencia, en mm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from gt1.geometry import FloatArray, homogeneous, rotation_from_vector, vector_from_rotation


@dataclass(frozen=True, eq=False)
class Pose:
    """Posición [mm] y orientación (matriz de rotación 3x3) respecto de un sistema padre."""

    position_mm: FloatArray
    rotation: FloatArray

    @classmethod
    def from_position(cls, position_mm: ArrayLike, rotation: FloatArray | None = None) -> Pose:
        """Pose en ``position_mm``; sin ``rotation``, con la orientación del sistema padre."""
        return cls(
            position_mm=np.asarray(position_mm, dtype=np.float64).reshape(3),
            rotation=np.eye(3) if rotation is None else rotation,
        )

    @classmethod
    def from_rotation_vector_deg(cls, position_mm: ArrayLike, rotation_vector_deg: ArrayLike) -> Pose:
        """Pose con la orientación dada como vector de rotación en grados (como la consola del UR)."""
        vector_rad = np.radians(np.asarray(rotation_vector_deg, dtype=np.float64))
        return cls.from_position(position_mm, rotation_from_vector(vector_rad))

    @classmethod
    def from_matrix(cls, transform: FloatArray) -> Pose:
        return cls(position_mm=transform[:3, 3].copy(), rotation=transform[:3, :3].copy())

    def matrix(self) -> FloatArray:
        """Transformación 4x4 ``padre_from_esta``."""
        return homogeneous(self.rotation, self.position_mm)

    def rotation_vector_deg(self) -> FloatArray:
        return np.degrees(vector_from_rotation(self.rotation))

    def is_close(self, other: Pose, position_tol_mm: float, rotation_tol_deg: float) -> bool:
        """Misma pose dentro de las tolerancias."""
        relative = self.rotation.T @ other.rotation
        angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
        distance_mm = float(np.linalg.norm(self.position_mm - other.position_mm))
        return distance_mm <= position_tol_mm and angle_deg <= rotation_tol_deg

    def describe(self) -> str:
        x, y, z = self.position_mm
        rx, ry, rz = self.rotation_vector_deg()
        return f"X={x:.1f} Y={y:.1f} Z={z:.1f} mm, RX={rx:.1f} RY={ry:.1f} RZ={rz:.1f}°"

"""Interfaz de los detectores de bandeja, la bandeja apoyada en la mesa a partir de su pose, y qué cae
dentro de ella."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import cv2
import numpy as np

from gt1.config.schemas import TrayConfig
from gt1.domain import Frame, TrayDetection, Zone
from gt1.geometry import FloatArray, rotation_about_z


class TrayDetector(ABC):
    @abstractmethod
    def detect(self, frame: Frame, zone: Zone) -> list[TrayDetection]:
        """Las bandejas cuyo centro está dentro de ``zone``, de la más clara a la menos clara."""


def tray_on_table(
    center_xy_mm: tuple[float, float],
    axis_angle_deg: float,
    table_z_mm: float,
    tray: TrayConfig,
    measurements: Mapping[str, float],
) -> TrayDetection:
    """Bandeja apoyada en la mesa, con su lado largo a ``axis_angle_deg``; las esquinas son las del borde."""
    half_length_mm, half_width_mm = tray.rim_length_mm / 2.0, tray.rim_width_mm / 2.0
    corners_in_tray_mm = np.array(
        [
            [-half_length_mm, -half_width_mm, 0.0],
            [half_length_mm, -half_width_mm, 0.0],
            [half_length_mm, half_width_mm, 0.0],
            [-half_length_mm, half_width_mm, 0.0],
        ]
    )
    corners_mm = corners_in_tray_mm @ rotation_about_z(axis_angle_deg).T
    corners_mm += np.array([*center_xy_mm, table_z_mm + tray.depth_mm])
    return TrayDetection(
        center_mm=np.array([*center_xy_mm, table_z_mm]),
        axis_angle_deg=axis_angle_deg,
        corners_mm=corners_mm,
        measurements=measurements,
    )


def tray_contains(tray: TrayDetection, point_mm: FloatArray) -> bool:
    """El punto (X, Y) cae dentro del borde de la bandeja, visto desde arriba."""
    rim_xy = tray.corners_mm[:, :2].astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(rim_xy, (float(point_mm[0]), float(point_mm[1])), False) >= 0

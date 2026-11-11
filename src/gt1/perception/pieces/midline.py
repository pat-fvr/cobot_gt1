"""Línea central de una pechuga: el eje de simetría de su contorno y el sentido hacia los lóbulos.

Vista desde arriba, la pechuga entera tiene forma de corazón: dos lóbulos
separados por un surco, simétricos respecto de la línea central, y una punta
angosta en el otro extremo. La línea central es el eje que mejor refleja el
contorno sobre sí mismo (el de mayor coincidencia entre el contorno y su
reflejo); se busca cerca del eje principal de inercia, que en una forma
alargada ya está a pocos grados. El sentido hacia los lóbulos es el de la
mitad más ancha: cortando la pieza por la mitad de su largo, la mitad con más
superficie. Se mide sobre toda la mitad y no en un solo ancho, porque las
piezas casi ovaladas tienen anchos parecidos en los dos extremos y un pliegue
de piel en la punta cambia el ancho en un punto.

Todo se mide en mm de la base, sobre el contorno proyectado: la perspectiva
de la cámara no deforma la simetría.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from gt1.geometry import FloatArray, wrap_angle_deg, wrap_axis_angle_deg

# Búsqueda del eje de simetría alrededor del eje principal: primero de a 1° y después de a 0,1° en torno
# al mejor. La línea central nunca está a más de ~20° del eje principal en las piezas medidas.
SYMMETRY_SEARCH_DEG = 30.0
COARSE_STEP_DEG = 1.0
FINE_STEP_DEG = 0.1
# El contorno se dibuja a esta escala para comparar su superficie con la de su reflejo.
RASTER_MM_PER_PX = 1.0


def midline_axis_deg(outline_mm: FloatArray) -> float:
    """Ángulo [-90, 90) de la línea central: el eje de simetría del contorno (N, 2) que pasa por su centro."""
    center = outline_center(outline_mm)
    raster = _Raster(outline_mm - center)
    principal_deg = _principal_axis_deg(outline_mm)
    best_deg = _most_symmetric_deg(raster, principal_deg, SYMMETRY_SEARCH_DEG, COARSE_STEP_DEG)
    best_deg = _most_symmetric_deg(raster, best_deg, COARSE_STEP_DEG, FINE_STEP_DEG)
    return wrap_axis_angle_deg(best_deg)


def lobes_direction_deg(outline_mm: FloatArray, axis_deg: float) -> float:
    """Sentido [-180, 180) de la línea central hacia la mitad más ancha (los lóbulos)."""
    centered_mm = outline_mm - outline_center(outline_mm)
    along_outline, _ = _axis_coordinates(centered_mm, axis_deg)
    middle = 0.5 * float(along_outline.max() + along_outline.min())
    along_surface, _ = _axis_coordinates(_Raster(centered_mm).filled_points_mm(), axis_deg)
    forward_area = np.count_nonzero(along_surface > middle)
    backward_area = np.count_nonzero(along_surface < middle)
    return wrap_angle_deg(axis_deg if forward_area >= backward_area else axis_deg + 180.0)


def outline_center(outline_mm: FloatArray) -> FloatArray:
    """Centro de la superficie del contorno (N, 2), en mm."""
    moments = cv2.moments(outline_mm.astype(np.float32))
    if moments["m00"] <= 0:
        raise ValueError(f"El contorno de {len(outline_mm)} puntos no encierra superficie.")
    return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])


def _most_symmetric_deg(raster: _Raster, around_deg: float, reach_deg: float, step_deg: float) -> float:
    steps = round(reach_deg / step_deg)
    candidates = [around_deg + step * step_deg for step in range(-steps, steps + 1)]
    return max(candidates, key=raster.mirror_overlap)


def _principal_axis_deg(outline_mm: FloatArray) -> float:
    moments = cv2.moments(outline_mm.astype(np.float32))
    return float(0.5 * np.degrees(np.arctan2(2.0 * moments["mu11"], moments["mu20"] - moments["mu02"])))


def _axis_coordinates(centered_mm: FloatArray, axis_deg: float) -> tuple[FloatArray, FloatArray]:
    """Coordenadas a lo largo de la línea central y a través de ella."""
    angle = np.radians(axis_deg)
    along_axis = np.array([np.cos(angle), np.sin(angle)])
    across_axis = np.array([-along_axis[1], along_axis[0]])
    return centered_mm @ along_axis, centered_mm @ across_axis


class _Raster:
    """El contorno dibujado en una máscara, para medir cuánto coincide con su reflejo en un eje."""

    def __init__(self, centered_mm: FloatArray) -> None:
        self._points = centered_mm / RASTER_MM_PER_PX
        self._offset = float(np.abs(self._points).max()) + 2.0
        side = int(np.ceil(2.0 * self._offset)) + 1
        self._mask = self._draw(self._points, side)
        self._area = cv2.countNonZero(self._mask)

    def filled_points_mm(self) -> FloatArray:
        """Los puntos de la superficie del contorno, uno por celda de la máscara, en mm desde su centro."""
        rows, columns = np.nonzero(self._mask)
        return (
            np.column_stack([columns, rows]).astype(np.float64) * RASTER_MM_PER_PX
            - self._offset * RASTER_MM_PER_PX
        )

    def mirror_overlap(self, axis_deg: float) -> float:
        """Superficie común del contorno y su reflejo en el eje, sobre la que cubren entre los dos."""
        angle = np.radians(axis_deg)
        direction = np.array([np.cos(angle), np.sin(angle)])
        reflected = 2.0 * np.outer(self._points @ direction, direction) - self._points
        mirrored = self._draw(reflected, self._mask.shape[0])
        common = cv2.countNonZero(cv2.bitwise_and(self._mask, mirrored))
        union = self._area + cv2.countNonZero(mirrored) - common
        return common / union if union else 0.0

    def _draw(self, points: FloatArray, side: int) -> NDArray[np.uint8]:
        mask = np.zeros((side, side), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(points + self._offset).astype(np.int32).reshape(-1, 1, 2)], 1)
        return mask

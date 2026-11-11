"""Geometría reutilizable: rotaciones, transformaciones rígidas y ajuste de puntos.

Solo numpy, sin estado y sin conocer el robot ni la cámara. Las matrices de
transformación son 4x4 homogéneas y se nombran ``destino_from_origen``: un
punto en el sistema ``origen`` multiplicado por ``base_from_camera`` queda en
el sistema ``base``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_points(points: ArrayLike, dimension: int) -> FloatArray:
    """Convierte ``points`` en un arreglo (N, dimension) de float64; falla si la forma no calza."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(f"Se esperaban puntos de {dimension} coordenadas; llegó un arreglo {array.shape}.")
    return array


def rotation_from_vector(rotation_vector_rad: ArrayLike) -> FloatArray:
    """Matriz de rotación 3x3 a partir de un vector de rotación (eje × ángulo, en radianes)."""
    vector = np.asarray(rotation_vector_rad, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def vector_from_rotation(rotation: FloatArray) -> FloatArray:
    """Vector de rotación (eje × ángulo, en radianes) de una matriz de rotación 3x3."""
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-12:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # Cerca de 180° la fórmula general pierde precisión: el eje sale de la diagonal.
        axis = np.sqrt(np.clip((np.diag(rotation) + 1.0) / 2.0, 0.0, None))
        largest = int(np.argmax(axis))
        signs = np.sign(rotation[largest, :] + rotation[:, largest])
        signs[largest] = 1.0
        axis = axis * np.where(signs == 0.0, 1.0, signs)
        return axis / np.linalg.norm(axis) * angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    return axis * angle


def rotation_about_z(angle_deg: float) -> FloatArray:
    """Rotación de ``angle_deg`` grados en torno al eje Z."""
    angle = np.radians(angle_deg)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    return np.array([[cos_angle, -sin_angle, 0.0], [sin_angle, cos_angle, 0.0], [0.0, 0.0, 1.0]])


def tool_pointing_down(tool_y_angle_deg: float) -> FloatArray:
    """Orientación de una herramienta que apunta hacia abajo (su Z hacia -Z de la base).

    Su eje Y queda horizontal, a ``tool_y_angle_deg`` grados del eje X de la base.
    """
    angle = np.radians(tool_y_angle_deg)
    tool_y = np.array([np.cos(angle), np.sin(angle), 0.0])
    tool_z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(tool_y, tool_z), tool_y, tool_z])


def axis_angle_nearest_deg(axis_angle_deg: float, reference_deg: float) -> float:
    """El ángulo de un eje sin sentido (``axis_angle_deg`` + k·180) más cercano a ``reference_deg``.

    Para girar la herramienta lo menos posible: alinear un eje nunca exige más de un cuarto de vuelta.
    """
    return reference_deg + wrap_axis_angle_deg(axis_angle_deg - reference_deg)


def rotation_angle_deg(first: FloatArray, second: FloatArray) -> float:
    """Ángulo [grados] de la rotación que lleva ``first`` a ``second``."""
    relative = first.T @ second
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def homogeneous(rotation: FloatArray, translation: ArrayLike) -> FloatArray:
    """Transformación 4x4 a partir de una rotación 3x3 y una traslación."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: FloatArray) -> FloatArray:
    """Inversa de una transformación rígida 4x4 (más estable que ``np.linalg.inv``)."""
    rotation = transform[:3, :3]
    return homogeneous(rotation.T, -rotation.T @ transform[:3, 3])


def transform_points(transform: FloatArray, points: ArrayLike) -> FloatArray:
    """Aplica una transformación 4x4 a puntos (N, 3)."""
    array = as_points(points, 3)
    return array @ transform[:3, :3].T + transform[:3, 3]


def fit_rigid_transform(source_points: ArrayLike, target_points: ArrayLike) -> FloatArray:
    """Transformación rígida ``target_from_source`` que mejor lleva unos puntos a otros (Kabsch).

    Necesita al menos 3 pares de puntos no alineados. Minimiza el error
    cuadrático; no escala.
    """
    source = as_points(source_points, 3)
    target = as_points(target_points, 3)
    if len(source) != len(target):
        raise ValueError(f"Cantidades distintas de puntos: {len(source)} y {len(target)}.")
    if len(source) < 3:
        raise ValueError("Se necesitan al menos 3 pares de puntos.")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    if np.linalg.matrix_rank(source - source_center, tol=1e-6) < 2:
        raise ValueError("Los puntos están alineados: hacen falta puntos repartidos en un plano.")
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    reflection = np.sign(np.linalg.det(right_t.T @ left.T))
    rotation = right_t.T @ np.diag([1.0, 1.0, reflection]) @ left.T
    return homogeneous(rotation, target_center - rotation @ source_center)


def wrap_axis_angle_deg(angle_deg: float) -> float:
    """Ángulo de un eje sin sentido (una dirección, no un vector) llevado a [-90, 90)."""
    return (angle_deg + 90.0) % 180.0 - 90.0


def wrap_angle_deg(angle_deg: float) -> float:
    """Ángulo de un sentido (un vector) llevado a [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def position_along_mm(point_mm: ArrayLike, toward_deg: float) -> float:
    """Cuánto avanzó ``point_mm`` (usa X e Y) a lo largo de la dirección ``toward_deg`` del plano XY."""
    angle_rad = np.radians(toward_deg)
    xy_mm = np.asarray(point_mm, dtype=np.float64)[:2]
    return float(xy_mm @ np.array([np.cos(angle_rad), np.sin(angle_rad)]))

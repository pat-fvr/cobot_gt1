from __future__ import annotations

import numpy as np
import pytest

from gt1.domain import Pose
from gt1.geometry import (
    fit_rigid_transform,
    homogeneous,
    invert_transform,
    rotation_about_z,
    rotation_from_vector,
    transform_points,
    vector_from_rotation,
    wrap_axis_angle_deg,
)


@pytest.mark.parametrize(
    "vector",
    [
        (0.0, 0.0, 0.0),
        (0.1, -0.2, 0.3),
        (0.0, 0.0, np.pi - 1e-9),
        (np.pi / np.sqrt(2), np.pi / np.sqrt(2), 0),
    ],
)
def test_rotation_vector_round_trip(vector: tuple[float, float, float]) -> None:
    rotation = rotation_from_vector(vector)
    assert np.allclose(rotation_from_vector(vector_from_rotation(rotation)), rotation, atol=1e-6)


def test_fit_rigid_transform_recovers_known_transform() -> None:
    expected = homogeneous(
        rotation_about_z(25.0) @ rotation_from_vector((0.02, -0.01, 0.0)), (300.0, -150.0, 40.0)
    )
    source = np.array([[0, 0, 0], [200, 0, 0], [0, 150, 0], [200, 150, 0], [100, 75, 0]], dtype=float)
    fitted = fit_rigid_transform(source, transform_points(expected, source))
    assert np.allclose(fitted, expected, atol=1e-6)


def test_fit_rigid_transform_rejects_collinear_points() -> None:
    line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    with pytest.raises(ValueError, match="alineados"):
        fit_rigid_transform(line, line)


def test_invert_transform() -> None:
    transform = homogeneous(rotation_about_z(40.0), (1.0, 2.0, 3.0))
    assert np.allclose(invert_transform(transform) @ transform, np.eye(4))


@pytest.mark.parametrize(("angle", "expected"), [(0, 0), (90, -90), (100, -80), (-95, 85), (180, 0)])
def test_wrap_axis_angle(angle: float, expected: float) -> None:
    assert wrap_axis_angle_deg(angle) == pytest.approx(expected)


def test_pose_rotation_vector_in_degrees_matches_ur_convention() -> None:
    pose = Pose.from_rotation_vector_deg((1.0, 2.0, 3.0), (0.0, 0.0, 90.0))
    assert np.allclose(pose.rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    assert np.allclose(pose.rotation_vector_deg(), [0.0, 0.0, 90.0])
    assert pose.is_close(Pose.from_rotation_vector_deg((1.0, 2.0, 3.2), (0.0, 0.0, 90.1)), 0.5, 0.5)

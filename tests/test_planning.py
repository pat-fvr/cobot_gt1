"""Estrategias de toma y colocación, y la salida de la vista: planes deterministas, sin robot."""

from __future__ import annotations

import numpy as np
import pytest

from gt1.config.schemas import (
    InputChannel,
    Joints6,
    OutputChannel,
    TopDownPickConfig,
    TopDownPlaceConfig,
    TrayStationSettings,
    WideSide,
)
from gt1.domain import (
    GripperCommand,
    InputPort,
    MoveJoints,
    MoveLinear,
    OutputPort,
    Piece,
    PieceStatus,
    Plan,
    Pose,
    SetOutput,
    TrayDetection,
    Wait,
    WaitForInput,
)
from gt1.planning import (
    TopDownPick,
    TopDownPlace,
    clear_view_plan,
    pair_pieces_with_trays,
    tray_station_drop_plan,
)

TABLE_Z_MM = -120.0
PICK = TopDownPickConfig(
    travel_speed_mm_s=200.0,
    approach_height_mm=150.0,
    grasp_height_mm=15.0,
    after_grip_descent_mm=50.0,
    lift_height_mm=200.0,
    descend_speed_mm_s=50.0,
    blend_radius_mm=30.0,
)
PLACE = TopDownPlaceConfig(travel_speed_mm_s=200.0, release_height_mm=60.0, lobes_toward_deg=90.0)
CLEAR_JOINTS_DEG: Joints6 = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)


def piece(
    center_xy_mm: tuple[float, float], axis_angle_deg: float, lobes_direction_deg: float | None = None
) -> Piece:
    return Piece(
        contour_px=np.zeros((4, 2), dtype=np.int32),
        center_mm=np.array([*center_xy_mm, TABLE_Z_MM + 30.0]),
        axis_angle_deg=axis_angle_deg,
        length_mm=210.0,
        width_mm=160.0,
        area_mm2=26000.0,
        status=PieceStatus.OK,
        lobes_direction_deg=lobes_direction_deg,
    )


def tray(
    axis_angle_deg: float, depth_mm: float = 50.0, center_xy_mm: tuple[float, float] = (300.0, 700.0)
) -> TrayDetection:
    corners = np.array([[x, y, TABLE_Z_MM + depth_mm] for x, y in [(0, 0), (1, 0), (1, 1), (0, 1)]], float)
    return TrayDetection(
        center_mm=np.array([*center_xy_mm, TABLE_Z_MM]),
        axis_angle_deg=axis_angle_deg,
        corners_mm=corners,
        measurements={},
    )


def moves(plan: Plan) -> list[MoveLinear]:
    return [step for step in plan.steps if isinstance(step, MoveLinear)]


def tool_y(pose: Pose) -> np.ndarray:
    return pose.rotation[:, 1]


def test_pick_opens_goes_above_descends_grips_descends_closed_and_lifts_over_the_piece_center() -> None:
    plan = TopDownPick(PICK, TABLE_Z_MM, "symmetric").plan(piece((800.0, 600.0), 0.0))
    kinds = [type(step) for step in plan.steps]
    assert kinds == [GripperCommand, MoveLinear, MoveLinear, GripperCommand, MoveLinear, MoveLinear]
    assert [step.action for step in plan.steps if isinstance(step, GripperCommand)] == ["release", "grip"]
    heights_mm = [move.target.position_mm[2] - TABLE_Z_MM for move in moves(plan)]
    assert heights_mm == pytest.approx([150.0, 15.0, 15.0 - 50.0, 200.0])
    for move in moves(plan):
        assert move.target.position_mm[:2] == pytest.approx([800.0, 600.0])


def test_pick_passes_in_a_curve_over_the_piece_and_at_the_top_of_the_lift_but_stops_to_grip() -> None:
    plan = TopDownPick(PICK, TABLE_Z_MM, "symmetric").plan(piece((800.0, 600.0), 0.0))
    approach, grasp, settle, lift = moves(plan)
    assert approach.blend_radius_mm == lift.blend_radius_mm == 30.0
    assert not grasp.blends_into_next
    assert not settle.blends_into_next


def test_pick_config_rejects_a_blend_longer_than_half_the_descent() -> None:
    with pytest.raises(ValueError, match="blend_radius_mm"):
        TopDownPickConfig(**{**PICK.model_dump(), "blend_radius_mm": 70.0})


def test_pick_without_descent_after_the_grip_lifts_right_away() -> None:
    config = PICK.model_copy(update={"after_grip_descent_mm": 0.0})
    plan = TopDownPick(config, TABLE_Z_MM, "symmetric").plan(piece((800.0, 600.0), 0.0))
    heights_mm = [move.target.position_mm[2] - TABLE_Z_MM for move in moves(plan)]
    assert heights_mm == pytest.approx([150.0, 15.0, 200.0])


@pytest.mark.parametrize("axis_angle_deg", [0.0, 35.0, -80.0])
def test_pick_points_the_tool_down_with_its_long_side_along_the_piece(axis_angle_deg: float) -> None:
    plan = TopDownPick(PICK, TABLE_Z_MM, "symmetric").plan(piece((800.0, 600.0), axis_angle_deg))
    angle = np.radians(axis_angle_deg)
    for move in moves(plan):
        assert move.target.rotation[:, 2] == pytest.approx([0.0, 0.0, -1.0])
        assert abs(float(tool_y(move.target) @ [np.cos(angle), np.sin(angle), 0.0])) == pytest.approx(1.0)


def unit(angle_deg: float) -> list[float]:
    angle = np.radians(angle_deg)
    return [float(np.cos(angle)), float(np.sin(angle)), 0.0]


@pytest.mark.parametrize(
    ("wide_side", "lobes_deg", "expected_tool_y_deg"),
    [("-y", 30.0, -150.0), ("-y", -120.0, 60.0), ("+y", 30.0, 30.0)],
)
def test_an_asymmetric_gripper_takes_the_piece_with_its_wide_side_toward_the_lobes(
    wide_side: WideSide, lobes_deg: float, expected_tool_y_deg: float
) -> None:
    breast = piece((800.0, 600.0), lobes_deg, lobes_deg)
    for move in moves(TopDownPick(PICK, TABLE_Z_MM, wide_side).plan(breast)):
        assert tool_y(move.target) == pytest.approx(unit(expected_tool_y_deg), abs=1e-9)


def test_an_asymmetric_gripper_without_the_lobes_measured_only_aligns_with_the_piece() -> None:
    plan = TopDownPick(PICK, TABLE_Z_MM, "-y").plan(piece((800.0, 600.0), 35.0))
    assert tool_y(moves(plan)[0].target) == pytest.approx(unit(35.0), abs=1e-9)


@pytest.mark.parametrize("lobes_deg", [80.0, -100.0, 170.0, 10.0])
def test_the_wide_side_still_faces_the_lobes_when_the_piece_is_released(lobes_deg: float) -> None:
    """La pieza va fija en el gripper: al soltar, el lado ancho (-Y) apunta a donde quedan los lóbulos."""
    breast = piece((800.0, 600.0), lobes_deg, lobes_deg)
    plan = TopDownPlace(PLACE, "-y").plan(breast, tray(90.0))
    for move in moves(plan):
        wide_side_direction = -tool_y(move.target)
        assert wide_side_direction == pytest.approx(unit(PLACE.lobes_toward_deg), abs=1e-9)


def test_place_goes_straight_to_the_release_height_over_the_tray_and_releases_there() -> None:
    """Sin bajar ni subir: después de soltar, el robot puede ir directo a la pieza siguiente."""
    plan = TopDownPlace(PLACE, "symmetric").plan(piece((800.0, 600.0), 0.0), tray(0.0))
    assert [type(step) for step in plan.steps] == [MoveLinear, GripperCommand]
    (release,) = moves(plan)
    assert release.target.position_mm == pytest.approx([300.0, 700.0, TABLE_Z_MM + 60.0])


@pytest.mark.parametrize(
    ("piece_deg", "tray_deg", "expected_tool_y_deg"),
    [(10.0, 0.0, 0.0), (80.0, -80.0, 100.0), (-60.0, 90.0, -90.0)],
)
def test_place_aligns_with_the_tray_turning_the_wrist_at_most_a_quarter(
    piece_deg: float, tray_deg: float, expected_tool_y_deg: float
) -> None:
    plan = TopDownPlace(PLACE, "symmetric").plan(piece((800.0, 600.0), piece_deg), tray(tray_deg))
    angle = np.radians(expected_tool_y_deg)
    assert tool_y(moves(plan)[0].target) == pytest.approx([np.cos(angle), np.sin(angle), 0.0])


@pytest.mark.parametrize(
    ("piece_deg", "lobes_deg", "tray_deg", "expected_tool_y_deg"),
    [
        (80.0, 80.0, 90.0, 90.0),  # lóbulos casi hacia donde se piden: gira 10°
        (80.0, -100.0, 90.0, 270.0),  # lóbulos al revés: media vuelta
        (0.0, 0.0, -90.0, 90.0),  # bandeja a -90°: su sentido más cercano a 90° es 90°
        (-30.0, 150.0, 80.0, -100.0),  # 150° → 80°: gira -70°
    ],
)
def test_place_turns_the_piece_until_its_lobes_point_where_configured(
    piece_deg: float, lobes_deg: float, tray_deg: float, expected_tool_y_deg: float
) -> None:
    plan = TopDownPlace(PLACE, "symmetric").plan(piece((800.0, 600.0), piece_deg, lobes_deg), tray(tray_deg))
    angle = np.radians(expected_tool_y_deg)
    for move in moves(plan):
        assert tool_y(move.target) == pytest.approx([np.cos(angle), np.sin(angle), 0.0], abs=1e-9)


def test_pairing_loads_pieces_and_trays_in_order_along_the_direction() -> None:
    """Hacia +Y: la pieza de menor Y va a la bandeja de menor Y; la que sobra al final se queda."""
    first_tray, second_tray = tray(0.0, center_xy_mm=(900.0, -200.0)), tray(0.0, center_xy_mm=(900.0, 300.0))
    first, second, left_over = (
        piece((500.0, -100.0), 0.0),
        piece((500.0, 200.0), 0.0),
        piece((500.0, 600.0), 0.0),
    )
    pairs = pair_pieces_with_trays([left_over, second, first], [second_tray, first_tray], 90.0)
    assert pairs == [(first, first_tray), (second, second_tray)]


def test_pairing_follows_the_configured_direction() -> None:
    west, east = piece((0.0, 0.0), 0.0), piece((100.0, 0.0), 0.0)
    (first_piece, _), _ = pair_pieces_with_trays([east, west], [tray(0.0), tray(0.0)], 180.0)
    assert first_piece is east


def test_pairing_without_trays_or_pieces_is_empty() -> None:
    assert pair_pieces_with_trays([piece((0.0, 0.0), 0.0)], [], 0.0) == []
    assert pair_pieces_with_trays([], [tray(0.0)], 0.0) == []


def test_place_refuses_a_release_height_that_would_hit_the_tray_rim() -> None:
    with pytest.raises(ValueError, match="profundidad de la bandeja"):
        TopDownPlace(PLACE, "symmetric").plan(piece((800.0, 600.0), 0.0), tray(0.0, depth_mm=160.0))


def test_pick_config_rejects_a_grasp_above_the_approach() -> None:
    with pytest.raises(ValueError, match="grasp_height_mm"):
        TopDownPickConfig(**{**PICK.model_dump(), "grasp_height_mm": 160.0})


ELSEWHERE_JOINTS_DEG = (30.0, -80.0, 100.0, -110.0, -90.0, 0.0)


def clear_from(z_mm: float, joints_deg: Joints6 = ELSEWHERE_JOINTS_DEG) -> Plan:
    return clear_view_plan(
        Pose.from_position((500.0, 400.0, z_mm)),
        joints_deg,
        safe_z_mm=100.0,
        clear_joints_deg=CLEAR_JOINTS_DEG,
        pose_name="home",
        rise_speed_mm_s=50.0,
        joint_speed_deg_s=30.0,
    )


def test_clear_view_rises_vertically_first_when_the_tool_is_low() -> None:
    rise, to_clear = clear_from(-100.0).steps
    assert isinstance(rise, MoveLinear)
    assert rise.target.position_mm == pytest.approx([500.0, 400.0, 100.0])
    assert isinstance(to_clear, MoveJoints)
    assert to_clear.joints_deg == CLEAR_JOINTS_DEG


def test_clear_view_goes_straight_to_the_clear_pose_when_the_tool_is_high() -> None:
    assert [type(step) for step in clear_from(250.0).steps] == [MoveJoints]


def test_clear_view_does_not_move_a_robot_already_at_the_clear_pose_even_below_the_safe_height() -> None:
    """La pose de soltar en la bandeja fija puede quedar bajo la altura segura: se captura desde ahí."""
    almost_there: Joints6 = (0.05, -89.95, 90.05, -90.05, -89.95, 0.0)  # lo que se mueven por ruido
    assert clear_from(-100.0, almost_there).steps == ()


def test_the_tray_station_releases_only_with_a_tray_and_then_opens_the_stopper_for_a_while() -> None:
    station = TrayStationSettings(
        stopper=OutputChannel(port=OutputPort.STANDARD, number=3),
        tray_sensor=InputChannel(port=InputPort.STANDARD, number=1),
        stopper_open_s=3.0,
    )
    close, wait_tray, release, open_stopper, let_through, close_again = tray_station_drop_plan(station).steps

    stopper = station.stopper.digital_output()
    assert close == SetOutput(stopper, True, close.description)
    assert wait_tray == WaitForInput(station.tray_sensor.digital_input(), True, wait_tray.description)
    assert isinstance(release, GripperCommand)
    assert release.action == "release"
    assert open_stopper == SetOutput(stopper, False, open_stopper.description)
    assert let_through == Wait(3.0, let_through.description)
    assert close_again == SetOutput(stopper, True, close_again.description)

"""Construcción de cada componente a partir de su configuración (el único lugar que elige implementaciones).

Agregar una implementación nueva = agregar su ``kind`` al esquema y un caso aquí.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from gt1 import paths
from gt1.calibration import CalibrationStore, NominalScaleProjector, PixelProjector
from gt1.calibration.boards import CalibrationBoard
from gt1.calibration.boards.charuco import CharucoBoard
from gt1.calibration.boards.chessboard import Chessboard
from gt1.config import ConfigRepository
from gt1.config.schemas import (
    BackgroundPieceDetectorConfig,
    BoardConfig,
    CameraConfig,
    CameraModelCalibrationConfig,
    CellConfig,
    CharucoBoardConfig,
    ChessboardConfig,
    DigitalOutputGripperConfig,
    FaceClassifierConfig,
    FixedToolConfig,
    FolderCameraConfig,
    GripperConfig,
    HistogramFaceClassifierConfig,
    HomographyCalibrationConfig,
    MvsCameraConfig,
    NominalScaleCalibrationConfig,
    PieceDetectorConfig,
    ReferencePieceDetectorConfig,
    RimAndFloorTrayDetectorConfig,
    RobotConfig,
    SimRobotConfig,
    SkinOutlinePieceDetectorConfig,
    TextureFaceClassifierConfig,
    TopDownPickConfig,
    TopDownPlaceConfig,
    TrayConfig,
    TrayDetectorConfig,
    UrRtdeRobotConfig,
    VideoFileCameraConfig,
    ViewZoneName,
    WebcamCameraConfig,
    WideSide,
    ZoneName,
)
from gt1.domain import Frame, TrayDetection, Zone
from gt1.hardware.camera import Camera
from gt1.hardware.camera.folder import FolderCamera
from gt1.hardware.camera.mvs import MvsCamera
from gt1.hardware.camera.video import VideoFileCamera
from gt1.hardware.camera.webcam import WebcamCamera
from gt1.hardware.gripper import Gripper
from gt1.hardware.gripper.digital_outputs import DigitalOutputGripper
from gt1.hardware.gripper.fixed import FixedTool
from gt1.hardware.robot import Robot
from gt1.hardware.robot.sim import SimRobot
from gt1.hardware.robot.ur_rtde import UrRtdeRobot
from gt1.perception.face import FaceClassifier
from gt1.perception.face.histogram import HistogramFaceClassifier
from gt1.perception.face.references import ReferenceStore
from gt1.perception.face.texture import TextureFaceClassifier
from gt1.perception.image_ops import full_view_zone
from gt1.perception.pieces import PieceDetector
from gt1.perception.pieces.background import BackgroundPieceDetector
from gt1.perception.pieces.empty_scene import EmptySceneStore
from gt1.perception.pieces.reference import ReferencePieceDetector
from gt1.perception.pieces.skin_outline import SkinOutlinePieceDetector
from gt1.perception.scene import ScenePerception
from gt1.perception.tray import TrayDetector, tray_on_table
from gt1.perception.tray.rim_and_floor import RimAndFloorTrayDetector
from gt1.planning import PickStrategy, PlaceStrategy, TopDownPick, TopDownPlace


def build_camera(config: CameraConfig) -> Camera:
    match config:
        case MvsCameraConfig():
            return MvsCamera(config)
        case WebcamCameraConfig():
            return WebcamCamera(config)
        case FolderCameraConfig():
            return FolderCamera(config)
        case VideoFileCameraConfig():
            return VideoFileCamera(config)


def build_board(config: BoardConfig) -> CalibrationBoard:
    match config:
        case CharucoBoardConfig():
            return CharucoBoard(config)
        case ChessboardConfig():
            return Chessboard(config)


def build_projector(camera: CameraConfig) -> PixelProjector:
    """Paso de píxeles a mm según la calibración de la cámara."""
    match camera.calibration:
        case HomographyCalibrationConfig() | CameraModelCalibrationConfig():
            store = CalibrationStore(paths.CALIBRATION_DIR, camera.camera_id)
            return store.load_projector(camera.calibration.method)
        case NominalScaleCalibrationConfig():
            return NominalScaleProjector(camera.calibration.mm_per_px)


def build_robot(config: RobotConfig) -> Robot:
    match config:
        case UrRtdeRobotConfig():
            return UrRtdeRobot(config)
        case SimRobotConfig():
            return SimRobot(config)


def build_gripper(name: str, config: GripperConfig, robot: Robot) -> Gripper:
    match config:
        case DigitalOutputGripperConfig():
            return DigitalOutputGripper(name, config, robot)
        case FixedToolConfig():
            return FixedTool(name, config)


def build_piece_detector(
    config: PieceDetectorConfig, projector: PixelProjector, camera_id: str
) -> PieceDetector:
    match config:
        case BackgroundPieceDetectorConfig():
            return BackgroundPieceDetector(config, projector)
        case ReferencePieceDetectorConfig():
            empty_scene = EmptySceneStore(paths.REFERENCES_DIR, camera_id).load()
            return ReferencePieceDetector(config, projector, empty_scene)
        case SkinOutlinePieceDetectorConfig():
            return SkinOutlinePieceDetector(config, projector)


def build_face_classifier(config: FaceClassifierConfig, camera_id: str) -> FaceClassifier:
    match config:
        case HistogramFaceClassifierConfig():
            references = ReferenceStore(paths.REFERENCES_DIR, camera_id).load_references()
            return HistogramFaceClassifier(config, references)
        case TextureFaceClassifierConfig():
            return TextureFaceClassifier(config)


def build_tray_detector(
    config: TrayDetectorConfig, tray: TrayConfig, projector: PixelProjector
) -> TrayDetector:
    match config:
        case RimAndFloorTrayDetectorConfig():
            return RimAndFloorTrayDetector(config, tray, projector)


def build_scene_perception(
    repository: ConfigRepository,
    camera: CameraConfig,
    piece_detector: str,
    face_classifier: FaceClassifier | None,
    zone: ViewZoneName,
) -> ScenePerception:
    """Percepción de la escena en ``zone`` con la cámara ``camera``; sin clasificador, sin cara.

    El clasificador llega ya construido: qué hacer si faltan sus referencias lo decide cada programa.
    """
    projector = build_projector(camera)
    return ScenePerception(
        detector=build_piece_detector(repository.piece_detector(piece_detector), projector, camera.camera_id),
        classifier=face_classifier,
        projector=projector,
        zone_for=_zone_provider(repository.cell(), zone, projector),
    )


def _zone_provider(
    cell: CellConfig, name: ViewZoneName, projector: PixelProjector
) -> Callable[[Frame], Zone]:
    """Zona a revisar en cada imagen: una de cell.toml (fija) o toda la imagen (``full_view``)."""
    if name == "full_view":
        return lambda frame: full_view_zone(projector, frame.size_px, cell.table_z_mm)
    zone = build_zone(cell, name)
    return lambda frame: zone


def build_zone(cell: CellConfig, name: ZoneName) -> Zone:
    zone = cell.zones.by_name(name)
    return Zone(
        name=name, polygon_mm=np.array(zone.polygon_mm, dtype=np.float64), surface_z_mm=cell.table_z_mm
    )


def build_pick_strategy(config: TopDownPickConfig, cell: CellConfig, gripper: GripperConfig) -> PickStrategy:
    return TopDownPick(config, cell.table_z_mm, gripper_wide_side(gripper))


def build_place_strategy(config: TopDownPlaceConfig, gripper: GripperConfig) -> PlaceStrategy:
    return TopDownPlace(config, gripper_wide_side(gripper))


def gripper_wide_side(config: GripperConfig) -> WideSide:
    """Hacia dónde queda la parte ancha del gripper; una herramienta sin dedos es simétrica."""
    match config:
        case DigitalOutputGripperConfig():
            return config.wide_side
        case FixedToolConfig():
            return "symmetric"


def build_fixed_tray(cell: CellConfig, tray: TrayConfig) -> TrayDetection:
    """La bandeja fija de ``cell.toml``, apoyada en la mesa, con las medidas de su archivo."""
    placement = cell.tray_placement
    return tray_on_table(placement.center_mm, placement.axis_angle_deg, cell.table_z_mm, tray, {})

"""Construcción de cada componente a partir de su configuración (el único lugar que elige implementaciones).

Agregar una implementación nueva = agregar su ``kind`` al esquema y un caso aquí.
"""

from __future__ import annotations

import numpy as np

from gt1 import paths
from gt1.calibration import PixelProjector
from gt1.config.schemas import (
    BackgroundPieceDetectorConfig,
    CameraConfig,
    CellConfig,
    DigitalOutputGripperConfig,
    FaceClassifierConfig,
    FixedToolConfig,
    FolderCameraConfig,
    GripperConfig,
    HistogramFaceClassifierConfig,
    MvsCameraConfig,
    PieceDetectorConfig,
    RobotConfig,
    SimRobotConfig,
    UrRtdeRobotConfig,
    WebcamCameraConfig,
    ZoneName,
)
from gt1.domain import Zone
from gt1.hardware.camera import Camera
from gt1.hardware.camera.folder import FolderCamera
from gt1.hardware.camera.mvs import MvsCamera
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
from gt1.perception.pieces import PieceDetector
from gt1.perception.pieces.background import BackgroundPieceDetector


def build_camera(config: CameraConfig) -> Camera:
    match config:
        case MvsCameraConfig():
            return MvsCamera(config)
        case WebcamCameraConfig():
            return WebcamCamera(config)
        case FolderCameraConfig():
            return FolderCamera(config)


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


def build_piece_detector(config: PieceDetectorConfig, projector: PixelProjector) -> PieceDetector:
    match config:
        case BackgroundPieceDetectorConfig():
            return BackgroundPieceDetector(config, projector)


def build_face_classifier(config: FaceClassifierConfig, camera_id: str) -> FaceClassifier:
    match config:
        case HistogramFaceClassifierConfig():
            references = ReferenceStore(paths.REFERENCES_DIR, camera_id).load_references()
            return HistogramFaceClassifier(config, references)


def build_zone(cell: CellConfig, name: ZoneName) -> Zone:
    zone = cell.zones.by_name(name)
    return Zone(
        name=name, polygon_mm=np.array(zone.polygon_mm, dtype=np.float64), surface_z_mm=cell.table_z_mm
    )

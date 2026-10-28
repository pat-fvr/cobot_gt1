"""Archivos de calibración de una cámara: ``calibration/<camera_id>/``.

intrinsics.json     modelo de la lente (lo usan los dos métodos)
homography.json     método "homography"
camera_model.json   método "camera_model"
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gt1.config.schemas import CalibrationMethod

from .extrinsics import CameraModelCalibration, HomographyCalibration
from .intrinsics import Intrinsics
from .projection import CameraModelProjector, HomographyProjector, PixelProjector


class CalibrationMissingError(Exception):
    """La cámara no tiene la calibración que se pide."""


class CalibrationStore:
    def __init__(self, calibration_dir: Path, camera_id: str) -> None:
        self.directory = calibration_dir / camera_id
        self.camera_id = camera_id

    @property
    def intrinsics_path(self) -> Path:
        return self.directory / "intrinsics.json"

    def method_path(self, method: CalibrationMethod) -> Path:
        return self.directory / f"{method}.json"

    def load_projector(self, method: CalibrationMethod) -> PixelProjector:
        intrinsics = self.load_intrinsics()
        data = self._read(self.method_path(method), f"la calibración '{method}'")
        if method == "homography":
            return HomographyProjector(
                intrinsics, np.array(data["base_from_pixel"]), float(data["plane_z_mm"])
            )
        return CameraModelProjector(intrinsics, np.array(data["base_from_camera"]))

    def load_intrinsics(self) -> Intrinsics:
        data = self._read(self.intrinsics_path, "el modelo de la lente (intrínsecos)")
        width, height = data["image_size_px"]
        return Intrinsics(
            camera_matrix=np.array(data["camera_matrix"]),
            distortion=np.array(data["distortion"]),
            image_size_px=(int(width), int(height)),
            rms_px=float(data["rms_px"]),
        )

    def save_intrinsics(self, intrinsics: Intrinsics, view_count: int) -> Path:
        return self._write(
            self.intrinsics_path,
            {
                "camera_matrix": intrinsics.camera_matrix.tolist(),
                "distortion": intrinsics.distortion.tolist(),
                "image_size_px": list(intrinsics.image_size_px),
                "rms_px": intrinsics.rms_px,
                "view_count": view_count,
            },
        )

    def save_homography(self, calibration: HomographyCalibration) -> Path:
        return self._write(
            self.method_path("homography"),
            {
                "base_from_pixel": calibration.base_from_pixel.tolist(),
                "plane_z_mm": calibration.plane_z_mm,
                "fit_rms_mm": calibration.fit_rms_mm,
                "touch_rms_mm": calibration.touch_rms_mm,
                "note": "base_from_pixel actúa sobre píxeles sin distorsión (ver intrinsics.json).",
            },
        )

    def save_camera_model(self, calibration: CameraModelCalibration) -> Path:
        return self._write(
            self.method_path("camera_model"),
            {
                "base_from_camera": calibration.base_from_camera.tolist(),
                "reprojection_rms_px": calibration.reprojection_rms_px,
                "touch_rms_mm": calibration.touch_rms_mm,
                "note": "camera = sistema óptico de OpenCV (Z hacia la escena); mm.",
            },
        )

    def _read(self, path: Path, what: str) -> dict[str, Any]:
        if not path.is_file():
            raise CalibrationMissingError(
                f"La cámara {self.camera_id} no tiene {what} ({path}). Calíbrela con gt1_tools/calibrate.py."
            )
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def _write(self, path: Path, data: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        stamped = {"created_at": datetime.now().isoformat(timespec="seconds"), **data}
        path.write_text(json.dumps(stamped, indent=2), encoding="utf-8")
        return path

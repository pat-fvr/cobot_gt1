"""Implementaciones mínimas de interfaces para los tests."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import cv2

from gt1.config.schemas import DigitalOutputGripperConfig, SimRobotConfig
from gt1.domain import DigitalInput, DigitalOutput, Frame
from gt1.domain.frame import Image
from gt1.hardware.camera import Camera
from gt1.hardware.robot.sim import SimRobot


class StaticCamera(Camera):
    """Entrega siempre la misma imagen; se puede declarar real o simulada."""

    def __init__(self, image: Image, simulated: bool) -> None:
        self._image = image
        self._simulated = simulated

    @property
    def camera_id(self) -> str:
        return "static"

    @property
    def is_simulated(self) -> bool:
        return self._simulated

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def capture(self) -> Frame:
        return Frame(image=self._image.copy(), captured_at_s=time.time())


class SequenceCamera(Camera):
    """Entrega estas imágenes en orden, una por captura (simulada): la escena cambia entre capturas."""

    def __init__(self, images: Sequence[Image]) -> None:
        self._images = list(images)
        self.captures = 0

    @property
    def camera_id(self) -> str:
        return "sequence"

    @property
    def is_simulated(self) -> bool:
        return True

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def capture(self) -> Frame:
        if self.captures >= len(self._images):
            raise RuntimeError(f"La prueba pidió más de {len(self._images)} imágenes.")
        image = self._images[self.captures]
        self.captures += 1
        return Frame(image=image.copy(), captured_at_s=time.time())


class TrayConveyorRobot(SimRobot):
    """Robot simulado junto a una cinta con bandejas: al cerrar el tope llega una bandeja al sensor y al
    abrirlo sale."""

    def __init__(self, config: SimRobotConfig, stopper: DigitalOutput, tray_sensor: DigitalInput) -> None:
        super().__init__(config)
        self._stopper = stopper
        self._tray_sensor = tray_sensor
        self.trays_released = 0

    def set_digital_output(self, output: DigitalOutput, value: bool) -> None:
        super().set_digital_output(output, value)
        if output != self._stopper:
            return
        if not value and self.digital_input(self._tray_sensor):
            self.trays_released += 1
        self.set_digital_input(self._tray_sensor, value)


def write_video(path: Path, images: Sequence[Image], fps: float) -> None:
    """Video MJPG con estas imágenes (BGR, todas del mismo tamaño)."""
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (width, height))
    for image in images:
        writer.write(image)
    writer.release()


# Gripper que cierra con DO1 = 1 y abre con DO1 = 0, sin espera. Como JSON, igual que los archivos de
# config/: el modo estricto acepta ahí listas para las tuplas.
DO1_GRIPPER = DigitalOutputGripperConfig.model_validate_json(
    json.dumps(
        {
            "kind": "digital_outputs",
            "description": "prueba",
            "tcp": {"position_mm": [0.0, 0.0, 136.0], "rotation_vector_deg": [0.0, 0.0, 0.0]},
            "payload": {"mass_kg": 0.5, "center_of_gravity_mm": [0.0, 0.0, 60.0]},
            "wide_side": "-y",
            "actions": {
                "grip": {
                    "description": "cerrar",
                    "outputs": [{"port": "standard", "number": 1, "value": True}],
                    "settle_s": 0.0,
                },
                "release": {
                    "description": "abrir",
                    "outputs": [{"port": "standard", "number": 1, "value": False}],
                    "settle_s": 0.0,
                },
            },
        }
    )
)

"""Muestras etiquetadas y referencias del clasificador por histograma: ``references/<camera_id>/``.

    samples/piel/*.png, samples/hueso/*.png   recorte de la pieza y su máscara (``*_mask.png``)
    face_histogram.npz                        referencias generadas a partir de las muestras

Se guardan recortes y no histogramas para poder regenerar las referencias con
otros parámetros (bins, espacio de color, erosión) sin volver a fotografiar.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from gt1.domain import Face, Frame, Piece
from gt1.domain.frame import Image
from gt1.perception.image_ops import Mask, polygon_mask

from .histogram import FaceReferences

LABELED_FACES = (Face.SKIN, Face.BONE)
CROP_MARGIN_PX = 10


class ReferencesMissingError(Exception):
    """La cámara no tiene referencias del clasificador."""


class ReferenceStore:
    def __init__(self, references_dir: Path, camera_id: str) -> None:
        self.directory = references_dir / camera_id
        self.camera_id = camera_id

    @property
    def references_path(self) -> Path:
        return self.directory / "face_histogram.npz"

    def sample_dir(self, face: Face) -> Path:
        return self.directory / "samples" / face.value

    def save_sample(self, frame: Frame, piece: Piece, face: Face) -> Path:
        """Guarda el recorte de una pieza etiquetada; devuelve la ruta de la imagen."""
        if face not in LABELED_FACES:
            raise ValueError(f"Solo se guardan muestras de {', '.join(f.value for f in LABELED_FACES)}.")
        mask = polygon_mask(frame.image.shape, piece.contour_px)
        x, y, width, height = cv2.boundingRect(piece.contour_px.reshape(-1, 1, 2))
        top, left = max(y - CROP_MARGIN_PX, 0), max(x - CROP_MARGIN_PX, 0)
        bottom, right = y + height + CROP_MARGIN_PX, x + width + CROP_MARGIN_PX
        directory = self.sample_dir(face)
        directory.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = directory / f"{stem}.png"
        cv2.imwrite(str(image_path), frame.image[top:bottom, left:right])
        cv2.imwrite(str(directory / f"{stem}_mask.png"), mask[top:bottom, left:right])
        return image_path

    def sample_count(self, face: Face) -> int:
        return sum(1 for _ in self._sample_paths(face))

    def load_samples(self, face: Face) -> Iterator[tuple[Image, Mask]]:
        for image_path in self._sample_paths(face):
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(image_path.with_name(f"{image_path.stem}_mask.png")), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise RuntimeError(f"Muestra incompleta: {image_path}.")
            yield np.asarray(image, dtype=np.uint8), np.asarray(mask, dtype=np.uint8)

    def save_references(self, references: FaceReferences) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.references_path,
            color_space=references.color_space,
            bins_per_channel=references.bins_per_channel,
            skin=references.skin,
            bone=references.bone,
            skin_samples=references.skin_samples,
            bone_samples=references.bone_samples,
        )
        return self.references_path

    def load_references(self) -> FaceReferences:
        if not self.references_path.is_file():
            raise ReferencesMissingError(
                f"La cámara {self.camera_id} no tiene referencias de cara ({self.references_path}). "
                "Genérelas con gt1_tools/build_face_references.py."
            )
        with np.load(self.references_path) as data:
            color_space = str(data["color_space"])
            if color_space not in ("gray", "hsv"):
                raise ValueError(f"Espacio de color desconocido en {self.references_path}: {color_space}.")
            return FaceReferences(
                color_space="gray" if color_space == "gray" else "hsv",
                bins_per_channel=int(data["bins_per_channel"]),
                skin=np.asarray(data["skin"], dtype=np.float64),
                bone=np.asarray(data["bone"], dtype=np.float64),
                skin_samples=int(data["skin_samples"]),
                bone_samples=int(data["bone_samples"]),
            )

    def _sample_paths(self, face: Face) -> Iterator[Path]:
        directory = self.sample_dir(face)
        if directory.is_dir():
            yield from (path for path in sorted(directory.glob("*.png")) if not path.stem.endswith("_mask"))

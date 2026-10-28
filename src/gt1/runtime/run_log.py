"""Registro de una corrida: ``runs/<programa>/<fecha_hora>/``.

attempts.jsonl     una línea JSON por intento: los datos para medir
images/            imágenes de cada intento
config/            copia de los archivos de configuración usados
run.log            lo que pasó (conexiones, advertencias, errores)
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

import cv2

from gt1.domain.frame import Image

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class RunLog:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._attempts = 0
        self._log_handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
        self._log_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(self._log_handler)

    @classmethod
    def create(cls, runs_dir: Path, program: str) -> RunLog:
        directory = runs_dir / program / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        (directory / "images").mkdir(parents=True)
        return cls(directory)

    @property
    def attempt_count(self) -> int:
        return self._attempts

    def record(self, data: Mapping[str, JsonValue], images: Mapping[str, Image]) -> int:
        """Guarda un intento y sus imágenes; devuelve su número (desde 1)."""
        self._attempts += 1
        image_paths: dict[str, JsonValue] = {}
        for name, image in images.items():
            relative = Path("images") / f"{self._attempts:04d}_{name}.png"
            cv2.imwrite(str(self.directory / relative), image)
            image_paths[name] = relative.as_posix()
        line: dict[str, JsonValue] = {
            "attempt": self._attempts,
            "time": datetime.now().isoformat(timespec="milliseconds"),
            **data,
            "images": image_paths,
        }
        with (self.directory / "attempts.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
        return self._attempts

    def snapshot_config(self, files: Iterable[Path], config_root: Path) -> None:
        """Copia los archivos de configuración usados, con la misma estructura de carpetas.

        Un archivo de fuera de ``config_root`` (p. ej. pasado con ``--config``) se copia en la raíz.
        """
        for path in files:
            relative = path.relative_to(config_root) if path.is_relative_to(config_root) else Path(path.name)
            target = self.directory / "config" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    def close(self) -> None:
        logging.getLogger().removeHandler(self._log_handler)
        self._log_handler.close()

"""Lo común a todos los programas de ``gt1_test/`` y ``gt1_tools/``.

``run_program`` lee los argumentos, configura el logging, carga la
configuración del programa (``config/tests/<nombre>.toml`` o
``config/tools/<nombre>.toml``), abre el registro de la corrida, ejecuta el
cuerpo del programa y, al terminar, guarda la copia de la configuración usada.

Línea de comandos: solo ``--config RUTA`` para usar otro archivo de
configuración del programa (p. ej. una variante con la webcam).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gt1 import paths
from gt1.calibration import CalibrationMissingError
from gt1.config import ConfigError, ConfigRepository
from gt1.config.loading import ProgramKind
from gt1.perception.face.references import ReferencesMissingError

from .run_log import LOG_FORMAT, RunLog
from .station import StationHardware, StationNotConfirmedError, confirm_station
from .terminal import ask_yes_no

logger = logging.getLogger(__name__)

# Errores con un mensaje que ya dice qué hacer: se muestran sin traza.
EXPLAINED_ERRORS = (ConfigError, CalibrationMissingError, ReferencesMissingError, StationNotConfirmedError)


@dataclass(frozen=True)
class Program[ConfigT]:
    name: str
    config: ConfigT
    repository: ConfigRepository
    run_log: RunLog

    def confirm_station(self, hardware: StationHardware) -> None:
        """Compara con station.toml, pide confirmación si hay cambios y carga la herramienta en el robot."""
        confirm_station(self.repository, hardware, ask_yes_no)


def run_program[ConfigT](
    kind: ProgramKind, name: str, schema: type[ConfigT], body: Callable[[Program[ConfigT]], None]
) -> None:
    arguments = _parse_arguments(kind, name)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    repository = ConfigRepository(paths.CONFIG_DIR)
    try:
        config = repository.program_file(arguments.config, schema)
    except ConfigError as error:
        logger.error("%s", error)
        sys.exit(1)

    run_log = RunLog.create(paths.RUNS_DIR, name)
    logger.info("%s: registro de la corrida en %s", name, run_log.directory)
    exit_code = 0
    try:
        body(Program(name=name, config=config, repository=repository, run_log=run_log))
    except EXPLAINED_ERRORS as error:
        logger.error("%s", error)
        exit_code = 1
    except KeyboardInterrupt:
        logger.info("Interrumpido por el operador.")
    except Exception:
        logger.exception("El programa terminó por un error inesperado.")
        exit_code = 1
    finally:
        run_log.snapshot_config(repository.loaded_files, paths.CONFIG_DIR)
        run_log.close()
    sys.exit(exit_code)


def _parse_arguments(kind: ProgramKind, name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=name)
    parser.add_argument(
        "--config",
        type=Path,
        default=paths.CONFIG_DIR / kind / f"{name}.toml",
        help=f"configuración del programa (por omisión config/{kind}/{name}.toml)",
    )
    return parser.parse_args()

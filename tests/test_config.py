"""Los archivos de config/ cumplen su esquema, y el esquema rechaza lo que debe rechazar."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from gt1 import paths
from gt1.config import ConfigError, ConfigRepository

PROGRAM_SCHEMAS = {
    ("tests", "face_test"): ("gt1_test.face_test", "FaceTestConfig"),
    ("tests", "gripper_test"): ("gt1_test.gripper_test", "GripperTestConfig"),
    ("tools", "camera_viewer"): ("gt1_tools.camera_viewer", "CameraViewerConfig"),
    ("tools", "calibrate"): ("gt1_tools.calibrate", "CalibrateConfig"),
    ("tools", "build_face_references"): ("gt1_tools.build_face_references", "BuildFaceReferencesConfig"),
}


@pytest.fixture
def repository() -> ConfigRepository:
    return ConfigRepository(paths.CONFIG_DIR)


def test_every_component_file_is_valid(repository: ConfigRepository) -> None:
    for name in repository.available("robots"):
        repository.robot(name)
    for name in repository.available("cameras"):
        repository.camera(name)
    for name in repository.available("grippers"):
        repository.gripper(name)
    for name in repository.available("trays"):
        repository.tray(name)
    for name in repository.available("boards"):
        repository.board(name)
    repository.piece_detector("piece_background")
    repository.face_classifier("face_histogram")
    repository.cell()
    repository.station()


@pytest.mark.parametrize("program", sorted(PROGRAM_SCHEMAS), ids=lambda key: key[1])
def test_every_program_file_is_valid_and_names_existing_components(
    repository: ConfigRepository, program: tuple[str, str]
) -> None:
    module_name, schema_name = PROGRAM_SCHEMAS[program]
    schema = getattr(importlib.import_module(module_name), schema_name)
    config = repository.program_file(paths.CONFIG_DIR / program[0] / f"{program[1]}.toml", schema)
    loaders = {
        "robot": repository.robot,
        "camera": repository.camera,
        "gripper": repository.gripper,
        "piece_detector": repository.piece_detector,
        "face_classifier": repository.face_classifier,
    }
    for component, name in config.components.model_dump().items():
        loaders[component](name)


def test_every_program_file_has_a_schema() -> None:
    files = {
        (kind, path.stem) for kind in ("tests", "tools") for path in (paths.CONFIG_DIR / kind).glob("*.toml")
    }
    assert files == set(PROGRAM_SCHEMAS)


def write_camera(root: Path, body: str) -> ConfigRepository:
    (root / "cameras").mkdir(parents=True)
    (root / "cameras" / "test.toml").write_text(body, encoding="utf-8")
    return ConfigRepository(root)


FOLDER_CAMERA = (
    'kind = "folder"\ncamera_id = "x"\ncalibration_method = "homography"\ndirectory = "d"\nloop = true\n'
)


def test_valid_file_loads(tmp_path: Path) -> None:
    assert write_camera(tmp_path, FOLDER_CAMERA).camera("test").camera_id == "x"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (FOLDER_CAMERA + "extra = 1\n", "extra"),
        (FOLDER_CAMERA.replace("loop = true\n", ""), "loop"),
        (FOLDER_CAMERA.replace("loop = true", 'loop = "true"'), "loop"),
        (FOLDER_CAMERA.replace('"homography"', '"magic"'), "calibration_method"),
    ],
    ids=["campo de más", "campo que falta", "tipo equivocado", "valor no permitido"],
)
def test_invalid_file_names_the_field(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        write_camera(tmp_path, body).camera("test")


def test_missing_component_lists_the_available_ones(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Disponibles en cameras/: test"):
        write_camera(tmp_path, FOLDER_CAMERA).camera("otra")


def test_gripper_without_release_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "grippers").mkdir()
    (tmp_path / "grippers" / "g.toml").write_text(
        """kind = "digital_outputs"
description = "g"
tcp = { position_mm = [0, 0, 100], rotation_vector_deg = [0, 0, 0] }
payload = { mass_kg = 1.0, center_of_gravity_mm = [0, 0, 50] }
[actions.grip]
description = "cerrar"
outputs = [{ port = "standard", number = 1, value = true }]
settle_s = 0.1
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="release"):
        ConfigRepository(tmp_path).gripper("g")

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
    ("tools", "face_preview"): ("gt1_tools.face_preview", "FacePreviewConfig"),
    ("tools", "record_video"): ("gt1_tools.record_video", "RecordVideoConfig"),
    ("tools", "video_frames"): ("gt1_tools.video_frames", "VideoFramesConfig"),
}


def program_files() -> list[Path]:
    return sorted(path for kind in ("tests", "tools") for path in (paths.CONFIG_DIR / kind).glob("*.toml"))


def program_of(path: Path) -> tuple[str, str]:
    """Programa de un archivo: ``face_test.toml`` o una variante ``face_test_<variante>.toml``."""
    kind = path.parent.name
    matches = [
        name
        for (k, name) in PROGRAM_SCHEMAS
        if k == kind and (path.stem == name or path.stem.startswith(f"{name}_"))
    ]
    assert matches, f"{kind}/{path.name} no corresponde a ningún programa de PROGRAM_SCHEMAS."
    return kind, max(matches, key=len)


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
    for name in repository.available("perception"):
        (repository.piece_detector if name.startswith("piece_") else repository.face_classifier)(name)
    repository.cell()
    repository.station()


@pytest.mark.parametrize("path", program_files(), ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_every_program_file_is_valid_and_names_existing_components(
    repository: ConfigRepository, path: Path
) -> None:
    module_name, schema_name = PROGRAM_SCHEMAS[program_of(path)]
    schema = getattr(importlib.import_module(module_name), schema_name)
    config = repository.program_file(path, schema)
    loaders = {
        "robot": repository.robot,
        "camera": repository.camera,
        "gripper": repository.gripper,
        "piece_detector": repository.piece_detector,
        "face_classifier": repository.face_classifier,
    }
    # Un programa sin hardware (p. ej. video_frames) no tiene [components].
    components: dict[str, str] = config.components.model_dump() if hasattr(config, "components") else {}
    for component, name in components.items():
        loaders[component](name)


def test_every_program_has_its_default_file() -> None:
    files = {(path.parent.name, path.stem) for path in program_files()}
    assert set(PROGRAM_SCHEMAS) <= files


def write_camera(root: Path, body: str) -> ConfigRepository:
    (root / "cameras").mkdir(parents=True)
    (root / "cameras" / "test.toml").write_text(body, encoding="utf-8")
    return ConfigRepository(root)


FOLDER_CAMERA = """kind = "folder"
camera_id = "x"
directory = "d"
loop = true

[calibration]
method = "nominal_scale"
mm_per_px = 0.5
"""


def test_valid_file_loads(tmp_path: Path) -> None:
    assert write_camera(tmp_path, FOLDER_CAMERA).camera("test").camera_id == "x"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (FOLDER_CAMERA.replace("loop = true\n", "loop = true\nextra = 1\n"), "extra"),
        (FOLDER_CAMERA.replace("loop = true\n", ""), "loop"),
        (FOLDER_CAMERA.replace("loop = true", 'loop = "true"'), "loop"),
        (FOLDER_CAMERA.replace('"nominal_scale"', '"magic"'), "calibration"),
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

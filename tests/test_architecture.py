"""Reglas de dependencias (ver AGENTS.md): se revisan los import de todos los archivos."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROGRAM_PACKAGES = ("gt1_test", "gt1_tools")

# Capa de gt1 -> capas de gt1 que puede importar (además de sí misma).
ALLOWED_LAYERS: dict[str, set[str]] = {
    "geometry": set(),
    "paths": set(),
    "domain": {"geometry"},
    "config": {"domain", "geometry"},
    "hardware": {"domain", "config", "geometry", "paths"},
    "calibration": {"domain", "config", "geometry"},
    "perception": {"domain", "config", "geometry", "calibration"},
    "planning": {"domain", "config", "geometry"},
    "runtime": {"geometry", "paths", "domain", "config", "hardware", "calibration", "perception", "planning"},
}


def module_name(path: Path) -> str:
    relative = path.relative_to(ROOT / "src") if path.is_relative_to(ROOT / "src") else path.relative_to(ROOT)
    parts = list(relative.with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imported_modules(path: Path) -> set[str]:
    """Módulos que importa un archivo, con los import relativos resueltos."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module_name(path).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                modules.add(".".join([*base, node.module] if node.module else base))
            elif node.module:
                modules.add(node.module)
    return modules


def python_files(folder: str) -> list[Path]:
    return sorted((ROOT / folder).rglob("*.py"))


@pytest.mark.parametrize("path", python_files("gt1_test") + python_files("gt1_tools"), ids=lambda p: p.name)
def test_programs_never_import_other_programs(path: Path) -> None:
    offending = {m for m in imported_modules(path) if m.split(".")[0] in PROGRAM_PACKAGES}
    assert not offending, f"{path.name} importa programas: {offending}. Mueva lo común a src/gt1."


@pytest.mark.parametrize("path", python_files("src/gt1"), ids=module_name)
def test_library_layers(path: Path) -> None:
    name = module_name(path).split(".")
    layer = name[1] if len(name) > 1 else ""
    if layer not in ALLOWED_LAYERS:
        return
    allowed = ALLOWED_LAYERS[layer] | {layer}
    for module in imported_modules(path):
        parts = module.split(".")
        assert parts[0] not in PROGRAM_PACKAGES, f"{module_name(path)} importa el programa {module}."
        if parts[0] == "gt1" and len(parts) > 1:
            assert parts[1] in allowed, f"{module_name(path)} ({layer}) no puede importar {module}."

# SPDX-License-Identifier: Apache-2.0

import ast
import subprocess
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import build_version
from build_version import resolve_trusted_scm_version, validate_source_version

ROOT = Path(__file__).resolve().parents[2]


def _describe_command_from_setup() -> list[str]:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "SCM_GIT_DESCRIBE_COMMAND" for target in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, list)
            return value
    raise AssertionError("SCM_GIT_DESCRIBE_COMMAND is not defined in setup.py")


def test_source_version_ignores_namespaced_mirror_tags() -> None:
    command = _describe_command_from_setup()
    described = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()

    assert described.startswith("v")
    assert not described.startswith("upstream/")
    version = resolve_trusted_scm_version(ROOT, command)
    assert Version(version) >= Version("0.23")


@pytest.mark.parametrize("version", ["0.0.0", "0.1.dev1", "0.19.1rc2.dev999"])
def test_stale_or_fallback_source_versions_are_rejected(version: str) -> None:
    with pytest.raises(RuntimeError, match="Refusing stale or fallback"):
        validate_source_version(version)


def test_shallow_checkout_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_version, "_git_output", lambda *_args: "true")
    with pytest.raises(RuntimeError, match="shallow checkout"):
        resolve_trusted_scm_version(ROOT, _describe_command_from_setup())


def test_wheel_build_exposes_torch_cmake_prefix() -> None:
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "torch.utils.cmake_prefix_path" in setup_source
    assert 'torch_query_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"' in setup_source
    assert "-DCMAKE_PREFIX_PATH=" in setup_source


def test_fastapi_constraint_overlaps_verified_core() -> None:
    requirements = [
        Requirement(line)
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    fastapi = next(item for item in requirements if item.name == "fastapi")

    assert Version("0.133.0") in fastapi.specifier
    assert Version("0.136.0") in fastapi.specifier
    assert Version("0.137.0") not in fastapi.specifier

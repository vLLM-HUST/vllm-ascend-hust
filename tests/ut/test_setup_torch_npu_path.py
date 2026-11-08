# SPDX-License-Identifier: Apache-2.0
"""The native build must use torch_npu from its selected Python environment."""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def _discovery_function(monkeypatch: pytest.MonkeyPatch, spec: object):
    source = Path(__file__).parents[2] / "setup.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "discover_torch_npu_path"
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
    scope = {"importlib": importlib, "os": os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    return scope["discover_torch_npu_path"]


def test_torch_npu_path_uses_build_interpreter_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = tmp_path / "site-packages" / "torch_npu"
    header = package / "include/torch_npu/csrc/core/npu/NPUStream.h"
    header.parent.mkdir(parents=True)
    header.write_text("// test header\n", encoding="utf-8")
    discover = _discovery_function(monkeypatch, SimpleNamespace(submodule_search_locations=[str(package)]))

    assert discover() == str(package)


def test_torch_npu_path_rejects_missing_package_or_headers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not installed"):
        _discovery_function(monkeypatch, None)()
    package = tmp_path / "torch_npu"
    package.mkdir()
    discover = _discovery_function(monkeypatch, SimpleNamespace(submodule_search_locations=[str(package)]))
    with pytest.raises(RuntimeError, match="development header is unavailable"):
        discover()

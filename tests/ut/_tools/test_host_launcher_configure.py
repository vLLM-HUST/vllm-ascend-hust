# SPDX-License-Identifier: Apache-2.0
"""Exercise the real setup configure method without importing device packages."""

import ast
import importlib.util
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("launcher", [None, "", "/opt/compiler cache/sccache", "ccache;--verbose"])
def test_setup_forwards_host_launcher_including_empty_default(tmp_path, monkeypatch, launcher):
    if launcher is None:
        monkeypatch.delenv("VLLM_ASCEND_HOST_COMPILER_LAUNCHER", raising=False)
    else:
        monkeypatch.setenv("VLLM_ASCEND_HOST_COMPILER_LAUNCHER", launcher)
    monkeypatch.setenv("SOC_VERSION", "ascend910b1")
    spec = importlib.util.spec_from_file_location("build_envs", ROOT / "vllm_ascend/envs.py")
    assert spec is not None and spec.loader is not None
    envs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(envs)
    module = ast.parse((ROOT / "setup.py").read_text())
    helper = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "check_or_set_default_env")
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "cmake_build_ext")
    configure = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "configure")
    subprocess = Mock()
    subprocess.check_output.side_effect = [b"/pybind", b"/torch", b"/site-packages"]
    namespace = dict(
        os=os,
        sys=sys,
        logging=logging,
        envs=envs,
        subprocess=subprocess,
        CMakeExtension=object,
        ROOT_DIR=str(ROOT),
        develop=type("Develop", (), {}),
        get_paths=lambda: {"include": "/python/include"},
    )
    exec(compile(ast.Module(body=[helper, configure], type_ignores=[]), str(ROOT / "setup.py"), "exec"), namespace)
    builder = SimpleNamespace(
        build_temp=str(tmp_path),
        build_lib="build/lib",
        distribution=SimpleNamespace(get_command_obj=lambda _: object()),
    )
    configure_fn = cast(Callable[..., Any], namespace["configure"])
    configure_fn(builder, SimpleNamespace(cmake_lists_dir=str(ROOT)))
    args = subprocess.check_call.call_args.args[0]
    assert f"-DVLLM_ASCEND_HOST_COMPILER_LAUNCHER={launcher or ''}" in args
    assert not any(a.startswith("-DCMAKE_CXX_COMPILER_LAUNCHER=") for a in args)

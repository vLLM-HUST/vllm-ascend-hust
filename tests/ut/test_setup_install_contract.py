# SPDX-License-Identifier: Apache-2.0
"""Contracts for the setuptools install/build boundary."""

import ast
from pathlib import Path


def test_custom_install_honors_skip_build() -> None:
    """Wheel installation must not compile Ascend extensions a second time."""
    setup_path = Path(__file__).parents[2] / "setup.py"
    module = ast.parse(setup_path.read_text(encoding="utf-8"))

    custom_install = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "custom_install"
    )
    run_method = next(node for node in custom_install.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    skip_build_guard = next(node for node in run_method.body if isinstance(node, ast.If))

    assert ast.unparse(skip_build_guard.test) == "not self.skip_build"
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_command"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "build_ext"
        for node in ast.walk(skip_build_guard)
    )

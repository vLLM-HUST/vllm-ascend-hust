# SPDX-License-Identifier: Apache-2.0

from types import ModuleType

import pytest

import vllm_ascend.kv_cache_planning as planning_compat


def test_load_kv_cache_planning_falls_back_to_release_module(monkeypatch) -> None:
    fallback = ModuleType("vllm.v1.core.kv_cache_utils")
    calls = []

    def import_module(name: str) -> ModuleType:
        calls.append(name)
        if name == "vllm.v1.core.kv_cache_planning":
            error = ModuleNotFoundError(name)
            error.name = name
            raise error
        return fallback

    monkeypatch.setattr(planning_compat, "import_module", import_module)

    assert planning_compat._load_kv_cache_planning() is fallback
    assert calls == [
        "vllm.v1.core.kv_cache_planning",
        "vllm.v1.core.kv_cache_utils",
    ]


def test_load_kv_cache_planning_preserves_nested_import_errors(monkeypatch) -> None:
    def import_module(name: str) -> ModuleType:
        error = ModuleNotFoundError("missing dependency")
        error.name = "missing_dependency"
        raise error

    monkeypatch.setattr(planning_compat, "import_module", import_module)

    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        planning_compat._load_kv_cache_planning()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Exercise relocation without loading a platform or initializing an NPU."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "vllm_ascend" / "_kv_cache_planning_compat.py"


def load_adapter(available, actual_module):
    spec = importlib.util.spec_from_file_location("tested_kv_planning_adapter", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with (
        patch("importlib.util.find_spec", return_value=object() if available else None),
        patch("importlib.import_module", return_value=actual_module) as imported,
    ):
        spec.loader.exec_module(module)
    return module, imported


@pytest.mark.parametrize(
    "available,expected",
    [(True, "vllm.v1.core.kv_cache_planning"), (False, "vllm.v1.core.kv_cache_utils")],
)
def test_patches_and_consumers_share_actual_host_namespace(available, expected):
    actual = ModuleType(expected)
    original = object()
    actual.get_kv_cache_groups = original
    adapter, imported = load_adapter(available, actual)
    imported.assert_called_once_with(expected)
    assert adapter.kv_cache_planning is actual
    assert adapter.get_kv_cache_groups is original
    replacement = object()
    adapter.kv_cache_planning.get_kv_cache_groups = replacement
    assert adapter.get_kv_cache_groups is replacement
    assert actual.get_kv_cache_groups is replacement


def test_unrelated_new_module_import_failure_does_not_fall_back():
    spec = importlib.util.spec_from_file_location("tested_kv_planning_failure", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.import_module", side_effect=ImportError("missing dependency")) as imported,
        pytest.raises(ImportError, match="missing dependency"),
    ):
        spec.loader.exec_module(module)
    imported.assert_called_once_with("vllm.v1.core.kv_cache_planning")


def test_missing_host_helper_is_not_synthesized():
    actual = ModuleType("actual_host")
    adapter, _ = load_adapter(True, actual)
    with pytest.raises(AttributeError):
        _ = adapter.unavailable_helper
    assert not hasattr(actual, "unavailable_helper")

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

PLATFORM_PATH = Path(__file__).parents[2] / "vllm_ascend" / "platform.py"


def _load_sync_helpers():
    tree = ast.parse(PLATFORM_PATH.read_text())
    names = {
        "_ensure_ascend_compilation_config_dict",
        "_sync_npugraph_ex_to_additional_config",
    }
    functions: list[ast.stmt] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=functions, type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(module, PLATFORM_PATH, "exec"), namespace)
    return namespace


HELPERS = _load_sync_helpers()
ENSURE_CONFIG = HELPERS["_ensure_ascend_compilation_config_dict"]
SYNC_CONFIG = HELPERS["_sync_npugraph_ex_to_additional_config"]


@pytest.mark.parametrize("additional_config", [None, {}, {"ascend_compilation_config": None}])
def test_sync_materializes_missing_worker_config(additional_config):
    vllm_config = SimpleNamespace(additional_config=additional_config)
    ascend_config = SimpleNamespace(ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=False))

    SYNC_CONFIG(vllm_config, ascend_config)

    assert vllm_config.additional_config == {"ascend_compilation_config": {"enable_npugraph_ex": False}}


def test_sync_preserves_serializable_worker_fields():
    vllm_config = SimpleNamespace(
        additional_config={
            "ascend_compilation_config": {"enable_static_kernel": True},
            "unrelated": "preserved",
        }
    )
    ascend_config = SimpleNamespace(ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=False))

    SYNC_CONFIG(vllm_config, ascend_config)

    assert vllm_config.additional_config == {
        "ascend_compilation_config": {
            "enable_npugraph_ex": False,
            "enable_static_kernel": True,
        },
        "unrelated": "preserved",
    }


@pytest.mark.parametrize("invalid", [False, [], "invalid"])
def test_sync_rejects_invalid_worker_config_without_overwriting(invalid):
    vllm_config = SimpleNamespace(additional_config={"ascend_compilation_config": invalid})
    ascend_config = SimpleNamespace(ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=False))

    with pytest.raises(TypeError, match="must be a dict or None"):
        SYNC_CONFIG(vllm_config, ascend_config)

    assert vllm_config.additional_config["ascend_compilation_config"] is invalid


def test_check_and_update_normalizes_before_ascend_config_init():
    source = PLATFORM_PATH.read_text()
    update_config = source.index("check_and_update_config")
    normalize = source.index("_ensure_ascend_compilation_config_dict(vllm_config)", update_config)
    initialize = source.index("init_ascend_config(vllm_config)", normalize)

    assert normalize < initialize
    vllm_config = SimpleNamespace(additional_config=None)
    assert ENSURE_CONFIG(vllm_config) == {}

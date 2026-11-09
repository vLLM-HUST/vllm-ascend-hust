# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for vLLM KV-cache planning helpers.

Current vLLM keeps these helpers in ``kv_cache_planning``. Supported release
branches still expose them from ``kv_cache_utils``.
"""

from importlib import import_module
from types import ModuleType


def _load_kv_cache_planning() -> ModuleType:
    try:
        return import_module("vllm.v1.core.kv_cache_planning")
    except ModuleNotFoundError as exc:
        if exc.name != "vllm.v1.core.kv_cache_planning":
            raise
        return import_module("vllm.v1.core.kv_cache_utils")


kv_cache_planning = _load_kv_cache_planning()

create_kv_cache_group_specs = kv_cache_planning.create_kv_cache_group_specs
get_kv_cache_groups = kv_cache_planning.get_kv_cache_groups
may_override_num_blocks = kv_cache_planning.may_override_num_blocks
unify_hybrid_kv_cache_specs = kv_cache_planning.unify_hybrid_kv_cache_specs

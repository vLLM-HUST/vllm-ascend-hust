# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Resolve the actual Host planning module across the upstream module split.

Planning algorithms moved out of kv_cache_utils without changing their contract.
Keep aliases to the real module, so Ascend's patches and Host consumers share
one namespace. Do not copy algorithms or backfill the obsolete Host namespace.
"""

from importlib import import_module
from importlib.util import find_spec
from typing import Any

_NEW_PLANNING_MODULE = "vllm.v1.core.kv_cache_planning"

if find_spec(_NEW_PLANNING_MODULE) is not None:
    kv_cache_planning = import_module(_NEW_PLANNING_MODULE)
else:
    kv_cache_planning = import_module("vllm.v1.core.kv_cache_utils")


def __getattr__(name: str) -> Any:
    """Resolve imported helpers from the actual selected Host module."""
    return getattr(kv_cache_planning, name)

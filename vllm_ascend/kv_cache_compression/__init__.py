# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral Ascend hooks for transactional KV cache compression."""

from vllm_ascend.kv_cache_compression.registry import (
    get_kv_cache_compression_provider,
)

__all__ = ["get_kv_cache_compression_provider"]

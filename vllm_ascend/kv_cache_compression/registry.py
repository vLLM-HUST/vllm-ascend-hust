# SPDX-License-Identifier: Apache-2.0
"""Lazy discovery of independently packaged Ascend compression providers."""

from importlib.metadata import EntryPoint, entry_points
from typing import Any

PROVIDER_ENTRY_POINT_GROUP = "vllm_ascend.kv_cache_compression_providers"


def _matching_entry_points(provider: str) -> list[EntryPoint]:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=PROVIDER_ENTRY_POINT_GROUP, name=provider))
    legacy_discovered: Any = discovered
    return [
        entry_point
        for entry_point in legacy_discovered.get(PROVIDER_ENTRY_POINT_GROUP, ())
        if entry_point.name == provider
    ]


def get_kv_cache_compression_provider(config: Any) -> Any:
    """Load exactly one provider matching the Core configuration."""
    matches = _matching_entry_points(config.provider)
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one installed Ascend KV cache compression "
            f"provider named {config.provider!r}, found {len(matches)}"
        )
    factory = matches[0].load()
    return factory(config)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True, slots=True)
class C8ContinuingPrefillProviderConfig:
    """Static C8 attention properties supplied to a provider factory."""

    layer_name: str
    num_heads: int
    num_kv_heads: int
    head_size: int
    scale: float
    kv_cache_dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class C8ContinuingPrefillRequest:
    """One host-approved C8 continuing-prefill invocation.

    ``query`` and ``output`` cover only the prefill tokens. ``key_cache`` and
    ``value_cache`` are the paged five-dimensional NZ INT8 views selected by
    ``block_table``. Query lengths are cumulative TND lengths; KV lengths are
    per-request valid lengths.

    A provider must write its result into ``output``. It must not retain any
    request tensor after the call. Capture-only temporary tensors that must
    survive graph replay are returned through
    :class:`C8ContinuingPrefillResult`.
    """

    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    key_antiquant_scale: torch.Tensor
    value_antiquant_scale: torch.Tensor
    key_antiquant_offset: torch.Tensor
    value_antiquant_offset: torch.Tensor
    output: torch.Tensor
    attention_mask: torch.Tensor | None
    actual_seq_lengths_q: tuple[int, ...]
    actual_seq_lengths_kv: tuple[int, ...]
    num_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    scale: float
    sparse_mode: int
    capturing: bool


@dataclass(frozen=True, slots=True)
class C8ContinuingPrefillResult:
    """Successful provider result.

    Returning this object means that ``request.output`` is complete. A provider
    declines before execution through ``is_eligible``. Once it accepts a
    request, exceptions and invalid results are propagated; they are never
    converted into a silent fallback.

    ``workspace`` contains tensors whose addresses are captured by an ACL
    graph. The host keeps strong references to capture workspaces for the
    lifetime of the attention implementation, matching the graph cache
    lifetime. Eager workspaces are retained until the next invocation.
    """

    workspace: tuple[torch.Tensor, ...] = ()


@runtime_checkable
class C8ContinuingPrefillProvider(Protocol):
    """External implementation of paged C8 continuing-prefill attention."""

    def is_eligible(self, request: C8ContinuingPrefillRequest) -> bool:
        """Return whether this provider accepts the invocation.

        This method must be side-effect free: it must not mutate request
        buffers, launch device work, synchronize the device, or allocate graph
        workspace. Returning ``False`` preserves the built-in host path.
        """
        ...

    def forward(self, request: C8ContinuingPrefillRequest) -> C8ContinuingPrefillResult: ...


def load_c8_continuing_prefill_provider(
    factory_path: str,
    config: C8ContinuingPrefillProviderConfig,
) -> C8ContinuingPrefillProvider:
    """Load ``module:factory`` and build one provider for an attention layer."""

    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name or ":" in attribute_name:
        raise ValueError(
            f"c8_continuing_prefill_provider must be a non-empty 'module:factory' path, got {factory_path!r}"
        )

    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"C8 continuing-prefill provider factory {factory_path!r} is not callable")

    provider = factory(config)
    if not isinstance(provider, C8ContinuingPrefillProvider):
        raise TypeError(
            f"C8 continuing-prefill provider factory {factory_path!r} returned "
            f"{type(provider).__name__}, which does not implement "
            "is_eligible(request) and forward(request)"
        )
    return provider

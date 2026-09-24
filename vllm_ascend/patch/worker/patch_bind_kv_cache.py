"""Compatibility wrapper for vLLM's KV-cache binding transition.

Current vLLM provides ``AttentionLayerBase.bind_kv_cache`` and its core
``bind_kv_cache`` helper handles standardized strided cache views. v0.29 still
needs the legacy Ascend binder because the layer method was abstract there.
The exported wrapper also remains for the v2 adaptor, which imports it as an
explicit dependency.
"""

from collections import defaultdict
from collections.abc import Sequence

import torch
import vllm.v1.worker.utils as utils
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.worker.utils import (
    bind_kv_cache as _core_bind_kv_cache,
)
from vllm.v1.worker.utils import (
    extract_layer_index,
)

from vllm_ascend.utils import vllm_version_is

_core_bind_mamba_cache = MambaBase.bind_kv_cache


def bind_mamba_cache(self, kv_cache: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
    """Accept native dense state planes, retaining core binding for raw pages.

    Ascend's layer-compact hybrid allocator has already unpacked each state
    into a contiguous plane. Sending those views through the core's raw-page
    unpacker would reinterpret the bytes a second time.
    """
    if not isinstance(kv_cache, tuple):
        return _core_bind_mamba_cache(self, kv_cache)
    shapes = tuple(self.get_state_shape())
    dtypes = tuple(self.get_state_dtype())
    if len(kv_cache) != len(shapes) or len(shapes) != len(dtypes):
        raise ValueError("Native Mamba cache state count does not match the layer")
    block_counts = set()
    for state, shape, dtype in zip(kv_cache, shapes, dtypes):
        if state.shape[1:] != shape or state.dtype != dtype or not state.is_contiguous():
            raise ValueError("Native Mamba cache state shape, dtype or contiguity is invalid")
        block_counts.add(state.shape[0])
    if len(block_counts) != 1:
        raise ValueError("Native Mamba cache states disagree on block count")
    self.kv_cache = kv_cache


if not vllm_version_is("0.29.0"):
    MambaBase.bind_kv_cache = bind_mamba_cache


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
    kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None,
) -> None:
    """
    Bind caches through the API implemented by the installed vLLM line.

    v0.29 still needs Ascend's legacy binder. Current core understands
    standardized strided views and layer-specific binding, so replacing it
    there would discard cache-group metadata.
    """
    if not vllm_version_is("0.29.0"):
        _core_bind_kv_cache(
            kv_caches,
            forward_context,
            runner_kv_caches,
            num_attn_module,
            kv_cache_groups,
        )
        return

    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        # remove some codes for the typical case of encoder-decoder model, e.g., bart.
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache
    # v0.29.0 predates ReplaySSM ring trackers.


if vllm_version_is("0.29.0"):
    utils.bind_kv_cache = bind_kv_cache


def bind_kv_cache_to_layers(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    num_attn_module: int = 1,
    kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None,
) -> None:
    """Ascend binding for vLLM main (#53781).

    Upstream init_kv_cache switched from bind_kv_cache to
    bind_kv_cache_to_layers on main, which calls each layer's bind_kv_cache
    with the standardized single-tensor layout (vLLM #51718). Ascend
    allocates per-layer (k, v) tuples, so assign the raw allocation directly,
    matching the Ascend bind_kv_cache patch above.
    """
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache
    ordered_layer_names = sorted(kv_caches, key=lambda name: extract_layer_index(name, num_attn_module))
    utils.share_replayssm_ring_trackers(ordered_layer_names, forward_context, kv_cache_groups)


utils.bind_kv_cache_to_layers = bind_kv_cache_to_layers

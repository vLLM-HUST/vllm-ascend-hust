# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native dense state planes over standardized layer-compact HMA pools.

The core owns pool sizes, layer offsets and block IDs. Ascend kernels require
each state plane to be contiguous across blocks, unlike the core's interleaved
per-block views. Partition every aliased pool using the SAME plane boundaries:
conv / key-or-SSM / value-or-padding. Different block IDs then remain disjoint
even when their owning groups use different state types. No per-step copy or
additional cache allocation is needed.
"""

import math

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_layout import KVCacheLayout


def allocate_native_hybrid_cache(
    config: KVCacheConfig, device: torch.device, layout: KVCacheLayout, kernel_block_sizes: list[int]
) -> dict[str, tuple[torch.Tensor, ...]]:
    if layout != KVCacheLayout.LBHNC:
        raise ValueError("Native hybrid state planes require layer-compact LBHNC pool placement")
    specs = {}
    for group_id, group in enumerate(config.kv_cache_groups):
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            specs[name] = (spec, kernel_block_sizes[group_id])
    sizes = {tensor.size for tensor in config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("Hybrid cache descriptors must share one backing allocation")
    blocks = config.num_blocks
    # Validate every descriptor before allocating device memory.
    pools: dict[tuple[int, int], tuple[int, int, int]] = {}
    for tensor in config.kv_cache_tensors:
        for index, name in enumerate(tensor.layers):
            spec, kernel_size = specs[name]
            page = spec.page_size_bytes
            offset = tensor.offset + index * tensor.layer_stride
            if tensor.block_stride != page or tensor.layer_stride < blocks * page:
                raise ValueError("Native hybrid cache requires non-interleaved layer pools")
            if offset < 0 or offset + blocks * page > tensor.size:
                raise ValueError("Hybrid layer pool lies outside its backing allocation")
            if isinstance(spec, MambaSpec):
                planes = tuple(
                    math.prod(shape) * get_dtype_size(dtype) for shape, dtype in zip(spec.shapes, spec.dtypes)
                )
                if len(planes) not in (1, 2):
                    raise ValueError("Native hybrid cache supports one or two Mamba state planes")
                prefix, state = (0, planes[0]) if len(planes) == 1 else planes
            elif isinstance(spec, AttentionSpec):
                if spec.head_size != spec.head_size_v or spec.block_size % kernel_size:
                    raise ValueError("Native hybrid cache requires equal K/V head sizes and divisible kernel blocks")
                state = spec.block_size * spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
                prefix = page - 2 * state
            else:
                raise ValueError(f"Unsupported native hybrid cache spec: {type(spec).__name__}")
            if prefix < 0 or prefix + state > page:
                raise ValueError("Native state planes exceed cache page size")
            key = (offset, blocks * page)
            boundaries = (prefix, state, page)
            previous = pools.setdefault(key, boundaries)
            if previous != boundaries:
                raise ValueError("Aliased attention/Mamba pools disagree on native plane boundaries")
    # Partial overlaps are not safe: all aliases must describe identical pools.
    ordered = sorted(pools)
    for (offset, size), (next_offset, _) in zip(ordered, ordered[1:]):
        if offset + size > next_offset:
            raise ValueError("Hybrid layer pools partially overlap")

    backing = torch.zeros(sizes.pop(), dtype=torch.int8, device=device)
    result = {}
    for tensor in config.kv_cache_tensors:
        for index, name in enumerate(tensor.layers):
            spec, kernel_size = specs[name]
            offset = tensor.offset + index * tensor.layer_stride
            raw = backing.narrow(0, offset, blocks * spec.page_size_bytes)
            prefix, state, _ = pools[(offset, raw.numel())]
            if isinstance(spec, MambaSpec):
                views, start = [], 0
                for shape, dtype in zip(spec.shapes, spec.dtypes):
                    size = blocks * math.prod(shape) * get_dtype_size(dtype)
                    views.append(raw.narrow(0, start, size).view(dtype).view(blocks, *shape))
                    start += size
                result[name] = tuple(views)
            else:
                shape = (blocks * spec.block_size // kernel_size, kernel_size, spec.num_kv_heads, spec.head_size)
                result[name] = tuple(
                    raw.narrow(0, blocks * (prefix + plane * state), blocks * state).view(spec.dtype).view(shape)
                    for plane in range(2)
                )
    return result

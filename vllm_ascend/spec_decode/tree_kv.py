# SPDX-License-Identifier: Apache-2.0
"""Token-granular KV compaction used after tree verification."""

from collections.abc import Iterable

import torch
import torch_npu


def verify_greedy_tree_device(
    draft_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    max_depth: int,
    placeholder_token_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify a uniform greedy draft tree without device-to-host transfers."""
    if draft_token_ids.numel() != batch_size * num_nodes:
        raise ValueError("draft token count does not match the uniform tree shape")
    if parent_indices.numel() != draft_token_ids.numel():
        raise ValueError("parent indices must align with draft tokens")
    if target_token_ids.numel() != draft_token_ids.numel():
        raise ValueError("target token count must align with draft tokens")
    if bonus_token_ids.numel() != batch_size:
        raise ValueError("one bonus token is required per request")

    drafts = draft_token_ids.reshape(batch_size, num_nodes)
    targets = target_token_ids.reshape(batch_size, num_nodes)
    parents = parent_indices.reshape(batch_size, num_nodes)
    predictions = torch.cat(
        (targets, bonus_token_ids.reshape(batch_size, 1)), dim=1
    )
    outputs = torch.full(
        (batch_size, max_depth + 1),
        placeholder_token_id,
        dtype=draft_token_ids.dtype,
        device=draft_token_ids.device,
    )
    accepted = torch.full(
        (batch_size, max_depth),
        -1,
        dtype=torch.int32,
        device=draft_token_ids.device,
    )

    node_ids = torch.arange(
        num_nodes, dtype=torch.int32, device=draft_token_ids.device
    ).unsqueeze(0)
    no_match = torch.full_like(node_ids, num_nodes)
    current_parent = torch.full(
        (batch_size,), -1, dtype=torch.int32, device=draft_token_ids.device
    )
    prediction_indices = torch.zeros(
        batch_size, dtype=torch.long, device=draft_token_ids.device
    )
    active = torch.ones(
        batch_size, dtype=torch.bool, device=draft_token_ids.device
    )
    batch_indices = torch.arange(
        batch_size, dtype=torch.long, device=draft_token_ids.device
    )

    for depth in range(max_depth):
        prediction = predictions[batch_indices, prediction_indices]
        outputs[:, depth] = torch.where(
            active,
            prediction,
            torch.full_like(prediction, placeholder_token_id),
        )
        matches = (
            (parents == current_parent.unsqueeze(1))
            & (drafts == prediction.unsqueeze(1))
            & active.unsqueeze(1)
        )
        selected = torch.where(matches, node_ids, no_match).amin(dim=1)
        matched = selected < num_nodes
        accepted[:, depth] = torch.where(
            matched, selected, torch.full_like(selected, -1)
        )
        prediction_indices = torch.where(
            matched, selected.to(torch.long) + 1, prediction_indices
        )
        current_parent = torch.where(matched, selected, current_parent)
        active = matched

    final_prediction = predictions[batch_indices, prediction_indices]
    outputs[:, max_depth] = torch.where(
        active,
        final_prediction,
        torch.full_like(final_prediction, placeholder_token_id),
    )
    return outputs, accepted


def _move_tensor_slots(
    cache: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    packed_kv: bool,
) -> None:
    if cache.ndim < 2:
        raise ValueError(f"unsupported KV cache shape: {tuple(cache.shape)}")
    if packed_kv:
        if cache.shape[0] != 2 or cache.ndim < 3:
            raise ValueError(f"invalid packed K/V cache shape: {tuple(cache.shape)}")
        for plane in cache.unbind(0):
            flat = plane.flatten(0, 1)
            source_values = torch.index_select(flat, 0, source_slots)
            if flat.device.type == "npu":
                torch_npu.npu_scatter_nd_update_(
                    flat,
                    destination_slots.to(torch.int32).view(-1, 1),
                    source_values,
                )
            else:
                flat.index_copy_(0, destination_slots, source_values)
    else:
        flat = cache.flatten(0, 1)
        source_values = torch.index_select(flat, 0, source_slots)
        if flat.device.type == "npu":
            torch_npu.npu_scatter_nd_update_(
                flat,
                destination_slots.to(torch.int32).view(-1, 1),
                source_values,
            )
        else:
            flat.index_copy_(0, destination_slots, source_values)


def move_kv_cache_slots(
    kv_caches: Iterable[torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]],
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
) -> None:
    """Move token slots in every layer, safely handling overlapping paths."""
    if source_slots.shape != destination_slots.shape:
        raise ValueError("source and destination slot tensors must have equal shapes")
    if source_slots.numel() == 0:
        return
    source_slots = source_slots.to(dtype=torch.long)
    destination_slots = destination_slots.to(dtype=torch.long)
    for layer_cache in kv_caches:
        if isinstance(layer_cache, torch.Tensor):
            _move_tensor_slots(
                layer_cache, source_slots, destination_slots, packed_kv=True
            )
        else:
            for cache in layer_cache:
                _move_tensor_slots(
                    cache, source_slots, destination_slots, packed_kv=False
                )

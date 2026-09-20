# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical logical snapshots of hybrid attention and recurrent state.

The serializer is intentionally independent of StateAxis.  It consumes the
cache views and block tables already owned by the Ascend model runner and
removes allocator identity from the result.  This module has no ``torch_npu``
dependency so its ordering and rejection rules can be tested on CPU.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

LayerKind = Literal["attention", "recurrent"]


@dataclass(frozen=True)
class HybridStateLayerSource:
    """One local model layer and the live tensors that hold its state."""

    name: str
    ordinal: int
    kind: LayerKind
    group_index: int
    block_size: int
    tensors: tuple[torch.Tensor, ...]
    # Recurrent caches can keep multiple temporal/speculative states.  The
    # caller resolves the live logical state to one block-table column.
    state_block_position: int = 0


@dataclass(frozen=True)
class CanonicalHybridStateSnapshot:
    """Allocator-independent state bytes for one logical request."""

    recurrent: bytes
    kv: bytes
    recurrent_layers: tuple[str, ...]
    attention_layers: tuple[str, ...]


@dataclass(frozen=True)
class ArmedHybridStateSnapshot:
    """One explicitly requested, default-off diagnostic capture."""

    request_id: str
    checkpoint_id: str
    expected_context_tokens: int
    output_directory: Path
    rank: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Copy a contiguous tensor as raw bytes, preserving its element encoding."""

    contiguous = tensor.detach().contiguous()
    return contiguous.view(torch.uint8).cpu().numpy().tobytes()


def _validate_sources(sources: Sequence[HybridStateLayerSource]) -> None:
    _require(bool(sources), "hybrid snapshot has no layer sources")
    names: set[str] = set()
    ordinals: set[int] = set()
    kinds: set[LayerKind] = set()
    for source in sources:
        _require(bool(source.name), "hybrid snapshot layer name is empty")
        _require(source.name not in names, "hybrid snapshot layer name is duplicated")
        _require(source.ordinal >= 0, "hybrid snapshot layer ordinal is invalid")
        _require(
            source.ordinal not in ordinals,
            "hybrid snapshot layer ordinal is duplicated",
        )
        _require(source.kind in ("attention", "recurrent"), "hybrid snapshot layer kind is invalid")
        _require(source.group_index >= 0, "hybrid snapshot group index is invalid")
        _require(source.block_size > 0, "hybrid snapshot block size is invalid")
        _require(bool(source.tensors), "hybrid snapshot layer has no tensors")
        names.add(source.name)
        ordinals.add(source.ordinal)
        kinds.add(source.kind)
    _require(kinds == {"attention", "recurrent"}, "snapshot is not hybrid")


def capture_canonical_hybrid_state(
    sources: Sequence[HybridStateLayerSource],
    block_tables: Mapping[int, Sequence[int]],
    context_tokens: int,
) -> CanonicalHybridStateSnapshot:
    """Capture one request in canonical logical-state order.

    ``block_tables`` contains only the active physical block ids for the
    request, keyed by KV-cache group.  Recurrent layers emit their component
    tensors in the supplied order (GDN uses convolution state then recurrent
    matrix).  Attention layers emit K before V and visit blocks in logical
    token order, excluding an unused final-block tail.
    """

    _validate_sources(sources)
    _require(context_tokens > 0, "hybrid snapshot context length is invalid")

    recurrent = bytearray()
    kv = bytearray()
    recurrent_layers: list[str] = []
    attention_layers: list[str] = []
    for source in sorted(sources, key=lambda item: item.ordinal):
        active_blocks = block_tables.get(source.group_index)
        _require(active_blocks is not None, "hybrid snapshot block table is missing")
        _require(bool(active_blocks), "hybrid snapshot block table is empty")
        _require(
            all(isinstance(block_id, int) and block_id >= 0 for block_id in active_blocks),
            "hybrid snapshot block id is invalid",
        )

        if source.kind == "recurrent":
            recurrent_layers.append(source.name)
            _require(
                len(source.tensors) == 2,
                "canonical recurrent layer must contain conv and recurrent tensors",
            )
            _require(
                0 <= source.state_block_position < len(active_blocks),
                "canonical recurrent state block position is invalid",
            )
            block_id = active_blocks[source.state_block_position]
            for tensor in source.tensors:
                _require(tensor.ndim >= 2, "canonical recurrent tensor rank is invalid")
                _require(block_id < tensor.shape[0], "canonical recurrent block id is out of range")
                recurrent.extend(_tensor_bytes(tensor[block_id]))
            continue

        _require(
            len(source.tensors) == 2,
            "canonical attention layer must contain K and V tensors",
        )
        attention_layers.append(source.name)
        required_blocks = (context_tokens + source.block_size - 1) // source.block_size
        _require(
            len(active_blocks) == required_blocks,
            "canonical attention block extent differs from context",
        )
        for tensor in source.tensors:  # K before V.
            _require(tensor.ndim >= 3, "canonical attention tensor rank is invalid")
            _require(
                tensor.shape[1] == source.block_size,
                "canonical attention tensor block size differs",
            )
            remaining = context_tokens
            for block_id in active_blocks:
                _require(block_id < tensor.shape[0], "canonical attention block id is out of range")
                valid_tokens = min(remaining, source.block_size)
                kv.extend(_tensor_bytes(tensor[block_id, :valid_tokens]))
                remaining -= valid_tokens
            _require(remaining == 0, "canonical attention snapshot is incomplete")

    _require(bool(recurrent), "canonical recurrent snapshot is empty")
    _require(bool(kv), "canonical KV snapshot is empty")
    return CanonicalHybridStateSnapshot(
        recurrent=bytes(recurrent),
        kv=bytes(kv),
        recurrent_layers=tuple(recurrent_layers),
        attention_layers=tuple(attention_layers),
    )


def validate_capture_request(
    request_id: str,
    checkpoint_id: str,
    expected_context_tokens: int,
    output_directory: str,
    rank: int,
) -> ArmedHybridStateSnapshot:
    """Validate an explicit capture request without creating any files."""

    _require(bool(request_id), "hybrid snapshot request id is empty")
    _require(
        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", checkpoint_id)),
        "hybrid snapshot checkpoint id is invalid",
    )
    _require(expected_context_tokens > 0, "hybrid snapshot expected context is invalid")
    _require(rank >= 0, "hybrid snapshot rank is invalid")
    directory = Path(output_directory)
    _require(directory.is_absolute(), "hybrid snapshot directory must be absolute")
    directory = directory.resolve(strict=True)
    data_root = Path("/data")
    _require(
        directory == data_root or data_root in directory.parents,
        "hybrid snapshot directory must be under /data",
    )
    _require(directory.is_dir(), "hybrid snapshot output is not a directory")
    return ArmedHybridStateSnapshot(
        request_id=request_id,
        checkpoint_id=checkpoint_id,
        expected_context_tokens=expected_context_tokens,
        output_directory=directory,
        rank=rank,
    )


def write_rank_snapshot(
    request: ArmedHybridStateSnapshot,
    snapshot: CanonicalHybridStateSnapshot,
    logits: bytes,
    state_lease_generation: int,
) -> dict[str, object]:
    """Create one rank's immutable artifacts and return rehashable receipts."""

    _require(bool(logits) and len(logits) % 4 == 0, "hybrid snapshot logits are invalid")
    _require(state_lease_generation > 0, "state lease generation is invalid")
    payloads = {
        "logits": logits,
        "recurrent": snapshot.recurrent,
        "kv": snapshot.kv,
    }
    _require(all(payloads.values()), "hybrid snapshot artifact is empty")
    receipts: dict[str, dict[str, object]] = {}
    created: list[Path] = []
    try:
        for name, payload in payloads.items():
            suffix = "f32le" if name == "logits" else "bin"
            filename = f"{request.checkpoint_id}.rank{request.rank}.{name}.{suffix}"
            path = request.output_directory / filename
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            created.append(path)
            receipts[name] = {
                "file": filename,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    except BaseException:
        # A partial checkpoint cannot be mistaken for evidence.  Files created
        # by this invocation are safe to remove because exclusive creation
        # proves they did not predate the request.
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {
        "schema_version": 2,
        "request_id": request.request_id,
        "state_lease_generation": state_lease_generation,
        "checkpoint_id": request.checkpoint_id,
        "rank": request.rank,
        "context_tokens": request.expected_context_tokens,
        "layer_order": {
            "recurrent": list(snapshot.recurrent_layers),
            "attention": list(snapshot.attention_layers),
        },
        "artifacts": receipts,
    }

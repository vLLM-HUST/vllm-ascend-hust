# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).parents[3]
    / "vllm_ascend"
    / "worker"
    / "hybrid_state_snapshot.py"
)
SPEC = importlib.util.spec_from_file_location("hybrid_state_snapshot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
snapshot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = snapshot
SPEC.loader.exec_module(snapshot)


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def test_canonical_snapshot_removes_physical_block_identity_and_padding() -> None:
    conv = torch.arange(4 * 2 * 3, dtype=torch.int16).reshape(4, 2, 3)
    recurrent = (100 + torch.arange(4 * 2 * 2, dtype=torch.int16)).reshape(4, 2, 2)
    key = (200 + torch.arange(5 * 4 * 2, dtype=torch.int16)).reshape(5, 4, 2)
    value = (400 + torch.arange(5 * 4 * 2, dtype=torch.int16)).reshape(5, 4, 2)

    sources = [
        snapshot.HybridStateLayerSource(
            name="layers.0.linear_attn",
            ordinal=0,
            kind="recurrent",
            group_index=1,
            block_size=64,
            tensors=(conv, recurrent),
        ),
        snapshot.HybridStateLayerSource(
            name="layers.1.self_attn",
            ordinal=1,
            kind="attention",
            group_index=0,
            block_size=4,
            tensors=(key, value),
        ),
    ]
    result = snapshot.capture_canonical_hybrid_state(
        sources,
        block_tables={0: [3, 1], 1: [2]},
        context_tokens=6,
    )

    assert result.recurrent == _bytes(conv[2]) + _bytes(recurrent[2])
    assert result.recurrent_layers == ("layers.0.linear_attn",)
    assert result.attention_layers == ("layers.1.self_attn",)
    assert result.kv == b"".join(
        (
            _bytes(key[3]),
            _bytes(key[1, :2]),
            _bytes(value[3]),
            _bytes(value[1, :2]),
        )
    )
    assert _bytes(key[1, 2:]) not in result.kv


def test_layer_ordinal_defines_canonical_order() -> None:
    conv0 = torch.tensor([[[10]]], dtype=torch.int16)
    recurrent0 = torch.tensor([[[11]]], dtype=torch.int16)
    conv1 = torch.tensor([[[20]]], dtype=torch.int16)
    recurrent1 = torch.tensor([[[21]]], dtype=torch.int16)
    key = torch.tensor([[[30]]], dtype=torch.int16)
    value = torch.tensor([[[31]]], dtype=torch.int16)
    sources = [
        snapshot.HybridStateLayerSource("l2.attn", 2, "attention", 0, 1, (key, value)),
        snapshot.HybridStateLayerSource("l1.gdn", 1, "recurrent", 1, 1, (conv1, recurrent1)),
        snapshot.HybridStateLayerSource("l0.gdn", 0, "recurrent", 1, 1, (conv0, recurrent0)),
    ]

    result = snapshot.capture_canonical_hybrid_state(
        sources, {0: [0], 1: [0]}, context_tokens=1
    )

    assert result.recurrent == b"".join(
        (_bytes(conv0[0]), _bytes(recurrent0[0]), _bytes(conv1[0]), _bytes(recurrent1[0]))
    )
    assert result.kv == _bytes(key[0]) + _bytes(value[0])


@pytest.mark.parametrize(
    ("block_tables", "context_tokens", "message"),
    [
        ({0: [0, 1], 1: [0]}, 1, "block extent"),
        ({0: [4], 1: [0]}, 1, "out of range"),
        ({0: [0], 1: []}, 1, "block table is empty"),
        ({0: [0], 1: [0]}, 0, "context length"),
    ],
)
def test_snapshot_rejects_ambiguous_or_stale_state(
    block_tables: dict[int, list[int]], context_tokens: int, message: str
) -> None:
    key = torch.zeros((2, 1, 1), dtype=torch.int16)
    value = torch.zeros_like(key)
    conv = torch.zeros((2, 1, 1), dtype=torch.int16)
    recurrent = torch.zeros_like(conv)
    sources = [
        snapshot.HybridStateLayerSource("gdn", 0, "recurrent", 1, 1, (conv, recurrent)),
        snapshot.HybridStateLayerSource("attn", 1, "attention", 0, 1, (key, value)),
    ]

    with pytest.raises(ValueError, match=message):
        snapshot.capture_canonical_hybrid_state(sources, block_tables, context_tokens)


def test_rank_writer_is_create_exclusive_and_returns_rehashable_receipts(tmp_path: Path) -> None:
    request = snapshot.ArmedHybridStateSnapshot("request", "checkpoint", 3, tmp_path, 2)
    state = snapshot.CanonicalHybridStateSnapshot(
        recurrent=b"recurrent",
        kv=b"kv-state",
        recurrent_layers=("layers.0.linear_attn",),
        attention_layers=("layers.1.self_attn",),
    )
    receipt = snapshot.write_rank_snapshot(request, state, b"\x00" * 8)

    assert receipt["layer_order"] == {
        "recurrent": ["layers.0.linear_attn"],
        "attention": ["layers.1.self_attn"],
    }
    for artifact in receipt["artifacts"].values():
        payload = (tmp_path / artifact["file"]).read_bytes()
        assert len(payload) == artifact["bytes"]
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]

    with pytest.raises(FileExistsError):
        snapshot.write_rank_snapshot(request, state, b"\x00" * 8)


def test_rank_writer_removes_only_new_partial_files(tmp_path: Path) -> None:
    request = snapshot.ArmedHybridStateSnapshot("request", "partial", 1, tmp_path, 0)
    recurrent_path = tmp_path / "partial.rank0.recurrent.bin"
    recurrent_path.write_bytes(b"preexisting")
    state = snapshot.CanonicalHybridStateSnapshot(
        recurrent=b"new-recurrent",
        kv=b"new-kv",
        recurrent_layers=("gdn",),
        attention_layers=("attn",),
    )

    with pytest.raises(FileExistsError):
        snapshot.write_rank_snapshot(request, state, b"\x00" * 4)

    assert recurrent_path.read_bytes() == b"preexisting"
    assert not (tmp_path / "partial.rank0.logits.f32le").exists()
    assert not (tmp_path / "partial.rank0.kv.bin").exists()

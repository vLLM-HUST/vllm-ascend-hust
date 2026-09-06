# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

LAYER_NAMES = tuple(f"model.layers.{index}.self_attn.attn" for index in range(32))


class FakeProvider:
    def __init__(self, *, compress: bool = True) -> None:
        self.compress = compress
        self.cleaned: list[str] = []
        self.commits: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
        self.build_kwargs = None
        self.finished = None

    def cleanup_request(self, request_id: str) -> None:
        self.cleaned.append(request_id)

    def mark_committed(self, request_id: str, block_ids: tuple[tuple[int, ...], ...]) -> None:
        self.commits.append((request_id, block_ids))

    def build_attention_batch_view(self, **kwargs):
        self.build_kwargs = kwargs
        return SimpleNamespace(requests=(SimpleNamespace(compress=self.compress),))

    def finish_model_forward(self, view, **kwargs):
        self.finished = (view, kwargs)
        return ["plan"]

    @staticmethod
    def get_layer_indices(layer_names: tuple[str, ...]) -> dict[str, int]:
        return {name: int(name.split(".layers.", 1)[1].split(".", 1)[0]) for name in layer_names}


class FakeBlockTable:
    def __init__(self) -> None:
        self.rows = []

    def add_row(self, block_ids, row_index: int) -> None:
        self.rows.append((block_ids, row_index))


def _runner(provider: FakeProvider) -> NPUModelRunner:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_compression_provider = provider
    runner._kv_cache_compression_step_view = None
    runner._kv_cache_compression_plans = None
    runner._kv_cache_compression_destination_block_ids = None
    runner._kv_cache_compression_logged_full_sizes = set()
    runner.requests = {
        "request": SimpleNamespace(block_ids=([1, 2, 3],)),
    }
    runner.input_batch = SimpleNamespace(
        req_ids=["request"],
        req_id_to_index={"request": 0},
        block_table=FakeBlockTable(),
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
        num_prompt_tokens=np.array([20], dtype=np.int32),
    )
    runner.optimistic_seq_lens_cpu = torch.tensor([20], dtype=torch.int32)
    group = SimpleNamespace(
        layer_names=LAYER_NAMES,
        kv_cache_spec=SimpleNamespace(block_size=128),
    )
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[group])
    runner.vllm_config = SimpleNamespace(kv_cache_compression_config=SimpleNamespace(schema_version=1))
    return runner


def test_disabled_runner_does_not_build_provider_view() -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_compression_provider = None

    assert (
        runner._build_kv_cache_compression_view(
            num_reqs=1,
            num_scheduled_tokens_np=np.array([1], dtype=np.int32),
        )
        is None
    )


def test_commit_ack_updates_request_block_table_and_provider() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    output = SimpleNamespace(kv_cache_compression_block_table_updates={"request": ([1, 2],)})

    runner._apply_kv_cache_compression_block_table_updates(output)

    assert runner.requests["request"].block_ids == ([1, 2],)
    assert runner.input_batch.block_table.rows == [(([1, 2],), 0)]
    assert provider.commits == [("request", ((1, 2),))]


def test_unknown_commit_ack_is_rejected_before_provider_mutation() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    output = SimpleNamespace(kv_cache_compression_block_table_updates={"unknown": ([1],)})

    with pytest.raises(RuntimeError, match="unknown request"):
        runner._apply_kv_cache_compression_block_table_updates(output)

    assert provider.commits == []
    assert runner.input_batch.block_table.rows == []


def test_finished_preempted_and_resumed_states_are_cleaned() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    output = SimpleNamespace(
        finished_req_ids={"finished"},
        preempted_req_ids={"preempted"},
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids={"resumed"}),
    )

    runner._cleanup_kv_cache_compression_states(output)

    assert set(provider.cleaned) == {"finished", "preempted", "resumed"}


def test_runner_builds_provider_view_from_current_semantic_state() -> None:
    provider = FakeProvider()
    runner = _runner(provider)

    view = runner._build_kv_cache_compression_view(
        num_reqs=1,
        num_scheduled_tokens_np=np.array([20], dtype=np.int32),
    )

    assert view is runner._kv_cache_compression_step_view
    assert provider.build_kwargs == {
        "request_ids": ("request",),
        "query_lengths": (20,),
        "semantic_num_tokens": (20,),
        "num_computed_tokens": (0,),
        "num_prompt_tokens": (20,),
        "block_ids": (((1, 2, 3),),),
        "layer_names": LAYER_NAMES,
        "block_size": 128,
        "destination_block_ids": None,
    }


def test_uncompressed_batch_does_not_create_transaction_state() -> None:
    provider = FakeProvider(compress=False)
    runner = _runner(provider)

    view = runner._build_kv_cache_compression_view(
        num_reqs=1,
        num_scheduled_tokens_np=np.array([20], dtype=np.int32),
    )

    assert view is None
    assert runner._kv_cache_compression_step_view is None


def test_successful_forward_exports_plans_and_clears_step_state() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    view = SimpleNamespace(requests=())
    runner._kv_cache_compression_step_view = view
    runner._kv_cache_compression_destination_block_ids = {"request": ((7, 8),)}

    runner._finish_kv_cache_compression_forward()

    assert provider.finished == (
        view,
        {"layer_names": LAYER_NAMES, "schema_version": 1},
    )
    assert runner._take_kv_cache_compression_plans() == ["plan"]
    assert runner._kv_cache_compression_step_view is None
    assert runner._kv_cache_compression_destination_block_ids is None


def test_full_decode_metadata_buffers_keep_stable_addresses() -> None:
    provider = FakeProvider()
    runner = _runner(provider)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    runner.max_num_reqs = 4
    runner.pin_memory = False
    runner.device = torch.device("cpu")
    runner.activate_kv_cache_compression_provider(provider)
    slots = runner._kv_cache_compression_full_slots
    lengths = runner._kv_cache_compression_full_lengths
    assert slots is not None
    assert lengths is not None
    slot_ptr = slots.untyped_storage().data_ptr()
    length_ptr = lengths.untyped_storage().data_ptr()

    class FakeFullView:
        def fill_full_decode_metadata(
            self,
            *,
            layer_names,
            slot_staging,
            length_staging,
            num_reqs_padded,
        ) -> None:
            slot_staging.fill_(-1)
            length_staging.fill_(1)
            slot_staging[:, :2].fill_(7)
            length_staging[:, :2].fill_(8)

    runner._prepare_full_kv_cache_compression_metadata(view=FakeFullView(), num_reqs=2, num_reqs_padded=4)

    assert slots.untyped_storage().data_ptr() == slot_ptr
    assert lengths.untyped_storage().data_ptr() == length_ptr
    assert torch.equal(slots[0], torch.tensor([7, 7, -1, -1]))
    assert torch.equal(lengths[0], torch.tensor([8, 8, 1, 1]))

#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import torch

from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    filter_chunked_req_indices,
    get_or_register_attention_buffer,
)


def test_get_or_register_attention_buffer() -> None:
    module_a = torch.nn.Module()
    module_b = torch.nn.Module()
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "layer.a": module_a,
                "layer.b": module_b,
            }
        )
    )
    factory_call_count = 0

    def factory() -> torch.Tensor:
        nonlocal factory_call_count
        factory_call_count += 1
        return torch.tensor([1, 2, 3])

    buffer = get_or_register_attention_buffer(
        vllm_config,
        ["layer.a", "layer.b"],
        "_test_buffer",
        factory,
    )

    assert factory_call_count == 1
    assert module_a._buffers["_test_buffer"] is buffer
    assert module_b._buffers["_test_buffer"] is buffer
    assert "_test_buffer" not in module_a.state_dict()
    assert "_test_buffer" not in module_b.state_dict()


def test_filter_chunked_req_indices_empty_mask() -> None:
    indices = filter_chunked_req_indices(
        torch.tensor([2, 1, 3]),
        [False, False, False],
    )

    torch.testing.assert_close(indices, torch.empty(0, dtype=torch.long))


def test_filter_chunked_req_indices_mixed_mask() -> None:
    indices = filter_chunked_req_indices(
        torch.tensor([2, 1, 3]),
        [True, False, True],
    )

    torch.testing.assert_close(indices, torch.tensor([0, 1, 3, 4, 5]))


def test_common_metadata_keeps_removed_vllm_compatibility_slots() -> None:
    legacy_seq_lens = torch.tensor([1], dtype=torch.int32)
    legacy_num_computed = torch.tensor([0], dtype=torch.int32)
    legacy_dcp_seq_lens = torch.tensor([1], dtype=torch.int32)

    metadata = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=1,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        _seq_lens_cpu=legacy_seq_lens,
        _num_computed_tokens_cpu=legacy_num_computed,
        dcp_local_seq_lens_cpu=legacy_dcp_seq_lens,
    )

    assert metadata._seq_lens_cpu is legacy_seq_lens
    assert metadata._num_computed_tokens_cpu is legacy_num_computed
    assert metadata.dcp_local_seq_lens_cpu is legacy_dcp_seq_lens

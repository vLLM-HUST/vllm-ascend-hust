# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch

from vllm_ascend.worker.v2.sample.apply_top_k_top_p import apply_top_k_top_p_npu


def test_apply_top_k_top_p_accepts_current_host_row_hint() -> None:
    logits = torch.tensor([[1.0, 2.0]])
    k = torch.tensor([1])
    p = torch.tensor([0.9])

    with patch(
        "vllm_ascend.worker.v2.sample.apply_top_k_top_p._apply_top_k_top_p_pytorch",
        return_value=logits,
    ) as apply_pytorch:
        result = apply_top_k_top_p_npu(
            logits,
            k,
            p,
            has_topp_only_rows=False,
        )

    assert result is logits
    apply_pytorch.assert_called_once_with(logits, k, p)

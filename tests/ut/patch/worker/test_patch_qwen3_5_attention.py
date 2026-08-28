from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.patch.worker import patch_qwen3_5


def test_qwen35_attention_uses_canonical_path_without_triton_fusion():
    qkv = torch.randn(3, 16)
    q = torch.randn(3, 8)
    k = torch.randn(3, 4)
    v = torch.randn(3, 4)
    gate = torch.randn(3, 8)
    attention_output = torch.randn(3, 8)
    projected = torch.randn(3, 16)
    positions = torch.arange(3)
    hidden_states = torch.randn(3, 16)
    output = torch.empty_like(projected)

    attention = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_text"),
        qkv_proj=MagicMock(return_value=(qkv, None)),
        _project_qkv_gate=MagicMock(return_value=(q, k, v, gate)),
        attn=MagicMock(return_value=attention_output),
        attn_output_gate=True,
        o_proj=MagicMock(return_value=(projected, None)),
    )

    with patch.object(patch_qwen3_5, "_HAS_TRITON_MROPE_FUSION", False):
        patch_qwen3_5.AscendQwen3NextAttention.forward(
            attention,
            positions,
            output,
            hidden_states,
        )

    attention.qkv_proj.assert_called_once_with(hidden_states)
    attention._project_qkv_gate.assert_called_once_with(qkv, positions)
    attention.attn.assert_called_once_with(q, k, v)
    expected_attention_output = attention_output * torch.sigmoid(gate)
    assert torch.allclose(attention.o_proj.call_args.args[0], expected_attention_output)
    assert torch.equal(output, projected)

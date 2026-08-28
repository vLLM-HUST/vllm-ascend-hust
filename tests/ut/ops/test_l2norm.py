from unittest.mock import patch

import torch
from vllm.triton_utils import triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.l2norm import l2norm_npu


def test_worker_initializes_ascend_triton_math_compatibility():
    assert triton.next_power_of_2 is next_power_of_2


def test_l2norm_npu_uses_equivalent_rmsnorm_parameters():
    x = torch.randn(2, 3, 8, dtype=torch.float32)
    expected = x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + 1e-6)

    def fake_rms_norm(value, weight, epsilon):
        dim = value.shape[-1]
        assert torch.allclose(weight, torch.full_like(weight, dim**-0.5))
        assert epsilon == 1e-6 / dim
        output = value / torch.sqrt(torch.mean(value * value, dim=-1, keepdim=True) + epsilon) * weight
        return output, torch.empty(0)

    with patch("torch_npu.npu_rms_norm", side_effect=fake_rms_norm) as rms_norm:
        actual = l2norm_npu(x)

    rms_norm.assert_called_once()
    torch.testing.assert_close(actual, expected)


def test_l2norm_npu_preserves_shape_and_dtype():
    x = torch.randn(7, 4, 16, dtype=torch.bfloat16)

    def fake_rms_norm(value, weight, epsilon):
        del epsilon
        return value * weight, torch.empty(0)

    with patch("torch_npu.npu_rms_norm", side_effect=fake_rms_norm):
        actual = l2norm_npu(x)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype

# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.spec_decode.pearl.mc2 import (
    detect_mc2_capability,
    matmul_allreduce_add_rmsnorm_or_fallback,
)


def test_mc2_capability_is_explicit_on_cpu():
    capability = detect_mc2_capability("cpu", tp_size=3)
    assert not capability.available
    assert "Ascend" in capability.reason


def test_mc2_fallback_matches_shapes_and_is_differentiable():
    x = torch.randn(2, 4, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    residual = torch.randn(2, 4, 5, requires_grad=True)
    gamma = torch.ones(5, requires_grad=True)
    output, added = matmul_allreduce_add_rmsnorm_or_fallback(
        x, weight, residual, gamma, tp_rank_size=1, use_fused=False
    )
    assert output.shape == residual.shape
    assert added.shape == residual.shape
    output.square().mean().backward()
    assert x.grad is not None

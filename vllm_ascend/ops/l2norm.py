# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import math

import torch
import torch_npu
from vllm.model_executor.layers.fla.ops.utils import tensor_cache


@tensor_cache
def _l2norm_unit_weight(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return the cached RMSNorm weight that makes RMSNorm equal L2 norm."""
    return torch.full((dim,), 1.0 / math.sqrt(dim), dtype=dtype, device=device)


def l2norm_npu(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize the last dimension with the native NPU RMSNorm kernel.

    RMSNorm computes ``x / sqrt(mean(x**2) + eps_rms) * weight``.  With
    ``weight=1/sqrt(dim)`` and ``eps_rms=eps/dim`` this is exactly
    ``x / sqrt(sum(x**2) + eps)``, matching the FLA L2-normalization
    definition without depending on a Triton runtime.
    """
    original_shape = x.shape
    dim = x.shape[-1]
    x_2d = x.reshape(-1, dim).contiguous()
    weight = _l2norm_unit_weight(dim, x.dtype, x.device)
    y, _ = torch_npu.npu_rms_norm(x_2d, weight, eps / dim)
    return y.reshape(original_shape)

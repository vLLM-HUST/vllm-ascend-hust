#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Fused MoE shared-expert epilogue multiply (default-off).

Replaces the broadcast multiply ``sigmoid(gate) * shared_out`` with a single
kernel. The host ``sigmoid`` is left untouched so the numeric result stays
bit-identical; only the multiply is fused.
"""

import os

import torch
from vllm.triton_utils import tl, triton

ENV_MOE_EPILOGUE = "VLLM_HUST_MOE_EPILOGUE"
ENV_MOE_EPILOGUE_MAX_ROWS = "VLLM_HUST_MOE_EPILOGUE_MAXROWS"

# Prefill rows (hundreds to thousands) run faster on the aclnn multiply; only
# the decode buckets are fused.
DEFAULT_MAX_ROWS = 32


def is_enabled() -> bool:
    return os.environ.get(ENV_MOE_EPILOGUE, "0") == "1"


def max_rows() -> int:
    try:
        return int(os.environ.get(ENV_MOE_EPILOGUE_MAX_ROWS, DEFAULT_MAX_ROWS))
    except ValueError:
        return DEFAULT_MAX_ROWS


@triton.jit
def _gate_mul_kernel(
    gate_ptr,
    src_ptr,
    dst_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``dst[row, :] = gate[row, 0] * src[row, :]`` in fp32 then rounded to bf16."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    scale = tl.load(gate_ptr + row)
    value = tl.load(src_ptr + row * D + cols, mask=mask)
    product = (scale.to(tl.float32) * value.to(tl.float32)).to(tl.bfloat16)
    tl.store(dst_ptr + row * D + cols, product, mask=mask)


def _fusable(gate: torch.Tensor, out: torch.Tensor) -> bool:
    if out.dim() != 2 or not out.is_contiguous():
        return False
    if gate.dim() != 2 or gate.shape[1] != 1:
        return False
    if gate.shape[0] != out.shape[0]:
        return False
    if out.dtype is not torch.bfloat16 or gate.dtype is not torch.bfloat16:
        return False
    return 0 < out.shape[0] <= max_rows()


def gate_mul(gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Broadcast multiply with a run-time shape decision.

    The decision has to stay inside this function: ``AscendSharedExperts.part2``
    runs under dynamo tracing, where a python branch on tensor shape folds on
    whichever shape is traced first.
    """
    if not is_enabled() or not _fusable(gate, out):
        return gate * out

    rows, width = out.shape
    result = torch.empty_like(out)
    _gate_mul_kernel[(rows,)](
        gate,
        out,
        result,
        D=width,
        BLOCK=triton.next_power_of_2(width),
    )
    return result

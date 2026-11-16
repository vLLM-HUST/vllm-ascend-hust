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
"""Default-off host kernel for GDN TP split_ba."""

import os

import torch
from vllm.triton_utils import tl, triton

ENV_SPLIT_BA = "VLLM_HUST_GDN_SPLIT_BA"


def is_split_ba_enabled() -> bool:
    return os.environ.get(ENV_SPLIT_BA, "0") == "1"


@triton.jit(do_not_specialize=["M", "stride_bam", "stride_bm", "stride_am"])
def gdn_split_ba_2d_kernel(
    ba_ptr,
    b_ptr,
    a_ptr,
    stride_bam,
    stride_ban,
    stride_bm,
    stride_bn,
    stride_am,
    stride_an,
    M,
    HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = rows < M
    b_in = ba_ptr + rows[:, None] * stride_bam + cols[None, :] * stride_ban
    a_in = ba_ptr + rows[:, None] * stride_bam + (cols[None, :] + HALF) * stride_ban
    b_out = b_ptr + rows[:, None] * stride_bm + cols[None, :] * stride_bn
    a_out = a_ptr + rows[:, None] * stride_am + cols[None, :] * stride_an
    tl.store(b_out, tl.load(b_in, mask=mask[:, None], other=0.0), mask=mask[:, None])
    tl.store(a_out, tl.load(a_in, mask=mask[:, None], other=0.0), mask=mask[:, None])


def gdn_split_ba(ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not is_split_ba_enabled():
        b, a = ba.chunk(2, dim=-1)
        return b.contiguous(), a.contiguous()
    rows, total = ba.shape
    half = total // 2
    b = torch.empty((rows, half), dtype=ba.dtype, device=ba.device)
    a = torch.empty((rows, half), dtype=ba.dtype, device=ba.device)
    gdn_split_ba_2d_kernel[(triton.cdiv(rows, 32),)](
        ba,
        b,
        a,
        ba.stride(0),
        ba.stride(1),
        b.stride(0),
        b.stride(1),
        a.stride(0),
        a.stride(1),
        rows,
        HALF=half,
        BLOCK_M=32,
        BLOCK_N=half,
    )
    return b, a

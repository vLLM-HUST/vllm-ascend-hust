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
"""Fused GDN projection unpack and mixed-qkv rearrange (default-off).

Two copies in the GDN non-interleaved path are fused:

1. ``vllm_ascend/ops/gdn.py`` projection branch
   ``split + reshape + chunk + 2x contiguous`` -> ``gdn_unpack``
   (``z`` stays outside the kernel: reshaping a contiguous slice is a view)
2. ``rearrange_mixed_qkv``
   ``split + 3x reshape + cat`` -> ``gdn_rearrange``

Both kernels use the same launch shape as the upstream
``fused_qkvzba_split_reshape_cat``: ``grid = min(vector_core, rows)`` with a
UB-aware inner loop, so prefill rows are not spread over thousands of
one-row programs. Every entry point falls back to the stock chain instead of
raising, so an unexpected layout never breaks the model.
"""

import os

import torch
from vllm.triton_utils import tl, triton

ENV_GDN_UNPACK_FUSED = "VLLM_HUST_GDN_UNPACK_FUSED"
ENV_GDN_OUTPUT_NORM_FUSED = "VLLM_HUST_GDN_OUTPUT_NORM_FUSED"

# A single row loses to the stock chain (one kernel launch costs more than two
# tiny copies), so the fused path starts at two rows.
DEFAULT_MIN_ROWS = 2

# Rows beyond this stay on the stock chain even though the blocked grid could
# handle them; keeps the blast radius small until prefill is measured.
DEFAULT_MAX_ROWS = 4096

_UB_KB = 85


def is_enabled() -> bool:
    return os.environ.get(ENV_GDN_UNPACK_FUSED, "0") == "1"


def is_output_norm_enabled() -> bool:
    return os.environ.get(ENV_GDN_OUTPUT_NORM_FUSED, "0") == "1"


def _max_rows() -> int:
    try:
        return int(os.environ.get("VLLM_HUST_GDN_UNPACK_MAXROWS", DEFAULT_MAX_ROWS))
    except ValueError:
        return DEFAULT_MAX_ROWS


def _vector_core_num() -> int:
    """AIV core count; 48 is the 910B2 value used as fallback."""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.get_device_properties(torch.npu.current_device()).vector_core_num
    except Exception:
        return 48


def _rows_per_iter(elements_per_row: int, elem_size: int) -> int:
    """Rows that fit the UB per inner iteration (read + write counted)."""
    per_row = max(1, elements_per_row * elem_size)
    return max(1, min(64, (_UB_KB * 1024) // per_row))


@triton.jit
def _unpack_kernel(
    qkvz,
    ba,
    mixed,
    b,
    a,
    total_rows,
    rows_per_vec,
    QKV: tl.constexpr,
    QKVZ: tl.constexpr,
    HALF: tl.constexpr,
    BA: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
    BLOCK_QKV: tl.constexpr,
    BLOCK_BA: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * rows_per_vec
    end = min(start + rows_per_vec, total_rows)

    cols_qkv = tl.arange(0, BLOCK_QKV)
    mask_qkv = cols_qkv < QKV
    cols_ba = tl.arange(0, BLOCK_BA)
    mask_ba = cols_ba < HALF

    for r0 in tl.range(start, end, ROWS_PER_ITER):
        rows = r0 + tl.arange(0, ROWS_PER_ITER)
        row_mask = rows < end

        src = qkvz + rows[:, None] * QKVZ
        keep_qkv = row_mask[:, None] & mask_qkv[None, :]
        tl.store(
            mixed + rows[:, None] * QKV + cols_qkv[None, :],
            tl.load(src + cols_qkv[None, :], mask=keep_qkv),
            mask=keep_qkv,
        )

        src_ba = ba + rows[:, None] * BA
        keep_ba = row_mask[:, None] & mask_ba[None, :]
        tl.store(
            b + rows[:, None] * HALF + cols_ba[None, :],
            tl.load(src_ba + cols_ba[None, :], mask=keep_ba),
            mask=keep_ba,
        )
        tl.store(
            a + rows[:, None] * HALF + cols_ba[None, :],
            tl.load(src_ba + HALF + cols_ba[None, :], mask=keep_ba),
            mask=keep_ba,
        )


@triton.jit
def _rearrange_kernel(
    mixed_qkv,
    q,
    k,
    v,
    total_rows,
    rows_per_vec,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    QKV: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * rows_per_vec
    end = min(start + rows_per_vec, total_rows)

    cols_q = tl.arange(0, BLOCK_Q)
    mask_q = cols_q < Q_DIM
    cols_v = tl.arange(0, BLOCK_V)
    mask_v = cols_v < V_DIM

    for r0 in tl.range(start, end, ROWS_PER_ITER):
        rows = r0 + tl.arange(0, ROWS_PER_ITER)
        row_mask = rows < end
        src = mixed_qkv + rows[:, None] * QKV

        keep_q = row_mask[:, None] & mask_q[None, :]
        tl.store(
            q + rows[:, None] * Q_DIM + cols_q[None, :],
            tl.load(src + cols_q[None, :], mask=keep_q),
            mask=keep_q,
        )
        tl.store(
            k + rows[:, None] * K_DIM + cols_q[None, :],
            tl.load(src + Q_DIM + cols_q[None, :], mask=keep_q),
            mask=keep_q,
        )

        keep_v = row_mask[:, None] & mask_v[None, :]
        tl.store(
            v + rows[:, None] * V_DIM + cols_v[None, :],
            tl.load(src + Q_DIM + K_DIM + cols_v[None, :], mask=keep_v),
            mask=keep_v,
        )


def unpack_fusable(qkvz: torch.Tensor, ba: torch.Tensor) -> bool:
    if not is_enabled():
        return False
    if qkvz.dim() != 2 or ba.dim() != 2:
        return False
    if not qkvz.is_contiguous() or not ba.is_contiguous():
        return False
    if qkvz.dtype is not torch.bfloat16 or ba.dtype is not torch.bfloat16:
        return False
    if qkvz.shape[0] != ba.shape[0]:
        return False
    return DEFAULT_MIN_ROWS <= qkvz.shape[0] <= _max_rows()


def rearrange_fusable(mixed_qkv: torch.Tensor) -> bool:
    if not is_enabled():
        return False
    if mixed_qkv.dim() != 2 or not mixed_qkv.is_contiguous():
        return False
    if mixed_qkv.dtype is not torch.bfloat16:
        return False
    return DEFAULT_MIN_ROWS <= mixed_qkv.shape[0] <= _max_rows()


def _stock_unpack(
    qkvz: torch.Tensor, ba: torch.Tensor, qkv_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stock chain. Outputs are fresh tensors: a custom op may not alias inputs.

    ``clone`` rather than ``contiguous``: a one-row slice is reported as already
    contiguous (the leading dim has size 1, so its stride is ignored), and
    ``contiguous`` would hand back a view of the input.
    """
    mixed = qkvz[:, :qkv_size].clone()
    b, a = ba.chunk(2, dim=-1)
    return mixed, b.clone(), a.clone()


def _stock_rearrange(
    mixed_qkv: torch.Tensor, q_dim: int, k_dim: int, v_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stock chain. Every output is a fresh tensor: custom ops may not alias.

    The host chain builds one ``cat`` buffer and slices it, which would make the
    three outputs alias each other; separate ``contiguous`` copies avoid that and
    produce the same values.
    """
    query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    return query.clone(), key.clone(), value.clone()


@torch.library.custom_op("vllm_ascend::gdn_unpack", mutates_args=())
def gdn_unpack(
    qkvz: torch.Tensor, ba: torch.Tensor, qkv_size: int, half: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``qkvz[T, qkv+z]`` + ``ba[T, 2*half]`` -> contiguous ``mixed`` / ``b`` / ``a``.

    Shape and layout checks live here on purpose: the caller runs under dynamo
    tracing, where a python branch on tensor shape folds on the first traced
    shape and decode would never take the fused path.
    """
    if not unpack_fusable(qkvz, ba) or ba.shape[1] != 2 * half or qkvz.shape[1] <= qkv_size:
        return _stock_unpack(qkvz, ba, qkv_size)

    rows = qkvz.shape[0]
    elem = qkvz.element_size()
    mixed = torch.empty((rows, qkv_size), dtype=qkvz.dtype, device=qkvz.device)
    b = torch.empty((rows, half), dtype=ba.dtype, device=ba.device)
    a = torch.empty((rows, half), dtype=ba.dtype, device=ba.device)

    qkvz_width = qkvz.shape[1]
    grid = max(1, min(_vector_core_num(), rows))
    per_vec = triton.cdiv(rows, grid)
    per_iter = min(_rows_per_iter(2 * qkvz_width, elem), per_vec)

    _unpack_kernel[(grid,)](
        qkvz,
        ba,
        mixed,
        b,
        a,
        rows,
        per_vec,
        QKV=qkv_size,
        QKVZ=qkvz_width,
        HALF=half,
        BA=ba.shape[1],
        ROWS_PER_ITER=per_iter,
        BLOCK_QKV=triton.next_power_of_2(qkv_size),
        BLOCK_BA=triton.next_power_of_2(half),
    )
    return mixed, b, a


@torch.library.custom_op("vllm_ascend::gdn_rearrange", mutates_args=())
def gdn_rearrange(
    mixed_qkv: torch.Tensor, q_dim: int, k_dim: int, v_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mixed_qkv[T, q+k+v]`` -> contiguous ``q`` / ``k`` / ``v`` of ``[T, dim]``."""
    if not rearrange_fusable(mixed_qkv) or mixed_qkv.shape[1] != q_dim + k_dim + v_dim:
        return _stock_rearrange(mixed_qkv, q_dim, k_dim, v_dim)

    rows = mixed_qkv.shape[0]
    elem = mixed_qkv.element_size()
    width = mixed_qkv.shape[1]
    q = torch.empty((rows, q_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    k = torch.empty((rows, k_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    v = torch.empty((rows, v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)

    grid = max(1, min(_vector_core_num(), rows))
    per_vec = triton.cdiv(rows, grid)
    per_iter = min(_rows_per_iter(2 * width, elem), per_vec)

    _rearrange_kernel[(grid,)](
        mixed_qkv,
        q,
        k,
        v,
        rows,
        per_vec,
        Q_DIM=q_dim,
        K_DIM=k_dim,
        V_DIM=v_dim,
        QKV=width,
        ROWS_PER_ITER=per_iter,
        BLOCK_Q=triton.next_power_of_2(max(q_dim, k_dim)),
        BLOCK_V=triton.next_power_of_2(v_dim),
    )
    return q, k, v


@gdn_unpack.register_fake
def _gdn_unpack_fake(
    qkvz: torch.Tensor, ba: torch.Tensor, qkv_size: int, half: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = qkvz.shape[0]
    return (
        qkvz.new_empty((rows, qkv_size)),
        ba.new_empty((rows, half)),
        ba.new_empty((rows, half)),
    )


@gdn_rearrange.register_fake
def _gdn_rearrange_fake(
    mixed_qkv: torch.Tensor, q_dim: int, k_dim: int, v_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = mixed_qkv.shape[0]
    return (
        mixed_qkv.new_empty((rows, q_dim)),
        mixed_qkv.new_empty((rows, k_dim)),
        mixed_qkv.new_empty((rows, v_dim)),
    )


@triton.jit(do_not_specialize=["stride_xm", "stride_zm", "stride_outm", "M", "N", "eps"])
def _output_norm_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    out_ptr,
    stride_xm,
    stride_zm,
    stride_outm,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    row_mask = rows < M
    col_mask = cols < N
    mask_2d = row_mask[:, None] & col_mask[None, :]

    w = tl.load(w_ptr + cols, mask=col_mask).to(tl.float32)
    x_ptrs = x_ptr + rows[:, None] * stride_xm + cols[None, :]
    z_ptrs = z_ptr + rows[:, None] * stride_zm + cols[None, :]
    x = tl.load(x_ptrs, mask=mask_2d).to(tl.float32)
    z = tl.load(z_ptrs, mask=mask_2d).to(tl.float32)

    xbar = tl.where(col_mask[None, :], x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = (x * rstd[:, None]) * w[None, :]
    y *= z * tl.sigmoid(z)

    out_ptrs = out_ptr + rows[:, None] * stride_outm + cols[None, :]
    tl.store(out_ptrs, y, mask=mask_2d)


def output_norm_fusable(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    norm_module: torch.nn.Module | None = None,
) -> bool:
    if not is_output_norm_enabled():
        return False
    if core_attn_out.dtype is not torch.bfloat16 or z.dtype is not torch.bfloat16:
        return False
    if core_attn_out.dim() != 3 or z.dim() != 3:
        return False
    if core_attn_out.shape != z.shape:
        return False
    # Layout contract: both tensors must reduce to a strided 2D view of
    # [T*H, D]; x comes from torch.zeros (contiguous), z may be a column
    # slice of mixed_qkvz (row stride > D, last dim still unit-stride).
    x2d = core_attn_out.view(core_attn_out.shape[0] * core_attn_out.shape[1], core_attn_out.shape[2])
    z2d = z.view(z.shape[0] * z.shape[1], z.shape[2]) if z.is_contiguous() else z.reshape(-1, z.shape[2])
    if x2d.stride(1) != 1 or z2d.stride(1) != 1:
        return False
    if norm_module is not None:
        if not hasattr(norm_module, "weight") or norm_module.weight is None:
            return False
        if norm_module.weight.dtype is not torch.bfloat16:
            return False
        if norm_module.weight.shape[0] != core_attn_out.shape[-1]:
            return False
    return True


def _stock_output_norm_rearrange(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    t, h, d = core_attn_out.shape
    c_2d = core_attn_out.reshape(-1, d)
    z_2d = z.reshape(-1, d)
    var = c_2d.to(torch.float32).pow(2).mean(-1, keepdim=True)
    rsqrt = torch.rsqrt(var + eps)
    c_normed = (c_2d.to(torch.float32) * rsqrt) * weight.to(torch.float32)
    silu_z = torch.nn.functional.silu(z_2d.to(torch.float32))
    out_2d = (c_normed * silu_z).to(core_attn_out.dtype)
    return out_2d.reshape(t, h * d)


@torch.library.custom_op("vllm_ascend::gdn_output_norm_rearrange", mutates_args=())
def gdn_output_norm_rearrange(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fuse GDN Part 3 RMSNormGated and rearrange into contiguous [T, H*D]."""
    if not output_norm_fusable(core_attn_out, z) or weight.shape[0] != core_attn_out.shape[-1]:
        return _stock_output_norm_rearrange(core_attn_out, z, weight, eps)

    t, h, d = core_attn_out.shape
    m = t * h
    n = d
    c_2d = core_attn_out.reshape(-1, d)
    z_2d = z.reshape(-1, d)
    out = torch.empty((t, h * d), dtype=core_attn_out.dtype, device=core_attn_out.device)
    out_view = out.view(m, n)

    block_m = 32
    block_n = triton.next_power_of_2(n)
    grid = (triton.cdiv(m, block_m),)

    _output_norm_kernel[grid](
        c_2d,
        z_2d,
        weight,
        out_view,
        c_2d.stride(0),
        z_2d.stride(0),
        out_view.stride(0),
        m,
        n,
        eps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out


@gdn_output_norm_rearrange.register_fake
def _gdn_output_norm_rearrange_fake(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    t, h, d = core_attn_out.shape
    return core_attn_out.new_empty((t, h * d))

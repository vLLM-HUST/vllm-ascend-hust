# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import json
import os
from pathlib import Path
from typing import Any

import torch

_CUSTOM_OP_MARKED: set[str] = set()
_ENABLE_CUSTOM_OP_IMPORT_ATTEMPTED = False
_ENABLE_CUSTOM_OP = None


def activation_sparse_pack_ref(
    x: torch.Tensor,
    threshold: torch.Tensor,
    *,
    inclusive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    threshold = threshold.to(dtype=torch.float32, device=x.device)
    if threshold.numel() == 1:
        threshold = threshold.expand(x.shape[0])
    values = torch.zeros_like(x)
    indices = torch.zeros(x.shape, dtype=torch.int32, device=x.device)
    counts = torch.zeros((x.shape[0],), dtype=torch.int32, device=x.device)
    compare = torch.ge if inclusive else torch.gt
    active = compare(x.abs().to(dtype=torch.float32), threshold.reshape(-1, 1))
    for row in range(x.shape[0]):
        row_indices = torch.nonzero(active[row], as_tuple=False).flatten()
        count = row_indices.numel()
        if count > 0:
            values[row, :count] = x[row, row_indices]
            indices[row, :count] = row_indices.to(dtype=torch.int32)
        counts[row] = count
    return values, indices, counts


def activation_sparse_topk_threshold_ref(
    x: torch.Tensor,
    keep: int,
) -> torch.Tensor:
    if keep < 1 or keep > x.shape[-1]:
        raise ValueError(f"keep must be in [1, {x.shape[-1]}], got {keep}.")
    kth = x.shape[-1] - keep + 1
    return torch.kthvalue(
        x.abs().to(dtype=torch.float32),
        kth,
        dim=-1,
    ).values


def activation_sparse_topk_threshold(
    x: torch.Tensor,
    keep: int,
) -> torch.Tensor:
    if not _can_use_custom_op(x, None):
        if _requires_backend_kernel():
            raise RuntimeError(
                "activation_sparse_topk_threshold requires the Ascend custom "
                "fp16/bf16 op when VLLM_SPARSE_GEMV_REQUIRE_KERNEL is set."
            )
        return activation_sparse_topk_threshold_ref(x, keep)
    return torch.ops._C_ascend.activation_sparse_topk_threshold(
        x.contiguous(),
        keep,
    )


def activation_sparse_linear_packed_ref(
    values: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (values.shape[0], weight.shape[0]),
        dtype=values.dtype,
        device=values.device,
    )
    for row in range(values.shape[0]):
        count = int(counts[row].item())
        if count == 0:
            out[row].zero_()
            continue
        cols = indices[row, :count].to(dtype=torch.long)
        out[row] = torch.matmul(values[row, :count], weight[:, cols].t())
    return out


def activation_sparse_linear_packed_t_ref(
    values: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    weight_t: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (values.shape[0], weight_t.shape[1]),
        dtype=values.dtype,
        device=values.device,
    )
    for row in range(values.shape[0]):
        count = int(counts[row].item())
        if count == 0:
            out[row].zero_()
            continue
        cols = indices[row, :count].to(dtype=torch.long)
        out[row] = torch.matmul(values[row, :count], weight_t[cols])
    return out


def activation_sparse_silu_and_mul_packed_t_ref(
    values: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    weight_t: torch.Tensor,
) -> torch.Tensor:
    gate_up = activation_sparse_linear_packed_t_ref(
        values,
        indices,
        counts,
        weight_t,
    )
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def activation_sparse_linear_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: torch.Tensor,
    *,
    inclusive: bool = False,
) -> torch.Tensor:
    threshold = threshold.to(dtype=torch.float32, device=x.device)
    if threshold.numel() == 1:
        threshold = threshold.reshape(1, 1)
    else:
        threshold = threshold.reshape(x.shape[0], 1)
    compare = torch.ge if inclusive else torch.gt
    sparse_x = torch.where(
        compare(x.abs().to(dtype=torch.float32), threshold),
        x,
        torch.zeros_like(x),
    )
    return torch.matmul(sparse_x, weight.t())


def activation_sparse_linear_direct(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: torch.Tensor,
    *,
    inclusive: bool = False,
) -> torch.Tensor:
    if not _can_use_custom_op(x, weight):
        if _requires_backend_kernel():
            raise RuntimeError(
                "activation_sparse_linear_direct requires the Ascend custom "
                "fp16/bf16 op when VLLM_SPARSE_GEMV_REQUIRE_KERNEL is set."
            )
        return activation_sparse_linear_ref(
            x,
            weight,
            threshold,
            inclusive=inclusive,
        )
    _record_custom_op_invocation(
        "activation_sparse_linear",
        {
            **_tensor_marker_payload("x", x),
            **_tensor_marker_payload("weight", weight),
            **_tensor_marker_payload("threshold", threshold),
            "inclusive": bool(inclusive),
        },
    )
    return torch.ops._C_ascend.activation_sparse_linear(
        x.contiguous(),
        weight.contiguous(),
        threshold.to(dtype=torch.float32, device=x.device).contiguous(),
        inclusive,
    )


def activation_sparse_linear_direct_t(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    threshold: torch.Tensor,
    *,
    inclusive: bool = False,
) -> torch.Tensor:
    if not _can_use_custom_op(x, weight_t):
        if _requires_backend_kernel():
            raise RuntimeError(
                "activation_sparse_linear_direct_t requires the Ascend custom "
                "fp16/bf16 op when VLLM_SPARSE_GEMV_REQUIRE_KERNEL is set."
            )
        return activation_sparse_linear_ref(
            x,
            weight_t.t().contiguous(),
            threshold,
            inclusive=inclusive,
        )
    _record_custom_op_invocation(
        "activation_sparse_linear_direct_t",
        {
            **_tensor_marker_payload("x", x),
            **_tensor_marker_payload("weight_t", weight_t),
            **_tensor_marker_payload("threshold", threshold),
            "inclusive": bool(inclusive),
        },
    )
    return torch.ops._C_ascend.activation_sparse_linear_direct_t(
        x.contiguous(),
        weight_t.contiguous(),
        threshold.to(dtype=torch.float32, device=x.device).contiguous(),
        inclusive,
    )


def activation_sparse_silu_and_mul_direct_t(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    threshold: torch.Tensor,
    *,
    inclusive: bool = False,
) -> torch.Tensor:
    if not _can_use_custom_op(x, weight_t):
        if _requires_backend_kernel():
            raise RuntimeError(
                "activation_sparse_silu_and_mul_direct_t requires the Ascend "
                "custom fp16/bf16 op when VLLM_SPARSE_GEMV_REQUIRE_KERNEL is set."
            )
        sparse_gate_up = activation_sparse_linear_ref(
            x,
            weight_t.t().contiguous(),
            threshold,
            inclusive=inclusive,
        )
        gate, up = sparse_gate_up.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
    if not _should_use_direct_t(x, weight_t):
        return activation_sparse_silu_and_mul_packed_t(
            x,
            weight_t,
            threshold,
            inclusive=inclusive,
        )
    _record_custom_op_invocation(
        "activation_sparse_silu_and_mul_direct_t",
        {
            **_tensor_marker_payload("x", x),
            **_tensor_marker_payload("weight_t", weight_t),
            **_tensor_marker_payload("threshold", threshold),
            "inclusive": bool(inclusive),
        },
    )
    return torch.ops._C_ascend.activation_sparse_silu_and_mul_direct_t(
        x.contiguous(),
        weight_t.contiguous(),
        threshold.to(dtype=torch.float32, device=x.device).contiguous(),
        inclusive,
    )


def activation_sparse_silu_and_mul_packed_t(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    threshold: torch.Tensor,
    *,
    inclusive: bool = False,
) -> torch.Tensor:
    if not _can_use_custom_op(x, weight_t):
        if _requires_backend_kernel():
            raise RuntimeError(
                "activation_sparse_silu_and_mul_packed_t requires the Ascend "
                "custom fp16/bf16 ops when VLLM_SPARSE_GEMV_REQUIRE_KERNEL is set."
            )
        values, indices, counts = activation_sparse_pack_ref(
            x,
            threshold,
            inclusive=inclusive,
        )
        return activation_sparse_silu_and_mul_packed_t_ref(
            values,
            indices,
            counts,
            weight_t,
        )
    _record_custom_op_invocation(
        "activation_sparse_silu_and_mul_packed_t",
        {
            **_tensor_marker_payload("x", x),
            **_tensor_marker_payload("weight_t", weight_t),
            **_tensor_marker_payload("threshold", threshold),
            "inclusive": bool(inclusive),
        },
    )
    values, indices, counts = torch.ops._C_ascend.activation_sparse_pack(
        x.contiguous(),
        threshold.to(dtype=torch.float32, device=x.device).contiguous(),
        inclusive,
    )
    return torch.ops._C_ascend.activation_sparse_silu_and_mul_packed_t(
        values,
        indices,
        counts,
        weight_t.contiguous(),
    )


def activation_sparse_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: torch.Tensor,
    *,
    inclusive: bool = False,
    weight_t: torch.Tensor | None = None,
) -> torch.Tensor:
    custom_weight = weight if weight_t is None else weight_t
    if not _can_use_custom_op(x, custom_weight):
        if _requires_backend_kernel():
            raise RuntimeError(
                "activation_sparse_linear requires the Ascend packed custom "
                "fp16/bf16 ops when VLLM_SPARSE_GEMV_REQUIRE_KERNEL is set."
            )
        values, indices, counts = activation_sparse_pack_ref(
            x,
            threshold,
            inclusive=inclusive,
        )
        if weight_t is None:
            return activation_sparse_linear_packed_ref(values, indices, counts, weight)
        return activation_sparse_linear_packed_t_ref(
            values,
            indices,
            counts,
            weight_t,
        )

    if weight_t is None:
        weight_t = weight.t().contiguous()
    if _should_use_direct_t(x, weight_t):
        return activation_sparse_linear_direct_t(
            x,
            weight_t,
            threshold,
            inclusive=inclusive,
        )

    _record_custom_op_invocation(
        "activation_sparse_linear_packed_t",
        {
            **_tensor_marker_payload("x", x),
            **_tensor_marker_payload("weight", weight),
            **_tensor_marker_payload("threshold", threshold),
            "inclusive": bool(inclusive),
            "weight_t_provided": weight_t is not None,
        },
    )
    values, indices, counts = torch.ops._C_ascend.activation_sparse_pack(
        x.contiguous(),
        threshold.to(dtype=torch.float32, device=x.device).contiguous(),
        inclusive,
    )
    return torch.ops._C_ascend.activation_sparse_linear_packed_t(
        values,
        indices,
        counts,
        weight_t.contiguous(),
    )


def _should_use_direct_t(x: torch.Tensor, weight_t: torch.Tensor) -> bool:
    mode = os.environ.get("VLLM_ASCEND_SPARSE_LINEAR_IMPL", "auto").lower()
    if mode in {"packed", "packed_t"}:
        return False
    if mode in {"direct", "direct_t"}:
        return True
    if mode != "auto":
        return False
    return (
        x.dim() == 2 and x.shape[0] == 1 and x.shape[1] <= 4096 and weight_t.dim() == 2 and weight_t.shape[1] >= 32768
    )


def _requires_backend_kernel() -> bool:
    value = os.environ.get("VLLM_SPARSE_GEMV_REQUIRE_KERNEL", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _can_use_custom_op(x: torch.Tensor, weight: torch.Tensor | None) -> bool:
    supported_dtype = x.dtype in (torch.float16, torch.bfloat16)
    return (
        getattr(x, "is_npu", False)
        and _custom_op_enabled()
        and supported_dtype
        and (weight is None or weight.dtype == x.dtype)
    )


def _tensor_marker_payload(name: str, tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {f"{name}_provided": False}
    return {
        f"{name}_provided": True,
        f"{name}_shape": list(tensor.shape),
        f"{name}_dtype": str(tensor.dtype),
        f"{name}_device": str(tensor.device),
        f"{name}_numel": int(tensor.numel()),
    }


def _record_custom_op_invocation(
    op_name: str,
    payload: dict[str, Any] | None = None,
) -> None:
    raw_marker_path = os.environ.get("VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH")
    if not raw_marker_path or op_name in _CUSTOM_OP_MARKED:
        return
    marker_path = Path(raw_marker_path)
    if not marker_path.is_absolute():
        raise ValueError("VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH must be absolute")
    if not marker_path.parent.is_dir():
        raise ValueError("VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH parent must already exist")
    marker_payload: dict[str, Any] = {
        "op": op_name,
        "pid": os.getpid(),
    }
    if payload is not None:
        marker_payload.update(payload)
    encoded = (json.dumps(marker_payload, sort_keys=True) + "\n").encode()
    descriptor = os.open(
        marker_path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"partial sparse dispatch receipt write: {written}/{len(encoded)}")
    finally:
        os.close(descriptor)
    _CUSTOM_OP_MARKED.add(op_name)


def _custom_op_enabled() -> bool:
    global _ENABLE_CUSTOM_OP_IMPORT_ATTEMPTED, _ENABLE_CUSTOM_OP
    if not _ENABLE_CUSTOM_OP_IMPORT_ATTEMPTED:
        _ENABLE_CUSTOM_OP_IMPORT_ATTEMPTED = True
        try:
            from vllm_ascend.utils import enable_custom_op
        except ModuleNotFoundError:
            _ENABLE_CUSTOM_OP = None
        else:
            _ENABLE_CUSTOM_OP = enable_custom_op
    return False if _ENABLE_CUSTOM_OP is None else _ENABLE_CUSTOM_OP()

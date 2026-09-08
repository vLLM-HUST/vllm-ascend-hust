# SPDX-License-Identifier: Apache-2.0
"""Optional MC2 fused dispatch for PEARL's TP residual path.

The production vLLM compiler pass already emits the AscendC
``matmul_allreduce_add_rmsnorm`` operator.  Native PEARL is intentionally
standalone, so this adapter exposes the same operator with a numerically
equivalent fallback.  Callers can probe support before opting in and retain
the fallback on CANN versions where the custom extension is unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


_MC2_LOAD_ATTEMPTED = False


@dataclass(frozen=True)
class MC2Capability:
    available: bool
    device: str
    tp_size: int
    operator: str
    reason: str


def resolve_hccl_comm_name(
    process_group: dist.ProcessGroup | None = None,
    *,
    device: torch.device | str | None = None,
    rank: int | None = None,
) -> str:
    """Resolve the communicator handle expected by the Ascend MC2 op.

    vLLM-Ascend has used both ``get_hccl_comm_name`` and backend-specific
    process-group helpers across CANN releases.  Keep this compatibility
    probing isolated so the numerical fallback remains usable when no helper
    is exposed (for example in CPU unit tests).
    """
    if process_group is None or not dist.is_available() or not dist.is_initialized():
        return ""
    try:
        backend = process_group._get_backend(torch.device(device or "npu"))  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError, TypeError):
        return ""
    getter = getattr(backend, "get_hccl_comm_name", None)
    if getter is None:
        getter = getattr(process_group, "get_hccl_comm_name", None)
    if getter is None:
        return ""
    candidates = [rank] if rank is not None else [None, dist.get_rank()]
    for candidate in candidates:
        try:
            value = getter() if candidate is None else getter(candidate)
        except (TypeError, RuntimeError, AttributeError):
            continue
        if isinstance(value, str) and value:
            return value
    return ""


def _matmul_op() -> Any | None:
    try:
        namespace = getattr(torch.ops, "_C_ascend")
        return getattr(namespace, "matmul_allreduce_add_rmsnorm")
    except (AttributeError, RuntimeError):
        return None


def detect_mc2_capability(
    device: torch.device | str | None = None,
    tp_size: int = 1,
    *,
    require_fused: bool = True,
) -> MC2Capability:
    """Report whether the compiled MC2 operator can be dispatched."""

    resolved = torch.device(device or "cpu")
    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    if resolved.type != "npu":
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            "matmul_allreduce_add_rmsnorm",
            "MC2 is an Ascend NPU operator",
        )
    global _MC2_LOAD_ATTEMPTED
    if not _MC2_LOAD_ATTEMPTED:
        _MC2_LOAD_ATTEMPTED = True
        try:
            # PEARL can be imported before the regular worker bootstrap. Load
            # the extension lazily for capability checks, but never fail a
            # request just because an optional custom-op library is absent.
            from vllm_ascend.utils import enable_custom_op

            enable_custom_op()
        except Exception:  # optional extension failures must keep fallback usable
            pass
    if _matmul_op() is None:
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            "matmul_allreduce_add_rmsnorm",
            "vllm_ascend custom extension is not loaded",
        )
    if require_fused and tp_size < 2:
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            "matmul_allreduce_add_rmsnorm",
            "fused MC2 is only useful for tensor-parallel groups",
        )
    return MC2Capability(
        True,
        str(resolved),
        int(tp_size),
        "matmul_allreduce_add_rmsnorm",
        "custom AscendC/aclnn dispatch is registered",
    )


def matmul_allreduce_add_rmsnorm_or_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    *,
    group_tp: str = "",
    tp_rank_size: int = 1,
    tp_rank_id: int = 0,
    epsilon: float = 1e-6,
    is_trans_b: bool = True,
    is_gather_add_out: bool = False,
    process_group: dist.ProcessGroup | None = None,
    use_fused: bool | None = None,
    strict_fused: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch MC2 or return the equivalent matmul/all-reduce/RMSNorm pair."""

    output_size = weight.shape[0] if is_trans_b else weight.shape[-1]
    if x.ndim < 2 or residual.shape != x.shape[:-1] + (output_size,):
        raise ValueError("MC2 input/output shapes are inconsistent")
    if gamma.numel() != residual.shape[-1]:
        raise ValueError("RMSNorm gamma must match the residual hidden size")
    if tp_rank_size < 1 or not 0 <= tp_rank_id < tp_rank_size:
        raise ValueError("invalid tensor-parallel rank metadata")
    capability = detect_mc2_capability(x.device, tp_rank_size)
    dispatch_fused = capability.available if use_fused is None else bool(use_fused) and capability.available
    if dispatch_fused and _matmul_op() is not None:
        try:
            return _matmul_op()(  # type: ignore[misc]
                x,
                weight,
                residual,
                gamma,
                group_tp,
                int(tp_rank_size),
                int(tp_rank_id),
                float(epsilon),
                bool(is_trans_b),
                bool(is_gather_add_out),
            )
        except Exception as error:
            if strict_fused:
                raise RuntimeError("MC2 fused dispatch failed in strict mode") from error
    if is_trans_b:
        matmul = F.linear(x, weight)
    else:
        matmul = torch.matmul(x, weight)
    if tp_rank_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(matmul, group=process_group)
    added = matmul + residual
    norm = added * torch.rsqrt(added.float().pow(2).mean(dim=-1, keepdim=True) + float(epsilon)).to(added.dtype)
    return norm * gamma, added


def capability_dict(device: torch.device | str | None = None, tp_size: int = 1) -> dict[str, Any]:
    """Return JSON-friendly MC2 capability information."""

    return asdict(detect_mc2_capability(device, tp_size))


__all__ = [
    "MC2Capability",
    "capability_dict",
    "detect_mc2_capability",
    "matmul_allreduce_add_rmsnorm_or_fallback",
    "resolve_hccl_comm_name",
]

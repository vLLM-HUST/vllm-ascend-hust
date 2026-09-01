import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from vllm_ascend.ops.sparse_linear import (
    _custom_op_enabled,
    activation_sparse_linear,
    activation_sparse_linear_direct,
    activation_sparse_linear_direct_t,
    activation_sparse_silu_and_mul_direct_t,
    activation_sparse_silu_and_mul_packed_t,
    activation_sparse_topk_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-dim", type=int, default=3584)
    parser.add_argument("--output-dim", type=int, default=18944)
    parser.add_argument(
        "--fused-silu-and-mul",
        action="store_true",
        help=("Benchmark fused gate/up projection plus SiLU*up. output-dim must be 2 * intermediate_dim."),
    )
    parser.add_argument("--sparsity", type=float, default=0.4)
    parser.add_argument(
        "--threshold-mode",
        choices=["scalar", "row_topk"],
        default="scalar",
        help=(
            "scalar matches precomputed TEAL thresholds; row_topk matches La RoSA's per-row top-k threshold generation."
        ),
    )
    parser.add_argument(
        "--row-topk-threshold-backend",
        choices=["kthvalue", "topk", "ascend"],
        default="kthvalue",
        help="Threshold primitive used by --threshold-mode row_topk.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--weight-format",
        choices=["nd", "nz"],
        default="nz",
        help=(
            "Storage format for the dense reference weight. vLLM converts "
            "linear weights to FRACTAL_NZ on Ascend, while the sparse kernel "
            "uses a cached contiguous ND transpose."
        ),
    )
    parser.add_argument("--inclusive", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--max-sparse-err",
        type=float,
        default=None,
        help="Fail if packed sparse max abs error exceeds this value.",
    )
    parser.add_argument(
        "--max-sparse-rel-err",
        type=float,
        default=None,
        help="Fail if packed sparse max abs error / max abs reference exceeds this value.",
    )
    parser.add_argument(
        "--max-direct-err",
        type=float,
        default=None,
        help="Fail if direct sparse max abs error exceeds this value.",
    )
    parser.add_argument(
        "--max-direct-rel-err",
        type=float,
        default=None,
        help="Fail if direct sparse max abs error / max abs reference exceeds this value.",
    )
    parser.add_argument(
        "--max-direct-t-err",
        type=float,
        default=None,
        help="Fail if direct_t sparse max abs error exceeds this value.",
    )
    parser.add_argument(
        "--max-direct-t-rel-err",
        type=float,
        default=None,
        help="Fail if direct_t sparse max abs error / max abs reference exceeds this value.",
    )
    parser.add_argument(
        "--min-packed-total-speedup",
        type=float,
        default=None,
        help="Fail if dense_ms / packed_total_ms is below this value.",
    )
    parser.add_argument(
        "--min-packed-total-with-threshold-speedup",
        type=float,
        default=None,
        help=(
            "Fail if dense_ms / packed_total_with_threshold_ms is below this "
            "value. Online threshold cost is included for row_topk mode."
        ),
    )
    parser.add_argument(
        "--min-packed-compute-speedup",
        type=float,
        default=None,
        help="Fail if dense_ms / packed_compute_ms is below this value.",
    )
    parser.add_argument(
        "--min-direct-speedup",
        type=float,
        default=None,
        help="Fail if dense_ms / direct_sparse_ms is below this value.",
    )
    parser.add_argument(
        "--min-direct-t-speedup",
        type=float,
        default=None,
        help="Fail if dense_ms / direct_t_sparse_ms is below this value.",
    )
    parser.add_argument(
        "--min-direct-t-with-threshold-speedup",
        type=float,
        default=None,
        help=("Fail if dense_ms / (direct_t_sparse_ms + online threshold_ms) is below this value."),
    )
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        help="Skip the direct sparse kernel and benchmark only the packed path.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def synchronize() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize()
    return (time.perf_counter() - start) / iters


def topk_keep(input_dim: int, sparsity: float) -> int:
    keep = int(input_dim * (1.0 - sparsity))
    if keep < 1:
        raise ValueError(
            "row_topk threshold mode requires at least one kept activation; "
            f"got input_dim={input_dim}, sparsity={sparsity}."
        )
    return keep


def build_threshold(
    x: torch.Tensor,
    sparsity: float,
    threshold_mode: str,
    row_topk_threshold_backend: str,
) -> torch.Tensor:
    if threshold_mode == "scalar":
        x_abs = x.abs().to(dtype=torch.float32)
        return torch.quantile(x_abs.flatten(), sparsity).reshape(())
    if threshold_mode == "row_topk":
        keep = topk_keep(x.shape[-1], sparsity)
        if row_topk_threshold_backend == "ascend":
            return activation_sparse_topk_threshold(x, keep)
        x_abs = x.abs().to(dtype=torch.float32)
        if row_topk_threshold_backend == "topk":
            topk_values, _ = torch.topk(x_abs, keep, dim=-1)
            return topk_values[..., -1].contiguous()
        kth = x.shape[-1] - keep + 1
        return torch.kthvalue(x_abs, kth, dim=-1).values.contiguous()
    raise ValueError(f"Unsupported threshold mode: {threshold_mode}")


def threshold_for_mask(threshold: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if threshold.numel() == 1:
        return threshold.reshape(())
    return threshold.reshape(x.shape[0], 1)


def apply_dense_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    fused_silu_and_mul: bool,
) -> torch.Tensor:
    out = torch.nn.functional.linear(x, weight)
    if not fused_silu_and_mul:
        return out
    gate, up = out.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def npu_format(tensor: torch.Tensor) -> int | None:
    try:
        import torch_npu

        return int(torch_npu.get_npu_format(tensor))
    except (AttributeError, ImportError, RuntimeError):
        return None


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.to(dtype=torch.float32)
    expected_f = expected.to(dtype=torch.float32)
    diff = actual_f - expected_f
    max_abs = diff.abs().max().item()
    rms = torch.sqrt(torch.mean(diff * diff)).item()
    ref_max_abs = expected_f.abs().max().item()
    ref_rms = torch.sqrt(torch.mean(expected_f * expected_f)).item()
    eps = torch.finfo(torch.float32).eps
    return {
        "max_abs": max_abs,
        "max_rel": max_abs / max(ref_max_abs, eps),
        "rms": rms,
        "rms_rel": rms / max(ref_rms, eps),
    }


def main() -> None:
    args = parse_args()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is required for this benchmark.")
    if not _custom_op_enabled():
        raise RuntimeError("Ascend custom ops must be enabled for this benchmark.")
    if not 0.0 <= args.sparsity < 1.0:
        raise ValueError("--sparsity must be in [0, 1).")
    if args.fused_silu_and_mul and args.output_dim % 2 != 0:
        raise ValueError("--fused-silu-and-mul requires an even --output-dim")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("npu")
    torch.manual_seed(0)

    x = torch.randn(args.batch_size, args.input_dim, device=device, dtype=dtype)
    weight = torch.randn(args.output_dim, args.input_dim, device=device, dtype=dtype)
    if args.weight_format == "nz":
        import torch_npu

        weight = torch_npu.npu_format_cast(weight, 29)
    weight_t = weight.t().contiguous()
    topk_keep_count = topk_keep(args.input_dim, args.sparsity) if args.threshold_mode == "row_topk" else None
    threshold = build_threshold(
        x,
        args.sparsity,
        args.threshold_mode,
        args.row_topk_threshold_backend,
    )
    effective_inclusive = args.inclusive or args.threshold_mode == "row_topk"
    compare = torch.ge if effective_inclusive else torch.gt

    dense_out = apply_dense_op(x, weight, args.fused_silu_and_mul)
    direct_out = None
    if not args.skip_direct and not args.fused_silu_and_mul:
        direct_out = activation_sparse_linear_direct(
            x,
            weight,
            threshold,
            inclusive=effective_inclusive,
        )
    if args.fused_silu_and_mul:
        packed_out = activation_sparse_silu_and_mul_packed_t(
            x,
            weight_t,
            threshold,
            inclusive=effective_inclusive,
        )
        direct_t_out = activation_sparse_silu_and_mul_direct_t(
            x,
            weight_t,
            threshold,
            inclusive=effective_inclusive,
        )
    else:
        packed_out = activation_sparse_linear(
            x,
            weight,
            threshold,
            inclusive=effective_inclusive,
            weight_t=weight_t,
        )
        direct_t_out = activation_sparse_linear_direct_t(
            x,
            weight_t,
            threshold,
            inclusive=effective_inclusive,
        )
    threshold_mask = threshold_for_mask(threshold, x)
    masked_x = torch.where(
        compare(x.abs().to(dtype=torch.float32), threshold_mask),
        x,
        torch.zeros_like(x),
    )
    expected_sparse = apply_dense_op(masked_x, weight, args.fused_silu_and_mul)
    synchronize()

    sparse_err = error_stats(packed_out, expected_sparse)
    sparse_max_abs_err = sparse_err["max_abs"]
    sparse_max_rel_err = sparse_err["max_rel"]
    sparse_rms_err = sparse_err["rms"]
    sparse_rms_rel_err = sparse_err["rms_rel"]
    direct_max_abs_err = None
    direct_max_rel_err = None
    direct_rms_err = None
    direct_rms_rel_err = None
    if direct_out is not None:
        direct_err = error_stats(direct_out, expected_sparse)
        direct_max_abs_err = direct_err["max_abs"]
        direct_max_rel_err = direct_err["max_rel"]
        direct_rms_err = direct_err["rms"]
        direct_rms_rel_err = direct_err["rms_rel"]
    direct_t_err = error_stats(direct_t_out, expected_sparse)
    direct_t_max_abs_err = direct_t_err["max_abs"]
    direct_t_max_rel_err = direct_t_err["max_rel"]
    direct_t_rms_err = direct_t_err["rms"]
    direct_t_rms_rel_err = direct_t_err["rms_rel"]
    expected_sparse_max_abs = expected_sparse.to(dtype=torch.float32).abs().max().item()
    expected_sparse_rms = torch.sqrt(torch.mean(expected_sparse.to(dtype=torch.float32) ** 2)).item()
    dense_sparse_max_abs_delta = (
        (dense_out.to(dtype=torch.float32) - expected_sparse.to(dtype=torch.float32)).abs().max().item()
    )

    values, indices, counts = torch.ops._C_ascend.activation_sparse_pack(
        x.contiguous(),
        threshold.to(dtype=torch.float32, device=x.device).contiguous(),
        effective_inclusive,
    )
    synchronize()

    dense_time = bench(
        lambda: apply_dense_op(x, weight, args.fused_silu_and_mul),
        args.warmup,
        args.iters,
    )
    threshold_time = bench(
        lambda: build_threshold(
            x,
            args.sparsity,
            args.threshold_mode,
            args.row_topk_threshold_backend,
        ),
        args.warmup,
        args.iters,
    )
    direct_sparse_time = None
    if not args.skip_direct and not args.fused_silu_and_mul:
        direct_sparse_time = bench(
            lambda: activation_sparse_linear_direct(
                x,
                weight,
                threshold,
                inclusive=effective_inclusive,
            ),
            args.warmup,
            args.iters,
        )
    if args.fused_silu_and_mul:
        direct_t_sparse_time = bench(
            lambda: activation_sparse_silu_and_mul_direct_t(
                x,
                weight_t,
                threshold,
                inclusive=effective_inclusive,
            ),
            args.warmup,
            args.iters,
        )
    else:
        direct_t_sparse_time = bench(
            lambda: activation_sparse_linear_direct_t(
                x,
                weight_t,
                threshold,
                inclusive=effective_inclusive,
            ),
            args.warmup,
            args.iters,
        )
    pack_time = bench(
        lambda: torch.ops._C_ascend.activation_sparse_pack(
            x.contiguous(),
            threshold.to(dtype=torch.float32, device=x.device).contiguous(),
            effective_inclusive,
        ),
        args.warmup,
        args.iters,
    )
    if args.fused_silu_and_mul:
        packed_compute_time = bench(
            lambda: torch.ops._C_ascend.activation_sparse_silu_and_mul_packed_t(
                values,
                indices,
                counts,
                weight_t,
            ),
            args.warmup,
            args.iters,
        )
        packed_total_time = bench(
            lambda: activation_sparse_silu_and_mul_packed_t(
                x,
                weight_t,
                threshold,
                inclusive=effective_inclusive,
            ),
            args.warmup,
            args.iters,
        )
    else:
        packed_compute_time = bench(
            lambda: torch.ops._C_ascend.activation_sparse_linear_packed_t(
                values,
                indices,
                counts,
                weight_t,
            ),
            args.warmup,
            args.iters,
        )
        packed_total_time = bench(
            lambda: activation_sparse_linear(
                x,
                weight,
                threshold,
                inclusive=effective_inclusive,
                weight_t=weight_t,
            ),
            args.warmup,
            args.iters,
        )
    threshold_online = args.threshold_mode == "row_topk"
    online_threshold_time = threshold_time if threshold_online else 0.0
    packed_total_with_threshold_time = packed_total_time + online_threshold_time
    direct_t_with_threshold_time = direct_t_sparse_time + online_threshold_time
    density = compare(x.abs().to(dtype=torch.float32), threshold_mask).float().mean().item()
    direct_speedup = None if direct_sparse_time is None else dense_time / direct_sparse_time
    direct_t_speedup = dense_time / direct_t_sparse_time
    direct_t_with_threshold_speedup = dense_time / direct_t_with_threshold_time
    packed_total_speedup = dense_time / packed_total_time
    packed_compute_speedup = dense_time / packed_compute_time
    packed_total_with_threshold_speedup = dense_time / packed_total_with_threshold_time
    failures = []
    if not math.isfinite(sparse_max_abs_err):
        failures.append(f"packed sparse max abs error is not finite: {sparse_max_abs_err}")
    if args.max_sparse_err is not None and (
        not math.isfinite(sparse_max_abs_err) or sparse_max_abs_err > args.max_sparse_err
    ):
        failures.append(f"packed sparse max abs error {sparse_max_abs_err:.6g} > {args.max_sparse_err:.6g}")
    if args.max_sparse_rel_err is not None and (
        not math.isfinite(sparse_max_rel_err) or sparse_max_rel_err > args.max_sparse_rel_err
    ):
        failures.append(f"packed sparse max relative error {sparse_max_rel_err:.6g} > {args.max_sparse_rel_err:.6g}")
    if direct_max_abs_err is not None and not math.isfinite(direct_max_abs_err):
        failures.append(f"direct sparse max abs error is not finite: {direct_max_abs_err}")
    if direct_max_rel_err is not None and not math.isfinite(direct_max_rel_err):
        failures.append(f"direct sparse max relative error is not finite: {direct_max_rel_err}")
    if not math.isfinite(direct_t_max_abs_err):
        failures.append(f"direct_t sparse max abs error is not finite: {direct_t_max_abs_err}")
    if not math.isfinite(direct_t_max_rel_err):
        failures.append(f"direct_t sparse max relative error is not finite: {direct_t_max_rel_err}")
    if (
        args.max_direct_err is not None
        and direct_max_abs_err is not None
        and (not math.isfinite(direct_max_abs_err) or direct_max_abs_err > args.max_direct_err)
    ):
        failures.append(f"direct sparse max abs error {direct_max_abs_err:.6g} > {args.max_direct_err:.6g}")
    if (
        args.max_direct_rel_err is not None
        and direct_max_rel_err is not None
        and (not math.isfinite(direct_max_rel_err) or direct_max_rel_err > args.max_direct_rel_err)
    ):
        failures.append(f"direct sparse max relative error {direct_max_rel_err:.6g} > {args.max_direct_rel_err:.6g}")
    if args.max_direct_t_err is not None and (
        not math.isfinite(direct_t_max_abs_err) or direct_t_max_abs_err > args.max_direct_t_err
    ):
        failures.append(f"direct_t sparse max abs error {direct_t_max_abs_err:.6g} > {args.max_direct_t_err:.6g}")
    if args.max_direct_t_rel_err is not None and (
        not math.isfinite(direct_t_max_rel_err) or direct_t_max_rel_err > args.max_direct_t_rel_err
    ):
        failures.append(
            f"direct_t sparse max relative error {direct_t_max_rel_err:.6g} > {args.max_direct_t_rel_err:.6g}"
        )
    if args.max_direct_err is not None and direct_max_abs_err is None:
        failures.append("--max-direct-err cannot be used with --skip-direct")
    if args.max_direct_rel_err is not None and direct_max_rel_err is None:
        failures.append("--max-direct-rel-err cannot be used with --skip-direct")
    if args.min_packed_total_speedup is not None and packed_total_speedup < args.min_packed_total_speedup:
        failures.append(f"packed total speedup {packed_total_speedup:.6g} < {args.min_packed_total_speedup:.6g}")
    if (
        args.min_packed_total_with_threshold_speedup is not None
        and packed_total_with_threshold_speedup < args.min_packed_total_with_threshold_speedup
    ):
        failures.append(
            "packed total with threshold speedup "
            f"{packed_total_with_threshold_speedup:.6g} < "
            f"{args.min_packed_total_with_threshold_speedup:.6g}"
        )
    if args.min_packed_compute_speedup is not None and packed_compute_speedup < args.min_packed_compute_speedup:
        failures.append(f"packed compute speedup {packed_compute_speedup:.6g} < {args.min_packed_compute_speedup:.6g}")
    if args.min_direct_speedup is not None and direct_speedup is not None and direct_speedup < args.min_direct_speedup:
        failures.append(f"direct speedup {direct_speedup:.6g} < {args.min_direct_speedup:.6g}")
    if args.min_direct_speedup is not None and direct_speedup is None:
        failures.append("--min-direct-speedup cannot be used with --skip-direct")
    if args.min_direct_t_speedup is not None and direct_t_speedup < args.min_direct_t_speedup:
        failures.append(f"direct_t speedup {direct_t_speedup:.6g} < {args.min_direct_t_speedup:.6g}")
    if (
        args.min_direct_t_with_threshold_speedup is not None
        and direct_t_with_threshold_speedup < args.min_direct_t_with_threshold_speedup
    ):
        failures.append(
            "direct_t with threshold speedup "
            f"{direct_t_with_threshold_speedup:.6g} < "
            f"{args.min_direct_t_with_threshold_speedup:.6g}"
        )

    result = {
        "batch_size": args.batch_size,
        "input_dim": args.input_dim,
        "output_dim": args.output_dim,
        "fused_silu_and_mul": args.fused_silu_and_mul,
        "dtype": args.dtype,
        "weight_format": args.weight_format,
        "weight_npu_format": npu_format(weight),
        "weight_t_npu_format": npu_format(weight_t),
        "threshold_mode": args.threshold_mode,
        "row_topk_threshold_backend": args.row_topk_threshold_backend,
        "requested_inclusive": args.inclusive,
        "inclusive": effective_inclusive,
        "topk_keep": topk_keep_count,
        "threshold_online": threshold_online,
        "requested_sparsity": args.sparsity,
        "measured_density": density,
        "measured_sparsity": 1.0 - density,
        "nnz_min": int(counts.min().item()),
        "nnz_max": int(counts.max().item()),
        "nnz_mean": float(counts.to(dtype=torch.float32).mean().item()),
        "sparse_max_abs_err": sparse_max_abs_err,
        "sparse_max_rel_err": sparse_max_rel_err,
        "sparse_rms_err": sparse_rms_err,
        "sparse_rms_rel_err": sparse_rms_rel_err,
        "direct_max_abs_err": direct_max_abs_err,
        "direct_max_rel_err": direct_max_rel_err,
        "direct_rms_err": direct_rms_err,
        "direct_rms_rel_err": direct_rms_rel_err,
        "direct_t_max_abs_err": direct_t_max_abs_err,
        "direct_t_max_rel_err": direct_t_max_rel_err,
        "direct_t_rms_err": direct_t_rms_err,
        "direct_t_rms_rel_err": direct_t_rms_rel_err,
        "expected_sparse_max_abs": expected_sparse_max_abs,
        "expected_sparse_rms": expected_sparse_rms,
        "dense_sparse_max_abs_delta": dense_sparse_max_abs_delta,
        "weight_t_cached": True,
        "env_output_tiles": {
            "VLLM_ASCEND_SPARSE_DIRECT_T_OUTPUT_TILE": os.environ.get("VLLM_ASCEND_SPARSE_DIRECT_T_OUTPUT_TILE"),
            "VLLM_ASCEND_SPARSE_PACKED_T_OUTPUT_TILE": os.environ.get("VLLM_ASCEND_SPARSE_PACKED_T_OUTPUT_TILE"),
            "VLLM_ASCEND_SPARSE_SILU_DIRECT_T_OUTPUT_TILE": os.environ.get(
                "VLLM_ASCEND_SPARSE_SILU_DIRECT_T_OUTPUT_TILE"
            ),
            "VLLM_ASCEND_SPARSE_SILU_PACKED_T_OUTPUT_TILE": os.environ.get(
                "VLLM_ASCEND_SPARSE_SILU_PACKED_T_OUTPUT_TILE"
            ),
        },
        "dense_ms": dense_time * 1000.0,
        "threshold_ms": threshold_time * 1000.0,
        "online_threshold_ms": online_threshold_time * 1000.0,
        "direct_sparse_ms": (None if direct_sparse_time is None else direct_sparse_time * 1000.0),
        "direct_t_sparse_ms": direct_t_sparse_time * 1000.0,
        "direct_t_with_threshold_ms": (direct_t_with_threshold_time * 1000.0),
        "pack_ms": pack_time * 1000.0,
        "packed_compute_ms": packed_compute_time * 1000.0,
        "packed_total_ms": packed_total_time * 1000.0,
        "packed_total_with_threshold_ms": (packed_total_with_threshold_time * 1000.0),
        "direct_speedup": direct_speedup,
        "direct_t_speedup": direct_t_speedup,
        "direct_t_with_threshold_speedup": (direct_t_with_threshold_speedup),
        "packed_total_speedup": packed_total_speedup,
        "packed_total_with_threshold_speedup": (packed_total_with_threshold_speedup),
        "packed_compute_speedup": packed_compute_speedup,
        "passed": not failures,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
    if failures:
        print("activation_sparse_linear benchmark failed gates:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

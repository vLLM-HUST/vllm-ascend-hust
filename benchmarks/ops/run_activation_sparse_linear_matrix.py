import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_PROJECTIONS = (
    (3584, 4608),  # qkv, Qwen2.5-7B style
    (3584, 37888),  # fused gate/up, Qwen2.5-7B style
    (18944, 3584),  # down projection, Qwen2.5-7B style
    (3584, 3584),  # o projection, Qwen2.5-7B style
)


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"value must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"value must be a positive integer, got {value!r}")
    return parsed


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be BxIxO, got {value!r}")
    try:
        batch_size, input_dim, output_dim = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"shape must contain integers, got {value!r}") from exc
    if batch_size <= 0 or input_dim <= 0 or output_dim <= 0:
        raise argparse.ArgumentTypeError(f"shape dimensions must be positive, got {value!r}")
    return batch_size, input_dim, output_dim


def default_shapes(batch_sizes: list[int]) -> list[tuple[int, int, int]]:
    seen_batch_sizes = list(dict.fromkeys(batch_sizes))
    return [
        (batch_size, input_dim, output_dim)
        for batch_size in seen_batch_sizes
        for input_dim, output_dim in DEFAULT_PROJECTIONS
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run activation_sparse_linear benchmarks across projection shapes and write a summary JSON.")
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=None,
        help="Projection shape as BxIxO. Can be repeated.",
    )
    parser.add_argument(
        "--default-batch-size",
        action="append",
        type=parse_positive_int,
        default=None,
        help=(
            "Batch size used with the default Qwen2.5 projection shapes. "
            "Ignored when --shape is supplied. Can be repeated."
        ),
    )
    parser.add_argument(
        "--sparsity",
        action="append",
        type=float,
        default=None,
        help="Requested sparsity. Can be repeated.",
    )
    parser.add_argument(
        "--threshold-mode",
        action="append",
        choices=["scalar", "row_topk"],
        default=None,
        help=("Threshold mode to benchmark. Can be repeated. Defaults to scalar and row_topk."),
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--inclusive", action="store_true")
    parser.add_argument(
        "--fused-silu-and-mul",
        action="store_true",
        help=(
            "Forward --fused-silu-and-mul to the per-shape benchmark. Use only "
            "with shapes whose output dim is 2 * intermediate_dim."
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--max-sparse-err",
        type=float,
        default=None,
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--max-sparse-rel-err",
        type=float,
        default=None,
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--max-direct-err",
        type=float,
        default=None,
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--max-direct-rel-err",
        type=float,
        default=None,
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--max-direct-t-err",
        type=float,
        default=None,
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--max-direct-t-rel-err",
        type=float,
        default=None,
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--min-packed-total-speedup",
        type=float,
        default=None,
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--min-packed-total-with-threshold-speedup",
        type=float,
        default=None,
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--min-packed-compute-speedup",
        type=float,
        default=None,
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--min-direct-speedup",
        type=float,
        default=None,
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--min-direct-t-speedup",
        type=float,
        default=None,
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        help="Forwarded to bench_activation_sparse_linear.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/activation_sparse_linear_bench"),
    )
    return parser.parse_args()


def add_optional_float(
    cmd: list[str],
    name: str,
    value: float | None,
) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def main() -> int:
    args = parse_args()
    default_batch_sizes = args.default_batch_size or [1]
    shapes = args.shape or default_shapes(default_batch_sizes)
    sparsities = args.sparsity or [0.25, 0.40, 0.50, 0.75]
    threshold_modes = args.threshold_mode or ["scalar", "row_topk"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bench_script = Path(__file__).with_name("bench_activation_sparse_linear.py")
    results = []
    failures = []
    for batch_size, input_dim, output_dim in shapes:
        shape_name = f"b{batch_size}_i{input_dim}_o{output_dim}"
        for sparsity in sparsities:
            sparsity_name = str(sparsity).replace(".", "p")
            for threshold_mode in threshold_modes:
                json_output = args.output_dir / f"{shape_name}_m{threshold_mode}_s{sparsity_name}_{args.dtype}.json"
                cmd = [
                    sys.executable,
                    str(bench_script),
                    "--batch-size",
                    str(batch_size),
                    "--input-dim",
                    str(input_dim),
                    "--output-dim",
                    str(output_dim),
                    "--sparsity",
                    str(sparsity),
                    "--threshold-mode",
                    threshold_mode,
                    "--dtype",
                    args.dtype,
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    "--json-output",
                    str(json_output),
                ]
                if args.inclusive:
                    cmd.append("--inclusive")
                if args.fused_silu_and_mul:
                    cmd.append("--fused-silu-and-mul")
                if args.skip_direct:
                    cmd.append("--skip-direct")
                add_optional_float(cmd, "--max-sparse-err", args.max_sparse_err)
                add_optional_float(
                    cmd,
                    "--max-sparse-rel-err",
                    args.max_sparse_rel_err,
                )
                add_optional_float(cmd, "--max-direct-err", args.max_direct_err)
                add_optional_float(
                    cmd,
                    "--max-direct-rel-err",
                    args.max_direct_rel_err,
                )
                add_optional_float(cmd, "--max-direct-t-err", args.max_direct_t_err)
                add_optional_float(
                    cmd,
                    "--max-direct-t-rel-err",
                    args.max_direct_t_rel_err,
                )
                add_optional_float(
                    cmd,
                    "--min-packed-total-speedup",
                    args.min_packed_total_speedup,
                )
                add_optional_float(
                    cmd,
                    "--min-packed-total-with-threshold-speedup",
                    args.min_packed_total_with_threshold_speedup,
                )
                add_optional_float(
                    cmd,
                    "--min-packed-compute-speedup",
                    args.min_packed_compute_speedup,
                )
                add_optional_float(
                    cmd,
                    "--min-direct-speedup",
                    args.min_direct_speedup,
                )
                add_optional_float(
                    cmd,
                    "--min-direct-t-speedup",
                    args.min_direct_t_speedup,
                )

                proc = subprocess.run(cmd, text=True, capture_output=True)
                if proc.stdout:
                    print(proc.stdout, end="")
                if proc.stderr:
                    print(proc.stderr, end="", file=sys.stderr)
                if json_output.exists():
                    results.append(json.loads(json_output.read_text()))
                else:
                    failures.append(
                        {
                            "shape": [batch_size, input_dim, output_dim],
                            "threshold_mode": threshold_mode,
                            "sparsity": sparsity,
                            "returncode": proc.returncode,
                            "reason": "missing_json_output",
                        }
                    )
                if proc.returncode != 0:
                    failures.append(
                        {
                            "shape": [batch_size, input_dim, output_dim],
                            "threshold_mode": threshold_mode,
                            "sparsity": sparsity,
                            "returncode": proc.returncode,
                            "reason": "benchmark_failed",
                        }
                    )

    summary = {
        "dtype": args.dtype,
        "inclusive": args.inclusive,
        "default_batch_sizes": ([] if args.shape else list(dict.fromkeys(default_batch_sizes))),
        "threshold_modes": threshold_modes,
        "cases": len(results),
        "passed": not failures and all(result.get("passed") for result in results),
        "failures": failures,
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        "shape\t mode\t sparsity\t dense_ms\t direct_t_ms\t "
        "direct_t_speedup\t packed_total_ms\t with_threshold_ms\t speedup\t "
        "with_threshold_speedup\t passed"
    )
    for result in results:
        print(
            f"{result['batch_size']}x{result['input_dim']}x{result['output_dim']}"
            f"\t {result['threshold_mode']}"
            f"\t {result['requested_sparsity']:.2f}"
            f"\t {result['dense_ms']:.4f}"
            f"\t {result['direct_t_sparse_ms']:.4f}"
            f"\t {result['direct_t_speedup']:.4f}"
            f"\t {result['packed_total_ms']:.4f}"
            f"\t {result['packed_total_with_threshold_ms']:.4f}"
            f"\t {result['packed_total_speedup']:.4f}"
            f"\t {result['packed_total_with_threshold_speedup']:.4f}"
            f"\t {result['passed']}"
        )
    print(f"summary_json\t{summary_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

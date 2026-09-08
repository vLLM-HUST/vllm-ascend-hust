# SPDX-License-Identifier: Apache-2.0
"""Compare linear dispatch paths for Qwen2.5-14B TP3 matrix shapes."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


TP3_PROJECTIONS = (
    ("qkv_nd_bias", 5120, 2688, False, True),
    ("gate_up_nz", 5120, 9216, True, False),
    ("attention_out_nz", 1920, 5120, True, False),
    ("down_nz", 4608, 5120, True, False),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--token-counts", type=int, nargs="+", default=[320, 464, 512])
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _capture(operation: Callable[[], torch.Tensor]) -> tuple[torch.npu.NPUGraph, torch.Tensor]:
    operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operation()
    torch.npu.synchronize()
    return graph, output


def _measure(graph: torch.npu.NPUGraph, warmup_steps: int, profile_steps: int) -> float:
    for _ in range(warmup_steps):
        graph.replay()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(profile_steps):
        graph.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1_000_000 / profile_steps


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().cpu())


def main() -> None:
    args = _parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(20260810)
    results: list[dict[str, float | int | str | bool]] = []
    for projection, input_size, output_size, use_nz, use_bias in TP3_PROJECTIONS:
        weight = (
            torch.randn(output_size, input_size, dtype=torch.float32, device="npu") * 0.02
        ).to(dtype)
        if use_nz:
            weight = torch_npu.npu_format_cast(weight, 29)
        bias = None
        if use_bias:
            bias = (
                torch.randn(output_size, dtype=torch.float32, device="npu") * 0.02
            ).to(dtype)

        for token_count in args.token_counts:
            source = (
                torch.randn(token_count, input_size, dtype=torch.float32, device="npu")
                * 0.02
            ).to(dtype)
            replay_source = (
                torch.randn_like(source, dtype=torch.float32) * 0.02
            ).to(dtype)

            def functional_linear() -> torch.Tensor:
                return F.linear(source, weight, bias)

            def npu_linear() -> torch.Tensor:
                return torch_npu.npu_linear(source, weight, bias)

            graphs: dict[str, tuple[torch.npu.NPUGraph, torch.Tensor]] = {}
            for strategy, operation in (
                ("functional_linear", functional_linear),
                ("npu_linear", npu_linear),
            ):
                try:
                    graph, output = _capture(operation)
                    latency_us = _measure(graph, args.warmup_steps, args.profile_steps)
                    graphs[strategy] = (graph, output)
                    status = "ok"
                except Exception as error:
                    latency_us = float("nan")
                    status = f"{type(error).__name__}: {error}"
                results.append({
                    "projection": projection,
                    "token_count": token_count,
                    "input_size": input_size,
                    "output_size": output_size,
                    "weight_nz": use_nz,
                    "bias": use_bias,
                    "strategy": strategy,
                    "latency_microseconds": latency_us,
                    "status": status,
                })

            source.copy_(replay_source)
            for graph, _ in graphs.values():
                graph.replay()
            torch.npu.synchronize()
            if "functional_linear" in graphs:
                expected = graphs["functional_linear"][1]
                for result in results[-2:]:
                    strategy = str(result["strategy"])
                    if strategy in graphs:
                        result["changed_input_replay_max_abs_error"] = _max_abs_error(
                            graphs[strategy][1],
                            expected,
                        )

    payload = {"dtype": args.dtype, "results": results}
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

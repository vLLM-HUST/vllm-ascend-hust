# SPDX-License-Identifier: Apache-2.0
"""Compare Qwen2.5 TP3 QKV weight layouts inside an Ascend graph."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[128, 256, 384, 512])
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--qkv-size-per-rank", type=int, default=2688)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _measure_eager(operation: Callable[[], torch.Tensor], warmup_steps: int, profile_steps: int) -> float:
    for _ in range(warmup_steps):
        operation()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(profile_steps):
        operation()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1_000_000 / profile_steps


def _capture(operation: Callable[[], torch.Tensor]) -> tuple[torch.npu.NPUGraph, torch.Tensor]:
    operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operation()
    torch.npu.synchronize()
    return graph, output


def _measure_graph(graph: torch.npu.NPUGraph, warmup_steps: int, profile_steps: int) -> float:
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
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(43)

    weight_nd = (
        torch.randn(
            (args.qkv_size_per_rank, args.hidden_size),
            dtype=torch.float32,
            device="npu",
        )
        * 0.02
    ).to(torch.bfloat16)
    weight_nz = torch_npu.npu_format_cast(weight_nd, 29)
    bias = (
        torch.randn(args.qkv_size_per_rank, dtype=torch.float32, device="npu") * 0.02
    ).to(torch.bfloat16)

    results: list[dict[str, float | int | str]] = []
    for token_count in args.token_counts:
        source = (
            torch.randn(
                (token_count, args.hidden_size),
                dtype=torch.float32,
                device="npu",
            )
            * 0.02
        ).to(torch.bfloat16)
        replay_source = (
            torch.randn(
                (token_count, args.hidden_size),
                dtype=torch.float32,
                device="npu",
            )
            * 0.02
        ).to(torch.bfloat16)

        def nd_bias() -> torch.Tensor:
            return F.linear(source, weight_nd, bias)

        def nz_bias() -> torch.Tensor:
            return F.linear(source, weight_nz, bias)

        def nz_split_bias() -> torch.Tensor:
            return F.linear(source, weight_nz) + bias

        for name, operation in (
            ("nd_bias", nd_bias),
            ("nz_bias", nz_bias),
            ("nz_split_bias", nz_split_bias),
        ):
            try:
                eager_latency_us = _measure_eager(
                    operation,
                    args.warmup_steps,
                    args.profile_steps,
                )
                graph, output = _capture(operation)
                graph_latency_us = _measure_graph(
                    graph,
                    args.warmup_steps,
                    args.profile_steps,
                )
                source.copy_(replay_source)
                graph.replay()
                torch.npu.synchronize()
                expected = F.linear(source, weight_nd, bias)
                torch.npu.synchronize()
                replay_error = _max_abs_error(output, expected)
                status = "ok"
            except Exception as error:
                eager_latency_us = float("nan")
                graph_latency_us = float("nan")
                replay_error = float("nan")
                status = f"{type(error).__name__}: {error}"
            results.append(
                {
                    "token_count": token_count,
                    "hidden_size": args.hidden_size,
                    "qkv_size_per_rank": args.qkv_size_per_rank,
                    "strategy": name,
                    "eager_latency_microseconds": eager_latency_us,
                    "graph_latency_microseconds": graph_latency_us,
                    "changed_input_replay_max_abs_error": replay_error,
                    "status": status,
                }
            )
        del source, replay_source

    payload = {"device": 0, "dtype": "bfloat16", "results": results}
    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

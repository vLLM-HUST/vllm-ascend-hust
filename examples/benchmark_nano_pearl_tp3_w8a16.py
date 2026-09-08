# SPDX-License-Identifier: Apache-2.0
"""Benchmark W8A16 linear paths for Qwen2.5-14B TP3 matrix shapes."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


TP3_PROJECTIONS = (
    ("qkv", 5120, 2688, False, True),
    ("gate_up", 5120, 9216, True, False),
    ("attention_out", 1920, 5120, True, False),
    ("down", 4608, 5120, True, False),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    difference = actual_float - expected_float
    expected_rms = expected_float.square().mean().sqrt().clamp_min(1e-12)
    cosine = F.cosine_similarity(actual_float.flatten(), expected_float.flatten(), dim=0)
    return {
        "max_abs_error": float(difference.abs().max().cpu()),
        "mean_abs_error": float(difference.abs().mean().cpu()),
        "relative_rms_error": float((difference.square().mean().sqrt() / expected_rms).cpu()),
        "cosine_similarity": float(cosine.cpu()),
    }


def main() -> None:
    args = _parse_args()
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(20260810)
    results: list[dict[str, float | int | str | bool]] = []

    for projection, input_size, output_size, baseline_nz, use_bias in TP3_PROJECTIONS:
        weight = (
            torch.randn(output_size, input_size, dtype=torch.float32, device="npu") * 0.02
        ).to(torch.bfloat16)
        scale = weight.float().abs().amax(dim=1).div(127).clamp_min(1e-8)
        quant_weight = (
            weight.float().div(scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        )
        quant_weight_nz = torch_npu.npu_format_cast(quant_weight.contiguous(), 29)
        antiquant_scale = scale.to(torch.bfloat16).unsqueeze(1)
        baseline_weight = torch_npu.npu_format_cast(weight, 29) if baseline_nz else weight
        baseline_bias = None
        quant_bias = None
        if use_bias:
            baseline_bias = (
                torch.randn(output_size, dtype=torch.float32, device="npu") * 0.02
            ).to(torch.bfloat16)
            quant_bias = baseline_bias.float()

        for token_count in args.token_counts:
            source = (
                torch.randn(token_count, input_size, dtype=torch.float32, device="npu") * 0.02
            ).to(torch.bfloat16)
            replay_source = (
                torch.randn_like(source, dtype=torch.float32) * 0.02
            ).to(torch.bfloat16)

            def baseline() -> torch.Tensor:
                return F.linear(source, baseline_weight, baseline_bias)

            def w8a16_nd() -> torch.Tensor:
                return torch_npu.npu_weight_quant_batchmatmul(
                    source,
                    quant_weight.transpose(-1, -2),
                    antiquant_scale.transpose(-1, -2),
                    bias=quant_bias,
                )

            def w8a16_nz() -> torch.Tensor:
                return torch_npu.npu_weight_quant_batchmatmul(
                    source,
                    quant_weight_nz.transpose(-1, -2),
                    antiquant_scale.transpose(-1, -2),
                    bias=quant_bias,
                )

            graphs: dict[str, tuple[torch.npu.NPUGraph, torch.Tensor]] = {}
            for strategy, operation in (
                ("baseline_bf16", baseline),
                ("w8a16_nd", w8a16_nd),
                ("w8a16_nz", w8a16_nz),
            ):
                try:
                    graph, output = _capture(operation)
                    latency_us = _measure(graph, args.warmup_steps, args.profile_steps)
                    graphs[strategy] = (graph, output)
                    status = "ok"
                except Exception as error:
                    latency_us = math.nan
                    status = f"{type(error).__name__}: {error}"
                results.append(
                    {
                        "projection": projection,
                        "token_count": token_count,
                        "input_size": input_size,
                        "output_size": output_size,
                        "baseline_weight_nz": baseline_nz,
                        "bias": use_bias,
                        "strategy": strategy,
                        "latency_microseconds": latency_us,
                        "status": status,
                    }
                )

            source.copy_(replay_source)
            for graph, _ in graphs.values():
                graph.replay()
            torch.npu.synchronize()
            if "baseline_bf16" in graphs:
                expected = graphs["baseline_bf16"][1]
                for result in results[-3:]:
                    strategy = str(result["strategy"])
                    if strategy in graphs:
                        result.update(_errors(graphs[strategy][1], expected))

    payload = {"activation_dtype": "bfloat16", "weight_dtype": "int8", "results": results}
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

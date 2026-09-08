# SPDX-License-Identifier: Apache-2.0
"""Benchmark production-style weight prefetch for native PEARL matrices."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


MIB = 1024 * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[320, 464, 512])
    parser.add_argument("--prefetch-mib", type=int, nargs="+", default=[4, 8, 12, 18])
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


def _run_case(
    name: str,
    operation_factory: Callable[[int], Callable[[], torch.Tensor]],
    baseline_factory: Callable[[], Callable[[], torch.Tensor]],
    replay_inputs: list[tuple[torch.Tensor, torch.Tensor]],
    prefetch_sizes_mib: list[int],
    warmup_steps: int,
    profile_steps: int,
) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    for prefetch_mib in [0, *prefetch_sizes_mib]:
        try:
            operation = baseline_factory() if prefetch_mib == 0 else operation_factory(prefetch_mib * MIB)
            graph, output = _capture(operation)
            latency_us = _measure(graph, warmup_steps, profile_steps)
            for destination, source in replay_inputs:
                destination.copy_(source)
            graph.replay()
            torch.npu.synchronize()
            expected = baseline_factory()()
            torch.npu.synchronize()
            error = _max_abs_error(output, expected)
            status = "ok"
        except Exception as exception:
            latency_us = float("nan")
            error = float("nan")
            status = f"{type(exception).__name__}: {exception}"
        results.append(
            {
                "projection": name,
                "prefetch_mib": prefetch_mib,
                "latency_microseconds": latency_us,
                "changed_input_replay_max_abs_error": error,
                "status": status,
            }
        )
    return results


def main() -> None:
    args = _parse_args()
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(20260810)
    prefetch_stream = torch.npu.Stream()
    results: list[dict[str, float | int | str]] = []

    for token_count in args.token_counts:
        hidden_size = 5120
        source = (torch.randn(token_count, hidden_size, device="npu") * 0.02).to(torch.bfloat16)
        replay_source = (torch.randn_like(source, dtype=torch.float32) * 0.02).to(torch.bfloat16)
        residual = (torch.randn_like(source, dtype=torch.float32) * 0.02).to(torch.bfloat16)
        replay_residual = (torch.randn_like(source, dtype=torch.float32) * 0.02).to(torch.bfloat16)
        norm_weight = torch.ones(hidden_size, dtype=torch.bfloat16, device="npu")

        def make_prefetched(
            weight: torch.Tensor,
            vector_operation: Callable[[], torch.Tensor],
            dependency: torch.Tensor,
            prefetch_bytes: int,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            calculation_stream = torch.npu.current_stream()
            prefetch_stream.wait_stream(calculation_stream)
            with torch.npu.stream(prefetch_stream):
                torch_npu.npu_prefetch(weight, dependency, prefetch_bytes, 0)
            vector_output = vector_operation()
            calculation_stream.wait_stream(prefetch_stream)
            return F.linear(vector_output, weight, bias)

        gate_up_weight = torch_npu.npu_format_cast(
            (torch.randn(9216, hidden_size, device="npu") * 0.02).to(torch.bfloat16),
            29,
        )

        def gate_up_vector() -> torch.Tensor:
            normalized, _, _ = torch_npu.npu_add_rms_norm(source, residual, norm_weight, 1e-6)
            return normalized

        def gate_up_baseline_factory() -> Callable[[], torch.Tensor]:
            def operation() -> torch.Tensor:
                return F.linear(gate_up_vector(), gate_up_weight)

            return operation

        def gate_up_prefetch_factory(prefetch_bytes: int) -> Callable[[], torch.Tensor]:
            def operation() -> torch.Tensor:
                return make_prefetched(gate_up_weight, gate_up_vector, source, prefetch_bytes)

            return operation

        gate_up_results = _run_case(
            "gate_up",
            gate_up_prefetch_factory,
            gate_up_baseline_factory,
            [(source, replay_source), (residual, replay_residual)],
            args.prefetch_mib,
            args.warmup_steps,
            args.profile_steps,
        )
        results.extend({"token_count": token_count, **result} for result in gate_up_results)
        del gate_up_weight

        gate_up = (torch.randn(token_count, 9216, device="npu") * 0.02).to(torch.bfloat16)
        replay_gate_up = (torch.randn_like(gate_up, dtype=torch.float32) * 0.02).to(torch.bfloat16)
        down_weight = torch_npu.npu_format_cast(
            (torch.randn(hidden_size, 4608, device="npu") * 0.02).to(torch.bfloat16),
            29,
        )

        def down_vector() -> torch.Tensor:
            return torch_npu.npu_swiglu(gate_up)

        def down_baseline_factory() -> Callable[[], torch.Tensor]:
            def operation() -> torch.Tensor:
                return F.linear(down_vector(), down_weight)

            return operation

        def down_prefetch_factory(prefetch_bytes: int) -> Callable[[], torch.Tensor]:
            def operation() -> torch.Tensor:
                return make_prefetched(down_weight, down_vector, gate_up, prefetch_bytes)

            return operation

        down_results = _run_case(
            "down",
            down_prefetch_factory,
            down_baseline_factory,
            [(gate_up, replay_gate_up)],
            args.prefetch_mib,
            args.warmup_steps,
            args.profile_steps,
        )
        results.extend({"token_count": token_count, **result} for result in down_results)
        del down_weight, gate_up, replay_gate_up

        qkv_weight = (torch.randn(2688, hidden_size, device="npu") * 0.02).to(torch.bfloat16)
        qkv_bias = (torch.randn(2688, device="npu") * 0.02).to(torch.bfloat16)

        def qkv_vector() -> torch.Tensor:
            normalized, _, _ = torch_npu.npu_add_rms_norm(source, residual, norm_weight, 1e-6)
            return normalized

        def qkv_baseline_factory() -> Callable[[], torch.Tensor]:
            def operation() -> torch.Tensor:
                return F.linear(qkv_vector(), qkv_weight, qkv_bias)

            return operation

        def qkv_prefetch_factory(prefetch_bytes: int) -> Callable[[], torch.Tensor]:
            def operation() -> torch.Tensor:
                return make_prefetched(qkv_weight, qkv_vector, source, prefetch_bytes, qkv_bias)

            return operation

        qkv_results = _run_case(
            "qkv",
            qkv_prefetch_factory,
            qkv_baseline_factory,
            [(source, replay_source), (residual, replay_residual)],
            args.prefetch_mib,
            args.warmup_steps,
            args.profile_steps,
        )
        results.extend({"token_count": token_count, **result} for result in qkv_results)
        del qkv_weight, qkv_bias, source, replay_source, residual, replay_residual

    payload = {"dtype": "bfloat16", "results": results}
    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Benchmark token-chunked matmul/all-reduce overlap for native TP3 PEARL."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[320, 464, 512])
    parser.add_argument("--input-sizes", type=int, nargs="+", default=[1920, 4608])
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--chunk-counts", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _capture(operation: Callable[[], Any]) -> tuple[torch.npu.NPUGraph, Any]:
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
    os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 3:
        raise ValueError(f"This benchmark requires TP3, received TP={world_size}.")
    torch.npu.config.allow_internal_format = True

    results: list[dict[str, float | int | str]] = []
    for token_count in args.token_counts:
        for input_size in args.input_sizes:
            torch.manual_seed(20260810 + rank)
            source = (torch.randn(token_count, input_size, device="npu") * 0.02).to(torch.bfloat16)
            replay_source = (torch.randn_like(source, dtype=torch.float32) * 0.02).to(torch.bfloat16)
            weight = torch_npu.npu_format_cast(
                (torch.randn(args.hidden_size, input_size, device="npu") * 0.02).to(torch.bfloat16),
                29,
            )

            def baseline() -> torch.Tensor:
                output = F.linear(source, weight)
                dist.all_reduce(output)
                return output

            dist.barrier()
            baseline_graph, baseline_output = _capture(baseline)
            baseline_latency_us = _measure(
                baseline_graph,
                args.warmup_steps,
                args.profile_steps,
            )
            source.copy_(replay_source)
            baseline_graph.replay()
            torch.npu.synchronize()
            expected = baseline_output.cpu()
            local_results: list[dict[str, float | int | str]] = [{
                "rank": rank,
                "token_count": token_count,
                "input_size": input_size,
                "hidden_size": args.hidden_size,
                "strategy": "nz_matmul_allreduce",
                "chunk_count": 1,
                "latency_microseconds": baseline_latency_us,
                "changed_input_replay_max_abs_error": 0.0,
                "status": "ok",
            }]

            for chunk_count in args.chunk_counts:
                source.copy_(replay_source)
                communication_stream = torch.npu.Stream()

                def overlapped() -> torch.Tensor:
                    calculation_stream = torch.npu.current_stream()
                    outputs: list[torch.Tensor] = []
                    for source_chunk in source.chunk(chunk_count, dim=0):
                        projected = F.linear(source_chunk, weight)
                        communication_stream.wait_stream(calculation_stream)
                        with torch.npu.stream(communication_stream):
                            dist.all_reduce(projected)
                        outputs.append(projected)
                    calculation_stream.wait_stream(communication_stream)
                    return torch.cat(outputs, dim=0)

                dist.barrier()
                try:
                    overlap_graph, overlap_output = _capture(overlapped)
                    overlap_latency_us = _measure(
                        overlap_graph,
                        args.warmup_steps,
                        args.profile_steps,
                    )
                    source.copy_(replay_source)
                    overlap_graph.replay()
                    torch.npu.synchronize()
                    error = _max_abs_error(overlap_output.cpu(), expected)
                    status = "ok"
                except Exception as exception:
                    overlap_latency_us = float("nan")
                    error = float("nan")
                    status = f"{type(exception).__name__}: {exception}"
                    print(
                        f"rank={rank} tokens={token_count} input={input_size} "
                        f"chunks={chunk_count}: {status}",
                        flush=True,
                    )
                local_results.append({
                    "rank": rank,
                    "token_count": token_count,
                    "input_size": input_size,
                    "hidden_size": args.hidden_size,
                    "strategy": "chunked_nz_matmul_allreduce_overlap",
                    "chunk_count": chunk_count,
                    "latency_microseconds": overlap_latency_us,
                    "changed_input_replay_max_abs_error": error,
                    "status": status,
                })

            gathered: list[list[dict[str, float | int | str]] | None] = [None] * world_size
            dist.all_gather_object(gathered, local_results)
            if rank == 0:
                for rank_results in gathered:
                    if rank_results is not None:
                        results.extend(rank_results)

            del source, replay_source, weight, baseline_output, expected

    if rank == 0:
        rendered = json.dumps({"world_size": world_size, "results": results}, indent=2)
        print(rendered)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

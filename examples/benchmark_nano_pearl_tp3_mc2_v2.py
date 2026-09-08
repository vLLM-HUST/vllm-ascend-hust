# SPDX-License-Identifier: Apache-2.0
"""Validate CANN MC2-v2 BF16/NZ kernels with a three-rank TP group."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-count", type=int, default=513)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--row-input-sizes", type=int, nargs="+", default=[1920, 4608])
    parser.add_argument("--column-output-sizes", type=int, nargs="+", default=[2560, 9216])
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=100)
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


def _format_nz(matrix: torch.Tensor) -> torch.Tensor:
    return torch_npu.npu_format_cast(matrix.contiguous(), 29)


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().cpu())


def _benchmark(
    name: str,
    operation: Callable[[], Any],
    reference: Callable[[Any], tuple[float, str]],
    warmup_steps: int,
    profile_steps: int,
) -> dict[str, float | str]:
    try:
        graph, output = _capture(operation)
        latency_us = _measure(graph, warmup_steps, profile_steps)
        error, reference_status = reference(output)
        status = "ok" if reference_status == "ok" else reference_status
    except Exception as exception:
        latency_us = float("nan")
        error = float("nan")
        status = f"{type(exception).__name__}: {exception}"
    return {
        "strategy": name,
        "latency_microseconds": latency_us,
        "max_abs_error": error,
        "status": status,
    }


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 3:
        raise ValueError(f"This benchmark requires TP3, received TP={world_size}.")
    if args.token_count % world_size:
        raise ValueError("token-count must be divisible by three for reduce-scatter.")
    torch.npu.config.allow_internal_format = True

    backend = dist.group.WORLD._get_backend(torch.device("npu"))
    hccl_group_name = backend.get_hccl_comm_name(rank)
    local_token_count = args.token_count // world_size
    results: list[dict[str, float | int | str]] = []

    for input_size in args.row_input_sizes:
        torch.manual_seed(1009 + rank)
        source = (torch.randn(args.token_count, input_size, device="npu") * 0.02).to(torch.bfloat16)
        weight_nd = (torch.randn(args.hidden_size, input_size, device="npu") * 0.02).to(torch.bfloat16)
        weight_nz = _format_nz(weight_nd)
        weight_nd_transposed = weight_nd.t().contiguous()
        residual = (torch.randn(args.token_count, args.hidden_size, device="npu") * 0.02).to(torch.bfloat16)
        dist.broadcast(residual, src=0)
        local_residual = residual.chunk(world_size, dim=0)[rank].contiguous()
        gamma = torch.ones(args.hidden_size, dtype=torch.bfloat16, device="npu")

        def baseline() -> tuple[torch.Tensor, torch.Tensor]:
            projected = F.linear(source, weight_nz)
            dist.all_reduce(projected)
            normalized, _, add_out = torch_npu.npu_add_rms_norm(projected, residual, gamma, 1e-6)
            return normalized, add_out

        dist.barrier()
        baseline_graph, baseline_output = _capture(baseline)
        baseline_latency = _measure(baseline_graph, args.warmup_steps, args.profile_steps)
        expected_normalized = baseline_output[0].cpu()
        expected_add_out = baseline_output[1].chunk(world_size, dim=0)[rank].contiguous().cpu()
        results.append({
            "rank": rank,
            "kind": "row",
            "input_size": input_size,
            "output_size": args.hidden_size,
            "strategy": "nz_matmul_allreduce_add_rmsnorm",
            "latency_microseconds": baseline_latency,
            "max_abs_error": 0.0,
            "status": "ok",
        })

        def mmrs_v2() -> tuple[torch.Tensor, torch.Tensor]:
            reduced, _ = torch_npu.npu_quant_mm_reduce_scatter(
                source,
                weight_nd_transposed,
                hccl_group_name,
                world_size,
            )
            normalized, _, add_out = torch_npu.npu_add_rms_norm(reduced, local_residual, gamma, 1e-6)
            gathered = torch.empty(
                (args.token_count, args.hidden_size),
                dtype=normalized.dtype,
                device=normalized.device,
            )
            dist.all_gather_into_tensor(gathered, normalized)
            return gathered, add_out

        def mmrs_base() -> tuple[torch.Tensor, torch.Tensor]:
            reduced = torch_npu.npu_mm_reduce_scatter_base(
                source,
                weight_nd_transposed,
                hccl_group_name,
                world_size,
                reduce_op="sum",
                bias=None,
                comm_turn=0,
                comm_mode="aiv",
            )
            normalized, _, add_out = torch_npu.npu_add_rms_norm(
                reduced,
                local_residual,
                gamma,
                1e-6,
            )
            gathered = torch.empty(
                (args.token_count, args.hidden_size),
                dtype=normalized.dtype,
                device=normalized.device,
            )
            dist.all_gather_into_tensor(gathered, normalized)
            return gathered, add_out

        def check_mmrs(output: tuple[torch.Tensor, torch.Tensor]) -> tuple[float, str]:
            normalized_error = _max_abs_error(output[0].cpu(), expected_normalized)
            add_out_error = _max_abs_error(output[1].cpu(), expected_add_out)
            return max(normalized_error, add_out_error), "ok"

        dist.barrier()
        mmrs_result = _benchmark(
            "nd_matmul_reduce_scatter_v2_add_rmsnorm_allgather",
            mmrs_v2,
            check_mmrs,
            args.warmup_steps,
            args.profile_steps,
        )
        results.append({
            "rank": rank,
            "kind": "row",
            "input_size": input_size,
            "output_size": args.hidden_size,
            **mmrs_result,
        })

        dist.barrier()
        mmrs_base_result = _benchmark(
            "nd_matmul_reduce_scatter_base_add_rmsnorm_allgather",
            mmrs_base,
            check_mmrs,
            args.warmup_steps,
            args.profile_steps,
        )
        results.append({
            "rank": rank,
            "kind": "row",
            "input_size": input_size,
            "output_size": args.hidden_size,
            **mmrs_base_result,
        })

    for output_size in args.column_output_sizes:
        torch.manual_seed(2027 + rank)
        global_source = (torch.randn(args.token_count, args.hidden_size, device="npu") * 0.02).to(torch.bfloat16)
        dist.broadcast(global_source, src=0)
        local_source = global_source.chunk(world_size, dim=0)[rank].contiguous()
        weight_nd = (torch.randn(output_size, args.hidden_size, device="npu") * 0.02).to(torch.bfloat16)
        weight_nz_transposed = _format_nz(weight_nd.t())
        weight_nd_transposed = weight_nd.t().contiguous()

        def baseline_column() -> torch.Tensor:
            gathered = torch.empty_like(global_source)
            dist.all_gather_into_tensor(gathered, local_source)
            return torch.matmul(gathered, weight_nz_transposed)

        dist.barrier()
        baseline_graph, baseline_output = _capture(baseline_column)
        baseline_latency = _measure(baseline_graph, args.warmup_steps, args.profile_steps)
        expected = baseline_output.cpu()
        results.append({
            "rank": rank,
            "kind": "column",
            "input_size": args.hidden_size,
            "output_size": output_size,
            "strategy": "allgather_nz_matmul",
            "latency_microseconds": baseline_latency,
            "max_abs_error": 0.0,
            "status": "ok",
        })

        def agmm_v2() -> torch.Tensor:
            output, _, _ = torch_npu.npu_all_gather_quant_mm(
                local_source,
                weight_nd_transposed,
                hccl_group_name,
                world_size,
                gather_output=False,
            )
            return output

        def check_agmm(output: torch.Tensor) -> tuple[float, str]:
            return _max_abs_error(output.cpu(), expected), "ok"

        dist.barrier()
        agmm_result = _benchmark(
            "allgather_matmul_v2_nd",
            agmm_v2,
            check_agmm,
            args.warmup_steps,
            args.profile_steps,
        )
        results.append({
            "rank": rank,
            "kind": "column",
            "input_size": args.hidden_size,
            "output_size": output_size,
            **agmm_result,
        })

    gathered_results: list[list[dict[str, float | int | str]] | None] = [None] * world_size
    dist.all_gather_object(gathered_results, results)
    if rank == 0:
        print(json.dumps({"world_size": world_size, "results": gathered_results}, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

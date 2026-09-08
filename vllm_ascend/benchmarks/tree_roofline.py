# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


class TreeRooflineComplete(RuntimeError):
    pass


_completed = False


def _memory_allocated() -> int:
    memory_allocated = getattr(torch.npu, "memory_allocated", None)
    return int(memory_allocated()) if memory_allocated is not None else 0


def _max_memory_allocated() -> int:
    max_memory_allocated = getattr(torch.npu, "max_memory_allocated", None)
    return int(max_memory_allocated()) if max_memory_allocated is not None else 0


def maybe_run_tree_roofline(
    run_target_forward: Callable[[], Any],
    scheduler_output: Any,
    cudagraph_mode: Any,
    run_target_with_logits: Callable[[], Any] | None = None,
) -> None:
    global _completed
    if _completed:
        return

    config_path = os.getenv("VLLM_ASCEND_TREE_ROOFLINE_CONFIG")
    if not config_path:
        return
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    measured_forward = run_target_forward
    if config.get("include_logits", False):
        if run_target_with_logits is None:
            raise RuntimeError("Full target verification requested without logits callback")
        measured_forward = run_target_with_logits
    expected_nodes = [int(value) for value in config["nodes_per_request"]]
    actual_nodes = [
        int(scheduler_output.num_scheduled_tokens[req_id])
        for req_id in scheduler_output.num_scheduled_tokens
    ]
    if actual_nodes != expected_nodes:
        return

    _completed = True
    warmup_iterations = int(config.get("warmup_iterations", 30))
    measured_iterations = int(config.get("measured_iterations", 200))
    for _ in range(warmup_iterations):
        measured_forward()
    torch.npu.synchronize()

    latencies_ms: list[float] = []
    memory_before = _memory_allocated()
    timing_method = config.get("timing_method", "host_sync")
    if timing_method == "npu_event_sync":
        for _ in range(measured_iterations):
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            start_event.record()
            measured_forward()
            end_event.record()
            torch.npu.synchronize()
            latencies_ms.append(float(start_event.elapsed_time(end_event)))
    elif timing_method == "host_sync":
        for _ in range(measured_iterations):
            torch.npu.synchronize()
            started = time.perf_counter_ns()
            measured_forward()
            torch.npu.synchronize()
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    else:
        raise ValueError(f"Unsupported timing method: {timing_method}")

    payload = {
        **config,
        "actual_nodes_per_request": actual_nodes,
        "actual_total_nodes": sum(actual_nodes),
        "latencies_ms": latencies_ms,
        "cudagraph_mode": cudagraph_mode.name,
        "timed_components": (
            ["transformer", "lm_head", "logits"]
            if config.get("include_logits", False)
            else ["transformer"]
        ),
        "memory_allocated_bytes": memory_before,
        "max_memory_allocated_bytes": _max_memory_allocated(),
    }
    output_path = Path(config["output_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    raise TreeRooflineComplete(f"tree Roofline result written to {output_path}")

# SPDX-License-Identifier: Apache-2.0
"""Default-off per-step/per-rank GMM1 group_list and batch-row diagnostic tracer.

Activated ONLY when ``VLLM_HUST_GMM_ROUTE_TRACE_DIR`` is set to a non-empty path.
When unset, this module is never imported by the hot path, allocates no device
memory, and performs no IO.

Graph-safety: ``record_group_list`` writes into a pre-allocated static NPU
tensor ``(num_moe_layers, local_experts)`` via in-place ``copy_``, which is
captured inside ``FULL_DECODE_ONLY`` NPU Graphs and executes on every replay.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

ENV_TRACE_DIR = "VLLM_HUST_GMM_ROUTE_TRACE_DIR"
ENV_WINDOW_ID = "VLLM_HUST_GMM_ROUTE_TRACE_WINDOW_ID"

_ACTIVE_TRACER: GmmRouteTracer | None = None


def trace_dir_from_env() -> Path | None:
    raw = os.environ.get(ENV_TRACE_DIR, "").strip()
    return Path(raw) if raw else None


def get_active_tracer() -> GmmRouteTracer | None:
    return _ACTIVE_TRACER


def reset_active_tracer() -> None:
    global _ACTIVE_TRACER
    _ACTIVE_TRACER = None


def extract_static_ep_map(expert_map: torch.Tensor | None, num_experts: int) -> tuple[int, dict[str, int]]:
    """Build static ``log2phy`` mapping ``{global_expert_id_str: local_slot}``."""
    if expert_map is None:
        return num_experts, {str(i): i for i in range(num_experts)}
    values = expert_map.detach().cpu().tolist()
    if len(values) != num_experts:
        raise ValueError(f"ascend_expert_map length {len(values)} != num_experts {num_experts}")
    log2phy: dict[str, int] = {}
    for global_id, slot in enumerate(values):
        slot_int = int(slot)
        if slot_int >= 0:
            log2phy[str(global_id)] = slot_int
    if not log2phy:
        raise ValueError("ascend_expert_map contains no local experts")
    local_experts = len(log2phy)
    if set(log2phy.values()) != set(range(local_experts)):
        raise ValueError(f"local slots not contiguous 0..{local_experts - 1}")
    return local_experts, log2phy


class GmmRouteTracer:
    """Captures per-step batch rows, phase, EP map, and per-layer type-1 group_list."""

    def __init__(
        self,
        *,
        trace_dir: Path,
        window_id: str,
        physical_rank: int,
        logical_device: int,
        moe_layers: int,
        num_experts: int,
        local_experts: int,
        log2phy: dict[str, int],
        layer_id_to_index: dict[int, int],
        layer_obj_to_index: dict[int, int] | None = None,
        device: torch.device,
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.window_id = window_id
        self.physical_rank = int(physical_rank)
        self.logical_device = int(logical_device)
        self.moe_layers = int(moe_layers)
        self.num_experts = int(num_experts)
        self.local_experts = int(local_experts)
        self.log2phy = dict(log2phy)
        self.layer_id_to_index = dict(layer_id_to_index)
        self.layer_obj_to_index = dict(layer_obj_to_index or {})
        self.device = device

        self.group_list_buffer = torch.zeros(
            (self.moe_layers, self.local_experts),
            dtype=torch.int64,
            device=self.device,
        )
        self.layer_group_list_types: list[int] = [-1] * self.moe_layers
        self._step_counter = 0
        self._steps_path = self.trace_dir / f"steps_rank{self.physical_rank}.jsonl"

        ep_map_path = self.trace_dir / f"ep_map_rank{self.physical_rank}.json"
        ep_map_path.write_text(
            json.dumps(
                {
                    "window_id": self.window_id,
                    "physical_rank": self.physical_rank,
                    "logical_device": self.logical_device,
                    "moe_layers": self.moe_layers,
                    "num_experts": self.num_experts,
                    "local_experts": self.local_experts,
                    "log2phy": self.log2phy,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def init_for_model(
        cls,
        model: Any,
        *,
        physical_rank: int,
        logical_device: int,
        device: torch.device,
        trace_dir: Path | None = None,
        window_id: str | None = None,
    ) -> GmmRouteTracer | None:
        global _ACTIVE_TRACER
        target_dir = trace_dir if trace_dir is not None else trace_dir_from_env()
        if target_dir is None:
            return None
        resolved_window = window_id or os.environ.get(ENV_WINDOW_ID, "").strip() or target_dir.name

        moe_modules: list[tuple[int, Any]] = []
        for mod in model.modules():
            if hasattr(mod, "ascend_expert_map") and hasattr(mod, "moe_config"):
                lid = int(getattr(mod, "layer_id", len(moe_modules)))
                moe_modules.append((lid, mod))
        if not moe_modules:
            return None
        moe_modules.sort(key=lambda item: item[0])

        first_mod = moe_modules[0][1]
        num_experts = int(first_mod.moe_config.num_experts)
        local_experts, log2phy = extract_static_ep_map(first_mod.ascend_expert_map, num_experts)
        layer_id_to_index: dict[int, int] = {}
        layer_obj_to_index: dict[int, int] = {}
        for idx, (lid, mod) in enumerate(moe_modules):
            mod_local, mod_map = extract_static_ep_map(mod.ascend_expert_map, num_experts)
            if mod_local != local_experts or mod_map != log2phy:
                raise ValueError(f"Non-static EP map across MoE layers at layer_id={lid}")
            layer_id_to_index[lid] = idx
            layer_obj_to_index[id(mod)] = idx

        tracer = cls(
            trace_dir=target_dir,
            window_id=resolved_window,
            physical_rank=physical_rank,
            logical_device=logical_device,
            moe_layers=len(moe_modules),
            num_experts=num_experts,
            local_experts=local_experts,
            log2phy=log2phy,
            layer_id_to_index=layer_id_to_index,
            layer_obj_to_index=layer_obj_to_index,
            device=device,
        )
        _ACTIVE_TRACER = tracer
        return tracer

    def record_group_list(self, layer: Any, group_list: torch.Tensor, group_list_type: int) -> None:
        idx = self.layer_obj_to_index.get(id(layer))
        if idx is None:
            lid = int(getattr(layer, "layer_id", -1))
            idx = self.layer_id_to_index.get(lid)
        if idx is None:
            return
        if group_list.ndim != 1 or int(group_list.shape[0]) != self.local_experts:
            raise ValueError(f"group_list shape {tuple(group_list.shape)} != ({self.local_experts},) at layer_id={lid}")
        self.group_list_buffer[idx].copy_(group_list.to(dtype=torch.int64))
        self.layer_group_list_types[idx] = int(group_list_type)

    def begin_step(self, *, is_capturing: bool = False) -> None:
        if not is_capturing:
            self.group_list_buffer.zero_()

    def end_step(self, *, phase: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        step_id = self._step_counter
        self._step_counter += 1
        counts_cpu = self.group_list_buffer.detach().cpu().tolist()
        group_lists = [
            {
                "window_id": self.window_id,
                "step_id": step_id,
                "layer": idx,
                "physical_rank": self.physical_rank,
                "group_list_type": int(self.layer_group_list_types[idx]),
                "counts": [int(c) for c in counts_cpu[idx]],
            }
            for idx in range(self.moe_layers)
        ]
        record = {
            "window_id": self.window_id,
            "step_id": step_id,
            "phase": phase,
            "physical_rank": self.physical_rank,
            "rows": rows,
            "group_lists": group_lists,
        }
        with self._steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def record_gmm1_group_list(layer: Any, group_list: torch.Tensor, group_list_type: int) -> None:
    tracer = _ACTIVE_TRACER
    if tracer is not None:
        tracer.record_group_list(layer, group_list, group_list_type)


def classify_step_phase(
    num_computed_tokens: list[int],
    num_prompt_tokens: list[int],
) -> str:
    if not num_computed_tokens:
        return "warmup"
    is_prefill = [c < p for c, p in zip(num_computed_tokens, num_prompt_tokens)]
    if all(is_prefill):
        return "prefill"
    if not any(is_prefill):
        return "decode"
    return "mixed"


def build_step_rows(
    req_ids: list[str],
    num_scheduled_by_req: dict[str, int],
    num_computed_tokens: list[int],
    num_tokens_padded: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for req_id, start_pos in zip(req_ids, num_computed_tokens):
        n_sched = int(num_scheduled_by_req.get(req_id, 0))
        for offset in range(n_sched):
            rows.append(
                {
                    "request_id": req_id,
                    "token_position": int(start_pos) + offset,
                    "kind": "forwarded",
                }
            )
    pad_count = int(num_tokens_padded) - len(rows)
    if pad_count < 0:
        raise ValueError(f"num_tokens_padded ({num_tokens_padded}) < forwarded rows ({len(rows)})")
    for _ in range(pad_count):
        rows.append({"request_id": None, "token_position": None, "kind": "padded"})
    return rows


def assemble_reconciliation_payload(
    *,
    trace_dir: Path,
    host_commits: dict[str, str],
    requests: list[dict[str, Any]],
    top_k: int,
    rank_device: dict[str, int],
    in_cohort_phases: tuple[str, ...] = ("prefill", "decode", "mixed"),
) -> dict[str, Any]:
    """Merge per-rank trace files into the gmm-route-reconciliation schema."""
    trace_path = Path(trace_dir)
    ep_files = sorted(trace_path.glob("ep_map_rank*.json"))
    if not ep_files:
        raise FileNotFoundError(f"No ep_map_rank*.json in {trace_path}")
    ep_records = [json.loads(p.read_text(encoding="utf-8")) for p in ep_files]
    window_id = ep_records[0]["window_id"]
    moe_layers = int(ep_records[0]["moe_layers"])
    num_experts = int(ep_records[0]["num_experts"])

    ranks_out = [
        {
            "physical_rank": int(r["physical_rank"]),
            "local_experts": int(r["local_experts"]),
            "log2phy": r["log2phy"],
        }
        for r in sorted(ep_records, key=lambda item: int(item["physical_rank"]))
    ]

    steps_by_id: dict[int, dict[str, Any]] = {}
    group_lists: list[dict[str, Any]] = []
    for rank_info in ranks_out:
        rank = rank_info["physical_rank"]
        step_file = trace_path / f"steps_rank{rank}.jsonl"
        if not step_file.is_file():
            raise FileNotFoundError(f"Missing {step_file}")
        for line in step_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = int(rec["step_id"])
            step_obj = {
                "window_id": rec["window_id"],
                "step_id": sid,
                "phase": rec["phase"],
                "rows": rec["rows"],
            }
            if sid not in steps_by_id:
                steps_by_id[sid] = step_obj
            elif steps_by_id[sid] != step_obj:
                raise ValueError(f"Cross-rank step metadata mismatch at step_id={sid}")
            group_lists.extend(rec["group_lists"])

    return {
        "window": {
            "window_id": window_id,
            "host_commits": dict(host_commits),
            "moe_layers": moe_layers,
            "num_experts": num_experts,
            "top_k": int(top_k),
            "ep_size": len(ranks_out),
            "in_cohort_phases": list(in_cohort_phases),
            "rank_device": dict(rank_device),
        },
        "requests": requests,
        "ep_map": {
            "window_id": window_id,
            "host_commits": dict(host_commits),
            "ranks": ranks_out,
        },
        "steps": [steps_by_id[sid] for sid in sorted(steps_by_id)],
        "group_lists": group_lists,
    }

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for default-off GMM1 route reconciliation tracer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe.route_trace import (
    ENV_TRACE_DIR,
    GmmRouteTracer,
    assemble_reconciliation_payload,
    build_step_rows,
    classify_step_phase,
    extract_static_ep_map,
    reset_active_tracer,
)

HOSTS = {
    "vllm": "f18cf803c5f63625e2c71253ddaf8b0bad0bad1a",
    "vllm-ascend": "320eced8c1bba71c0329bf59caa721a2382e7499",
}


class _DummyMoELayer(torch.nn.Module):
    def __init__(self, layer_id: int, num_experts: int, expert_map: torch.Tensor) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.moe_config = SimpleNamespace(num_experts=num_experts)
        self.ascend_expert_map = expert_map


class _DummyModel(torch.nn.Module):
    def __init__(self, layers: list[_DummyMoELayer]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


def _load_reconcile():
    reconcile_path = Path(
        "/root/vllm-workspace/vllm-ascend-split-batch-hust/tools/analysis/gmm_route_reconciliation.py"
    )
    if not reconcile_path.is_file():
        pytest.skip("gmm_route_reconciliation.py not present")
    spec = importlib.util.spec_from_file_location("gmm_route_reconciliation", reconcile_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_off_returns_none(monkeypatch):
    monkeypatch.delenv(ENV_TRACE_DIR, raising=False)
    reset_active_tracer()
    model = _DummyModel([_DummyMoELayer(0, 4, torch.tensor([0, 1, -1, -1]))])
    assert GmmRouteTracer.init_for_model(model, physical_rank=0, logical_device=0, device=torch.device("cpu")) is None


def test_extract_static_ep_map():
    local, mapping = extract_static_ep_map(torch.tensor([-1, 0, -1, 1]), 4)
    assert local == 2
    assert mapping == {"1": 0, "3": 1}


def test_classify_step_phase_and_rows():
    assert classify_step_phase([0, 0], [4, 5]) == "prefill"
    assert classify_step_phase([4, 5], [4, 5]) == "decode"
    assert classify_step_phase([0, 5], [4, 5]) == "mixed"
    rows = build_step_rows(["r0", "r1"], {"r0": 2, "r1": 1}, [0, 3], num_tokens_padded=5)
    assert rows == [
        {"request_id": "r0", "token_position": 0, "kind": "forwarded"},
        {"request_id": "r0", "token_position": 1, "kind": "forwarded"},
        {"request_id": "r1", "token_position": 3, "kind": "forwarded"},
        {"request_id": None, "token_position": None, "kind": "padded"},
        {"request_id": None, "token_position": None, "kind": "padded"},
    ]


def test_two_rank_trace_reconciles_end_to_end(tmp_path):
    recon = _load_reconcile()
    reset_active_tracer()

    # 2 layers, 4 global experts, 2 EP ranks (rank 0: experts 0,1; rank 1: experts 2,3), top_k=2
    map_r0 = torch.tensor([0, 1, -1, -1])
    map_r1 = torch.tensor([-1, -1, 0, 1])
    m0 = _DummyModel([_DummyMoELayer(0, 4, map_r0), _DummyMoELayer(1, 4, map_r0)])
    m1 = _DummyModel([_DummyMoELayer(0, 4, map_r1), _DummyMoELayer(1, 4, map_r1)])

    t0 = GmmRouteTracer.init_for_model(
        m0, physical_rank=0, logical_device=0, device=torch.device("cpu"), trace_dir=tmp_path, window_id="w-ut-0"
    )
    t1 = GmmRouteTracer.init_for_model(
        m1, physical_rank=1, logical_device=1, device=torch.device("cpu"), trace_dir=tmp_path, window_id="w-ut-0"
    )
    assert t0 is not None and t1 is not None

    # Step 0: warmup (excluded from cohort)
    for t, m in ((t0, m0), (t1, m1)):
        t.begin_step()
        for layer in m.layers:
            t.record_group_list(layer, torch.tensor([1, 1]), 1)
        t.end_step(phase="warmup", rows=[{"request_id": None, "token_position": None, "kind": "padded"}])

    # Step 1: prefill for R0 (2 prompt tokens, 0 padded)
    # token 0 -> layer0:[0,2], layer1:[0,1] (rank 1 gets [0,0] on layer 1 -> zero-routed rank test!)
    # token 1 -> layer0:[1,3], layer1:[0,1]
    rows_s1 = build_step_rows(["R0"], {"R0": 2}, [0], num_tokens_padded=2)
    t0.begin_step()
    t0.record_group_list(m0.layers[0], torch.tensor([1, 1]), 1)
    t0.record_group_list(m0.layers[1], torch.tensor([2, 2]), 1)
    t0.end_step(phase="prefill", rows=rows_s1)

    t1.begin_step()
    t1.record_group_list(m1.layers[0], torch.tensor([1, 1]), 1)
    t1.record_group_list(m1.layers[1], torch.tensor([0, 0]), 1)
    t1.end_step(phase="prefill", rows=rows_s1)

    # Step 2: decode for R0 (token_position=2, padded to 2 -> 1 padded row, must NOT add counts)
    # token 2 -> layer0:[2,3], layer1:[1,2]
    rows_s2 = build_step_rows(["R0"], {"R0": 1}, [2], num_tokens_padded=2)
    t0.begin_step()
    t0.record_group_list(m0.layers[0], torch.tensor([0, 0]), 1)
    t0.record_group_list(m0.layers[1], torch.tensor([0, 1]), 1)
    t0.end_step(phase="decode", rows=rows_s2)

    t1.begin_step()
    t1.record_group_list(m1.layers[0], torch.tensor([1, 1]), 1)
    t1.record_group_list(m1.layers[1], torch.tensor([1, 0]), 1)
    t1.end_step(phase="decode", rows=rows_s2)

    requests = [
        {
            "window_id": "w-ut-0",
            "request_id": "R0",
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "routed_experts": [
                [[0, 2], [0, 1]],
                [[1, 3], [0, 1]],
                [[2, 3], [1, 2]],
            ],
        }
    ]
    payload = assemble_reconciliation_payload(
        trace_dir=tmp_path,
        host_commits=HOSTS,
        requests=requests,
        top_k=2,
        rank_device={"0": 2, "1": 3},
    )
    summary = recon.reconcile(payload)
    assert summary["status"] == "reconciled"
    assert summary["valid_tokens"] == 3
    assert summary["padded_rows"] == 1
    assert summary["out_of_cohort_steps"] == [0]
    assert len(summary["calls"]) == 8

    # Fail-closed check: corrupting one count (e.g. padding leakage) must reject
    payload["group_lists"][-1]["counts"][0] += 1
    with pytest.raises(recon.Rejected):
        recon.reconcile(payload)
    reset_active_tracer()

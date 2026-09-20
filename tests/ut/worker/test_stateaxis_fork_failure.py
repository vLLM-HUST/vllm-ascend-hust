# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "vllm_ascend" / "worker" / "stateaxis_fork_failure.py"
SPEC = importlib.util.spec_from_file_location("stateaxis_fork_failure", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
failure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = failure
SPEC.loader.exec_module(failure)


def test_stage_vocabulary_is_fixed_and_complete() -> None:
    assert failure.FAILURE_STAGES == (
        "source-before-forward",
        "source-after-forward",
        "source-after-sample",
        "child-before-forward",
        "child-after-forward",
        "child-after-sample",
    )
    with pytest.raises(failure.StateAxisFailureArmError, match="stage is invalid"):
        failure.validate_failure_arm("0_request-deadbeef", "after-key-copy")
    with pytest.raises(failure.StateAxisFailureArmError, match="role differs"):
        failure.validate_failure_arm(
            "1_request-deadbeef", "source-before-forward"
        )


def test_injector_is_request_scoped_and_consumed_before_raise() -> None:
    assert issubclass(failure.StateAxisInjectedWorkerFailure, SystemExit)
    assert not issubclass(failure.StateAxisInjectedWorkerFailure, Exception)
    injector = failure.StateAxisForkFailureInjector()
    arm = injector.arm("1_request-deadbeef", "child-after-forward")
    assert arm.phase == "after-forward"
    assert injector.maybe_fail("before-forward", {arm.request_id}) is False
    assert injector.maybe_fail("after-forward", {"unrelated"}) is False

    with pytest.raises(
        failure.StateAxisInjectedWorkerFailure,
        match="child-after-forward",
    ):
        injector.maybe_fail("after-forward", {"source", arm.request_id})

    assert injector.armed is None
    assert injector.maybe_fail("after-forward", {arm.request_id}) is False


def test_duplicate_arm_and_malformed_identity_fail_closed() -> None:
    injector = failure.StateAxisForkFailureInjector()
    injector.arm("0_request-deadbeef", "source-before-forward")
    with pytest.raises(failure.StateAxisFailureArmError, match="already armed"):
        injector.arm("1_request-deadbeef", "child-before-forward")
    with pytest.raises(failure.StateAxisFailureArmError, match="request_id"):
        failure.validate_failure_arm("bad\nid", "source-before-forward")
    with pytest.raises(failure.StateAxisFailureArmError, match="phase is invalid"):
        injector.maybe_fail("unknown", {"0_request-deadbeef"})


def test_rank_receipts_bind_request_without_exposing_it() -> None:
    request_id = "0_request-deadbeef"
    receipts = [
        failure.failure_arm_receipt(
            rank=rank,
            target_rank=2,
            request_id=request_id,
            stage="source-after-sample",
            armed=rank == 2,
        )
        for rank in range(4)
    ]
    assert [receipt["armed"] for receipt in receipts] == [False, False, True, False]
    assert all(
        receipt["request_id_sha256"]
        == hashlib.sha256(request_id.encode()).hexdigest()
        for receipt in receipts
    )
    assert request_id not in repr(receipts)


def test_topology_and_target_rank_are_frozen_to_isolated_tp4() -> None:
    topology = {
        "tensor_parallel_size": 4,
        "world_size": 4,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "speculative": False,
    }
    failure.validate_failure_topology(**topology)
    for rank in range(4):
        failure.validate_target_rank(rank)

    with pytest.raises(failure.StateAxisFailureArmError, match="isolated TP4"):
        failure.validate_failure_topology(
            **{**topology, "pipeline_parallel_size": 2}
        )
    with pytest.raises(failure.StateAxisFailureArmError, match="speculation"):
        failure.validate_failure_topology(**{**topology, "speculative": True})
    with pytest.raises(failure.StateAxisFailureArmError, match="target rank"):
        failure.validate_target_rank(True)


def test_model_runner_wires_all_three_execution_phases_once() -> None:
    path = ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    phases = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "_maybe_inject_stateaxis_fork_failure" or not node.args:
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            phases.append(argument.value)
    assert sorted(phases) == sorted(failure.FAILURE_PHASES)

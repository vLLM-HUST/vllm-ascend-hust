# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, explicitly armed StateAxis worker-failure diagnostics.

This module is dependency-free so its admission and one-shot semantics can be
tested without importing torch_npu. Normal serving never creates an arm.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from dataclasses import dataclass
from typing import Final, Literal

FailurePhase = Literal["before-forward", "after-forward", "after-sample"]
FailureRole = Literal["source", "child"]
FAILURE_ROLES: Final = ("source", "child")
FAILURE_PHASES: Final = ("before-forward", "after-forward", "after-sample")
FAILURE_STAGES: Final = tuple(
    f"{role}-{phase}" for role in FAILURE_ROLES for phase in FAILURE_PHASES
)


class StateAxisFailureArmError(ValueError):
    """The requested diagnostic arm is malformed or conflicts with one pending."""


class StateAxisInjectedWorkerFailure(SystemExit):
    """One deliberately injected failure that escapes the worker RPC loop."""


@dataclass(frozen=True)
class ArmedStateAxisForkFailure:
    request_id: str
    stage: str
    phase: FailurePhase


def validate_failure_arm(request_id: str, stage: str) -> ArmedStateAxisForkFailure:
    if not isinstance(request_id, str) or not request_id or "\n" in request_id:
        raise StateAxisFailureArmError("StateAxis failure request_id is invalid")
    if stage not in FAILURE_STAGES:
        raise StateAxisFailureArmError("StateAxis failure stage is invalid")
    role, phase = stage.split("-", 1)
    assert phase in FAILURE_PHASES
    expected_prefix = "0_" if role == "source" else "1_"
    if not request_id.startswith(expected_prefix):
        raise StateAxisFailureArmError(
            "StateAxis failure request role differs from its stage"
        )
    return ArmedStateAxisForkFailure(request_id, stage, phase)


def validate_failure_topology(
    *,
    tensor_parallel_size: int,
    world_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    prefill_context_parallel_size: int,
    decode_context_parallel_size: int,
    speculative: bool,
) -> None:
    topology = (
        tensor_parallel_size,
        world_size,
        pipeline_parallel_size,
        data_parallel_size,
        prefill_context_parallel_size,
        decode_context_parallel_size,
    )
    if topology != (4, 4, 1, 1, 1, 1):
        raise StateAxisFailureArmError(
            "StateAxis failure injection requires isolated TP4"
        )
    if speculative:
        raise StateAxisFailureArmError(
            "StateAxis failure injection does not support speculation"
        )


def validate_target_rank(target_rank: int) -> None:
    if (
        not isinstance(target_rank, int)
        or isinstance(target_rank, bool)
        or not 0 <= target_rank < 4
    ):
        raise StateAxisFailureArmError("StateAxis failure target rank is invalid")


def failure_arm_receipt(
    *, rank: int, target_rank: int, request_id: str, stage: str, armed: bool
) -> dict[str, object]:
    if not isinstance(rank, int) or rank < 0:
        raise StateAxisFailureArmError("StateAxis failure worker rank is invalid")
    validate_target_rank(target_rank)
    arm = validate_failure_arm(request_id, stage)
    if armed != (rank == target_rank):
        raise StateAxisFailureArmError("StateAxis failure armed flag differs from rank")
    return {
        "schema_version": 1,
        "rank": rank,
        "target_rank": target_rank,
        "stage": arm.stage,
        "request_id_sha256": hashlib.sha256(request_id.encode()).hexdigest(),
        "armed": armed,
    }


class StateAxisForkFailureInjector:
    """Hold at most one one-shot worker failure arm."""

    def __init__(self) -> None:
        self._armed: ArmedStateAxisForkFailure | None = None

    @property
    def armed(self) -> ArmedStateAxisForkFailure | None:
        return self._armed

    def arm(self, request_id: str, stage: str) -> ArmedStateAxisForkFailure:
        if self._armed is not None:
            raise StateAxisFailureArmError("StateAxis worker failure is already armed")
        self._armed = validate_failure_arm(request_id, stage)
        return self._armed

    def maybe_fail(
        self, phase: FailurePhase, scheduled_request_ids: Collection[str]
    ) -> bool:
        if phase not in FAILURE_PHASES:
            raise StateAxisFailureArmError("StateAxis failure phase is invalid")
        arm = self._armed
        if arm is None or arm.phase != phase or arm.request_id not in scheduled_request_ids:
            return False
        self._armed = None
        raise StateAxisInjectedWorkerFailure(
            f"StateAxis injected worker failure at {arm.stage}"
        )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One unified-HCCL PEARL draft/target verification round."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import torch

from vllm_ascend.spec_decode.pearl.protocol import (
    PearlProposalBatch,
    PearlVerificationBatch,
    broadcast_proposals,
    broadcast_verifications,
)
from vllm_ascend.spec_decode.pearl.topology import PearlProcessGroups
from vllm_ascend.spec_decode.pearl.verifier import PearlTargetVerifier
from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmSchedule, SpecRhythmScheduler


@dataclass(frozen=True)
class PearlRoundExecutor:
    """Execute PEARL's proposal, verification, and correction collectives.

    The executor is invoked identically by every member of a unified HCCL
    world. Only the draft leader provides a proposal; only the target leader
    provides target logits. Every rank receives the resulting correction batch.
    """

    groups: PearlProcessGroups
    device: torch.device | str
    verifier: PearlTargetVerifier

    @torch.inference_mode()
    def execute(
        self,
        proposals: PearlProposalBatch | None,
        target_logits: torch.Tensor | None,
        *,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[PearlProposalBatch | None, PearlVerificationBatch]:
        """Run one PEARL candidate window through target verification."""
        rank = self.groups.rank
        topology = self.groups.topology
        if rank == topology.draft_leader_rank:
            if proposals is None:
                raise ValueError("The PEARL draft leader must provide proposals.")
        elif proposals is not None:
            raise ValueError("Only the PEARL draft leader may provide proposals.")

        received_proposals: PearlProposalBatch | None = None
        if self.groups.is_verification_worker:
            received_proposals = broadcast_proposals(
                proposals,
                source_rank=topology.draft_leader_rank,
                group=self.groups.verification_group,
                device=self.device,
            )

        if rank == topology.target_leader_rank:
            if target_logits is None:
                raise ValueError("The PEARL target leader must provide target logits.")
            assert received_proposals is not None
            verifications = self.verifier.verify(
                received_proposals,
                target_logits,
                temperature=temperature,
                generator=generator,
            )
        else:
            if target_logits is not None:
                raise ValueError("Only the PEARL target leader may provide target logits.")
            verifications = None

        return received_proposals, broadcast_verifications(
            verifications,
            source_rank=topology.target_leader_rank,
            device=self.device,
        )


@dataclass(frozen=True)
class PearlDualBatchResult:
    """Results from one concurrently scheduled draft/target window."""

    schedule: SpecRhythmSchedule
    draft_result: Any
    target_result: Any
    elapsed_seconds: float


class PearlDualModelScheduler:
    """Bridge SpecRhythm metadata to a generic two-model service.

    ``draft_runner`` and ``target_runner`` are called concurrently with
    ``(schedule, role)``. They own model execution and communication; this
    adapter only establishes the lifecycle boundary and returns both results
    as one atomic window. A worker calls :meth:`finish_verification` after
    target verification to advance prefix epochs and promote eager tickets.
    """

    def __init__(
        self,
        scheduler: SpecRhythmScheduler,
        draft_runner: Callable[[SpecRhythmSchedule, str], Any],
        target_runner: Callable[[SpecRhythmSchedule, str], Any],
    ) -> None:
        self.scheduler = scheduler
        self.draft_runner = draft_runner
        self.target_runner = target_runner

    def execute_step(
        self,
        *,
        projected_wait_ms: float,
        context_len: int,
        draft_token_budget: int | None = None,
        batch_size: int | None = None,
    ) -> PearlDualBatchResult:
        schedule = self.scheduler.schedule(
            projected_wait_ms=projected_wait_ms,
            context_len=context_len,
            draft_token_budget=draft_token_budget,
            batch_size=batch_size,
        )
        started = perf_counter()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pearl") as pool:
            draft_future = pool.submit(self.draft_runner, schedule, "draft")
            target_future = pool.submit(self.target_runner, schedule, "target")
            draft_result = draft_future.result()
            target_result = target_future.result()
        return PearlDualBatchResult(
            schedule=schedule,
            draft_result=draft_result,
            target_result=target_result,
            elapsed_seconds=perf_counter() - started,
        )

    def finish_verification(self, request_index: int, **kwargs: Any) -> Any:
        """Commit one target verdict through the scheduler's guarded lifecycle."""
        return self.scheduler.finish_verification(request_index, **kwargs)

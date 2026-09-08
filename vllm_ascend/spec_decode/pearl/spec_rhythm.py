# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SpecRhythm control plane and guarded two-batch mailbox state.

The paper separates the mechanism that creates a draft window from the policy
that allocates it.  This module follows that boundary: it contains no model or
distributed calls, and owns only deterministic request accounting, budget
shaping, and proposal lifecycle transitions.  The native PEARL engine consumes
the resulting plans and keeps token/KV mutation on the owning model workers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class SpecRhythmRuntimeState:
    """Online state used by the paper's urgency and acceptance policy."""

    request_index: int
    home_batch_id: int
    slo_tpot_ms: float | None = None
    slo_class: str | None = None
    max_gamma: int | None = None
    # ``N`` in the paper is the number of tokens delivered before the first
    # verification round.  The denominator is guarded with ``max(1, N)``
    # where it is used, so a fresh request must start at zero rather than one.
    delivered_tokens: int = 0
    decode_elapsed_ms: float = 0.0
    acceptance_ema: float = 1.0
    draft_confidence_ema: float = 1.0
    prefix_epoch: int = 0
    verification_rounds: int = 0

    def __post_init__(self) -> None:
        if self.request_index < 0:
            raise ValueError("SpecRhythm request indices must be non-negative.")
        if self.home_batch_id not in (0, 1):
            raise ValueError("SpecRhythm home_batch_id must be zero or one.")
        if self.slo_tpot_ms is not None and self.slo_tpot_ms <= 0:
            raise ValueError("SpecRhythm TPOT SLO must be positive when supplied.")
        if self.max_gamma is not None and self.max_gamma <= 0:
            raise ValueError("SpecRhythm per-request max gamma must be positive.")
        if self.delivered_tokens < 0 or self.decode_elapsed_ms < 0:
            raise ValueError("SpecRhythm progress counters must be non-negative.")
        self.acceptance_ema = _clamp(float(self.acceptance_ema))
        self.draft_confidence_ema = _clamp(float(self.draft_confidence_ema))

    @property
    def expected_acceptance_benefit(self) -> float:
        return self.acceptance_ema * self.draft_confidence_ema

    @property
    def observed_tpot_ms(self) -> float:
        return self.decode_elapsed_ms / max(1, self.delivered_tokens)

    def projected_progress_gap(self, projected_wait_ms: float) -> int:
        """Return ``a_need`` from section 4.3 of the SpecRhythm paper."""

        if self.slo_tpot_ms is None:
            return 0
        projected_tokens = (
            self.decode_elapsed_ms + max(0.0, float(projected_wait_ms))
        ) / self.slo_tpot_ms
        return int(math.ceil(max(0.0, projected_tokens - self.delivered_tokens)))

    def urgency(self, projected_wait_ms: float) -> float:
        if self.slo_tpot_ms is None:
            return 0.0
        projected_tpot = (
            self.decode_elapsed_ms + max(0.0, float(projected_wait_ms))
        ) / max(1, self.delivered_tokens)
        return max(0.0, projected_tpot / self.slo_tpot_ms)

    def add_decode_time(self, elapsed_ms: float) -> None:
        self.decode_elapsed_ms += max(0.0, float(elapsed_ms))

    def record_verification(
        self,
        *,
        proposed_tokens: int,
        accepted_tokens: int,
        delivered_tokens: int,
        draft_confidence: float | None,
        ema_alpha: float,
    ) -> None:
        proposed = max(0, int(proposed_tokens))
        accepted = max(0, min(int(accepted_tokens), proposed))
        delivered = max(0, int(delivered_tokens))
        alpha = _clamp(float(ema_alpha), 1e-6, 1.0)
        if proposed:
            sample = accepted / proposed
            self.acceptance_ema = alpha * sample + (1.0 - alpha) * self.acceptance_ema
        if draft_confidence is not None:
            confidence = _clamp(float(draft_confidence))
            self.draft_confidence_ema = (
                alpha * confidence + (1.0 - alpha) * self.draft_confidence_ema
            )
        self.delivered_tokens += delivered
        self.verification_rounds += 1
        self.prefix_epoch += 1


@dataclass(frozen=True)
class SpecRhythmBudgetPlan:
    """Per-window allocation constrained by draft work and target rooflines."""

    plan_id: int
    normal_budgets: Mapping[int, int]
    eager_budgets: Mapping[int, int]
    progress_gaps: Mapping[int, int]
    eager_priorities: Mapping[int, float]
    verification_roof: int
    draft_token_budget: int
    allocated_draft_tokens: int

    def __post_init__(self) -> None:
        normal = dict(self.normal_budgets)
        eager = dict(self.eager_budgets)
        if set(normal).intersection(eager):
            raise ValueError("Normal and eager SpecRhythm draft sets must be disjoint.")
        if any(value <= 0 for value in (*normal.values(), *eager.values())):
            raise ValueError("Every allocated SpecRhythm proposal must contain a token.")
        normal_tokens = sum(normal.values())
        eager_tokens = sum(eager.values())
        if normal_tokens > self.verification_roof:
            raise ValueError("Normal proposals exceed the target verification roofline.")
        if eager_tokens > self.verification_roof:
            raise ValueError("Eager proposals exceed the target verification roofline.")
        if normal_tokens + eager_tokens > self.verification_roof:
            raise ValueError(
                "Normal and eager proposals exceed the global target verification roofline."
            )
        if self.allocated_draft_tokens != sum(normal.values()) + sum(eager.values()):
            raise ValueError("SpecRhythm allocated draft-token accounting is inconsistent.")
        if self.allocated_draft_tokens > self.draft_token_budget:
            raise ValueError("SpecRhythm proposals exceed the available draft window.")


class SpecRhythmBudgetShaper:
    """Two-stage SLO-aware candidate allocator from paper section 4.4."""

    def __init__(
        self,
        *,
        min_gamma: int,
        max_gamma: int,
        acceptance_floor: float = 0.4,
        acceptance_ema_alpha: float = 0.2,
        roofline: Mapping[str, int] | None = None,
    ) -> None:
        if min_gamma <= 0 or max_gamma < min_gamma:
            raise ValueError("SpecRhythm requires 1 <= min_gamma <= max_gamma.")
        self.min_gamma = int(min_gamma)
        self.max_gamma = int(max_gamma)
        self.acceptance_floor = _clamp(float(acceptance_floor))
        self.acceptance_ema_alpha = _clamp(
            float(acceptance_ema_alpha), 1e-6, 1.0
        )
        self.roofline = {str(key): int(value) for key, value in (roofline or {}).items()}
        if any(value <= 0 for value in self.roofline.values()):
            raise ValueError("SpecRhythm roofline entries must be positive token budgets.")

    def verification_roof(self, batch_size: int, context_len: int) -> int:
        """Look up the profiled batch-level candidate-token roofline."""

        batch_size = max(1, int(batch_size))
        context_bucket = max(1, math.ceil(max(1, int(context_len)) / 512))
        for key in (
            f"{batch_size}:{context_bucket}",
            f"batch:{batch_size}",
            "default",
        ):
            if key in self.roofline:
                return max(batch_size, self.roofline[key])
        return batch_size * self.max_gamma

    def shape(
        self,
        *,
        plan_id: int,
        normal_request_indices: Sequence[int],
        eager_request_indices: Sequence[int],
        states: Mapping[int, SpecRhythmRuntimeState],
        projected_wait_ms: float,
        context_len: int,
        draft_token_budget: int | None = None,
        batch_size: int | None = None,
    ) -> SpecRhythmBudgetPlan:
        normal = tuple(dict.fromkeys(int(index) for index in normal_request_indices))
        eager = tuple(dict.fromkeys(int(index) for index in eager_request_indices))
        if set(normal).intersection(eager):
            raise ValueError("A request cannot be both a normal and eager draft in one step.")
        requested = (*normal, *eager)
        if any(index not in states for index in requested):
            raise ValueError("SpecRhythm cannot shape an unregistered request.")

        roof = self.verification_roof(
            max(len(requested), 1) if batch_size is None else batch_size,
            context_len,
        )
        available_draft = (
            2 * roof if draft_token_budget is None else max(0, int(draft_token_budget))
        )
        normal_caps = {
            index: min(self.max_gamma, states[index].max_gamma or self.max_gamma)
            for index in normal
        }
        minimum_needed = sum(min(self.min_gamma, normal_caps[index]) for index in normal)
        if minimum_needed > roof or minimum_needed > available_draft:
            raise ValueError(
                "The SpecRhythm roofline cannot fit the minimum normal proposal budget."
            )

        normal_budgets = {
            index: min(self.min_gamma, normal_caps[index]) for index in normal
        }
        eager_budgets: dict[int, int] = {}
        remaining_draft = available_draft - minimum_needed
        # The roofline is a target-side *global* candidate budget.  Normal and
        # eager proposals share it because both are verified in the same target
        # step.  Keeping one counter also makes the invariant explicit for
        # future tree-shaped allocations.
        remaining_roof = roof - minimum_needed
        gaps = {
            index: states[index].projected_progress_gap(projected_wait_ms)
            for index in requested
        }
        priorities = {
            index: gaps[index] * states[index].expected_acceptance_benefit
            for index in eager
        }

        # Stage 1: close projected progress gaps. Prefix depth is allocated one
        # token at a time, so no child candidate can exist without its parent.
        urgent = sorted(
            requested,
            key=lambda index: (
                gaps[index] * states[index].expected_acceptance_benefit,
                states[index].urgency(projected_wait_ms),
                -index,
            ),
            reverse=True,
        )
        for index in urgent:
            state = states[index]
            if gaps[index] <= 0 or state.expected_acceptance_benefit < self.acceptance_floor:
                continue
            cap = min(self.max_gamma, state.max_gamma or self.max_gamma)
            if index in normal_budgets:
                wanted = max(0, min(cap, gaps[index]) - normal_budgets[index])
                grant = min(wanted, remaining_roof, remaining_draft)
                normal_budgets[index] += grant
                remaining_roof -= grant
            else:
                wanted = min(cap, max(self.min_gamma, gaps[index]))
                grant = min(wanted, remaining_roof, remaining_draft)
                if grant > 0:
                    eager_budgets[index] = grant
                    remaining_roof -= grant
            remaining_draft -= grant
            if remaining_draft <= 0:
                break

        # Stage 2: spend residual capacity on the highest marginal expected
        # progress. Deeper tokens decay by the measured acceptance benefit.
        while remaining_draft > 0:
            candidates: list[tuple[float, int, str]] = []
            for kind, budgets, indices in (
                ("normal", normal_budgets, normal),
                ("eager", eager_budgets, eager),
            ):
                if remaining_roof <= 0:
                    continue
                for index in indices:
                    state = states[index]
                    current = budgets.get(index, 0)
                    cap = min(self.max_gamma, state.max_gamma or self.max_gamma)
                    if current >= cap:
                        continue
                    benefit = state.expected_acceptance_benefit
                    if kind == "eager" and benefit < self.acceptance_floor:
                        continue
                    urgency_weight = 1.0 + state.urgency(projected_wait_ms)
                    marginal = urgency_weight * (benefit ** (current + 1))
                    candidates.append((marginal, -index, kind))
            if not candidates:
                break
            _, negative_index, kind = max(candidates)
            index = -negative_index
            if kind == "normal":
                normal_budgets[index] += 1
            else:
                eager_budgets[index] = eager_budgets.get(index, 0) + 1
            remaining_roof -= 1
            remaining_draft -= 1

        allocated = sum(normal_budgets.values()) + sum(eager_budgets.values())
        return SpecRhythmBudgetPlan(
            plan_id=int(plan_id),
            normal_budgets=normal_budgets,
            eager_budgets=eager_budgets,
            progress_gaps=gaps,
            eager_priorities=priorities,
            verification_roof=roof,
            draft_token_budget=available_draft,
            allocated_draft_tokens=allocated,
        )


class ProposalLifecycle(str, Enum):
    AVAILABLE = "available"
    STAGED_EAGER = "staged_eager"
    CONSUMED = "consumed"
    INVALIDATED = "invalidated"


@dataclass
class SpecRhythmProposalTicket:
    """Guard metadata carried with a device-resident proposal payload."""

    proposal_id: int
    request_index: int
    home_batch_id: int
    gamma: int
    required_prefix_epoch: int
    eager: bool = False
    lifecycle: ProposalLifecycle = ProposalLifecycle.AVAILABLE

    def __post_init__(self) -> None:
        if self.proposal_id < 0 or self.request_index < 0 or self.gamma <= 0:
            raise ValueError("SpecRhythm proposal identifiers and gamma must be positive.")
        if self.home_batch_id not in (0, 1) or self.required_prefix_epoch < 0:
            raise ValueError("SpecRhythm proposal routing metadata is invalid.")


class PipelinePhase(str, Enum):
    WARMUP = "warmup"
    STEADY = "steady"
    DRAIN = "drain"
    COMPLETE = "complete"


@dataclass(frozen=True)
class SpecRhythmExecutionPlan:
    plan_id: int
    phase: PipelinePhase
    target_home_batch_id: int | None
    draft_home_batch_id: int | None
    target_request_indices: tuple[int, ...]
    normal_draft_request_indices: tuple[int, ...]
    eager_candidate_indices: tuple[int, ...]


@dataclass
class SpecRhythmPipelineController:
    """Own two alternating logical batches and guarded proposal lifecycles."""

    request_states: dict[int, SpecRhythmRuntimeState]
    next_target_home_batch_id: int = 0
    plan_id: int = 0
    _next_proposal_id: int = 0
    ready: dict[int, SpecRhythmProposalTicket] = field(default_factory=dict)
    staged_eager: dict[int, SpecRhythmProposalTicket] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.next_target_home_batch_id not in (0, 1):
            raise ValueError("SpecRhythm next target batch must be zero or one.")

    def build_plan(self, active_request_indices: Sequence[int]) -> SpecRhythmExecutionPlan:
        active = tuple(dict.fromkeys(int(index) for index in active_request_indices))
        if any(index not in self.request_states for index in active):
            raise ValueError("SpecRhythm cannot schedule an unregistered request.")
        active_set = set(active)
        self._discard_inactive_payloads(active_set)
        ready_homes = {
            self.request_states[index].home_batch_id
            for index in active
            if index in self.ready
        }
        target_home: int | None
        if self.next_target_home_batch_id in ready_homes:
            target_home = self.next_target_home_batch_id
        elif ready_homes:
            target_home = min(ready_homes)
        else:
            target_home = None
        draft_home = (
            1 - target_home
            if target_home is not None
            else self.next_target_home_batch_id
        )
        if target_home is None and not any(
            self.request_states[index].home_batch_id == draft_home
            for index in active
        ):
            draft_home = 1 - draft_home
        target = tuple(
            index
            for index in active
            if target_home is not None
            and self.request_states[index].home_batch_id == target_home
            and index in self.ready
        )
        normal = tuple(
            index
            for index in active
            if self.request_states[index].home_batch_id == draft_home
            and index not in self.ready
            and index not in self.staged_eager
        )
        eager = tuple(
            index
            for index in target
            if index not in self.staged_eager
        )
        if target:
            phase = PipelinePhase.STEADY
        elif normal:
            phase = PipelinePhase.WARMUP
        elif self.ready:
            phase = PipelinePhase.DRAIN
        else:
            phase = PipelinePhase.COMPLETE
        plan = SpecRhythmExecutionPlan(
            plan_id=self.plan_id,
            phase=phase,
            target_home_batch_id=target_home,
            draft_home_batch_id=draft_home if normal or eager else None,
            target_request_indices=target,
            normal_draft_request_indices=normal,
            eager_candidate_indices=eager,
        )
        self.plan_id += 1
        return plan

    def new_ticket(
        self,
        request_index: int,
        *,
        gamma: int,
        eager: bool,
    ) -> SpecRhythmProposalTicket:
        state = self.request_states[request_index]
        ticket = SpecRhythmProposalTicket(
            proposal_id=self._next_proposal_id,
            request_index=request_index,
            home_batch_id=state.home_batch_id,
            gamma=int(gamma),
            required_prefix_epoch=state.prefix_epoch + int(eager),
            eager=bool(eager),
            lifecycle=(
                ProposalLifecycle.STAGED_EAGER
                if eager
                else ProposalLifecycle.AVAILABLE
            ),
        )
        self._next_proposal_id += 1
        return ticket

    def publish(self, tickets: Sequence[SpecRhythmProposalTicket]) -> None:
        for ticket in tickets:
            index = ticket.request_index
            if ticket.eager:
                if index in self.staged_eager:
                    raise RuntimeError("SpecRhythm request already has a staged eager continuation.")
                self.staged_eager[index] = ticket
            else:
                if index in self.ready:
                    raise RuntimeError("SpecRhythm request already has an available proposal.")
                if ticket.required_prefix_epoch != self.request_states[index].prefix_epoch:
                    raise RuntimeError("SpecRhythm normal proposal was built from a stale prefix.")
                self.ready[index] = ticket

    def finish_verification(
        self,
        request_index: int,
        *,
        fully_accepted: bool,
        proposed_tokens: int,
        accepted_tokens: int,
        delivered_tokens: int,
        draft_confidence: float | None,
        ema_alpha: float,
    ) -> SpecRhythmProposalTicket | None:
        current = self.ready.pop(request_index, None)
        if current is None:
            raise RuntimeError("SpecRhythm target attempted to consume a missing proposal.")
        state = self.request_states[request_index]
        if current.required_prefix_epoch != state.prefix_epoch:
            raise RuntimeError("SpecRhythm target attempted to verify a stale proposal.")
        current.lifecycle = ProposalLifecycle.CONSUMED
        state.record_verification(
            proposed_tokens=proposed_tokens,
            accepted_tokens=accepted_tokens,
            delivered_tokens=delivered_tokens,
            draft_confidence=draft_confidence,
            ema_alpha=ema_alpha,
        )
        eager = self.staged_eager.pop(request_index, None)
        if eager is None:
            promoted = None
        elif fully_accepted and eager.required_prefix_epoch == state.prefix_epoch:
            eager.lifecycle = ProposalLifecycle.AVAILABLE
            self.ready[request_index] = eager
            promoted = eager
        else:
            eager.lifecycle = ProposalLifecycle.INVALIDATED
            promoted = None
        self.next_target_home_batch_id = 1 - state.home_batch_id
        return promoted

    def invalidate_request(self, request_index: int) -> None:
        for mapping in (self.ready, self.staged_eager):
            ticket = mapping.pop(request_index, None)
            if ticket is not None:
                ticket.lifecycle = ProposalLifecycle.INVALIDATED

    def _discard_inactive_payloads(self, active: set[int]) -> None:
        for index in set(self.ready).union(self.staged_eager) - active:
            self.invalidate_request(index)


@dataclass(frozen=True)
class SpecRhythmSchedule:
    """Portable scheduler output for a generic vLLM service adapter."""

    execution: SpecRhythmExecutionPlan
    budget: SpecRhythmBudgetPlan


class SpecRhythmScheduler:
    """Request admission/preemption adapter shared by non-native frontends.

    The scheduler owns metadata and lifecycle only. A worker integration can
    consume :class:`SpecRhythmSchedule.budget`, execute its own draft/target
    forwards, then call :meth:`finish_verification` to commit the guarded
    prefix. This keeps generic request admission independent from the Ascend
    HCCL transport.
    """

    def __init__(self, shaper: SpecRhythmBudgetShaper, *, max_num_seqs: int = 512) -> None:
        if max_num_seqs <= 0:
            raise ValueError("SpecRhythm scheduler max_num_seqs must be positive")
        self.shaper = shaper
        self.max_num_seqs = int(max_num_seqs)
        self.request_states: dict[int, SpecRhythmRuntimeState] = {}
        self._active: list[int] = []
        self.controller = SpecRhythmPipelineController(self.request_states)

    @property
    def active_request_indices(self) -> tuple[int, ...]:
        return tuple(self._active)

    def admit(
        self,
        request_index: int,
        *,
        home_batch_id: int | None = None,
        slo_tpot_ms: float | None = None,
        slo_class: str | None = None,
        max_gamma: int | None = None,
    ) -> SpecRhythmRuntimeState:
        """Register and admit one request into the alternating batch set."""

        index = int(request_index)
        if index in self.request_states:
            raise ValueError(f"SpecRhythm request {index} is already registered")
        if len(self._active) >= self.max_num_seqs:
            raise RuntimeError("SpecRhythm scheduler has no free request slots")
        if home_batch_id is None:
            counts = [
                sum(self.request_states[current].home_batch_id == home for current in self._active)
                for home in (0, 1)
            ]
            home_batch_id = 0 if counts[0] <= counts[1] else 1
        state = SpecRhythmRuntimeState(
            request_index=index,
            home_batch_id=int(home_batch_id),
            slo_tpot_ms=slo_tpot_ms,
            slo_class=slo_class,
            max_gamma=max_gamma,
        )
        self.request_states[index] = state
        self._active.append(index)
        return state

    def preempt(self, request_index: int) -> None:
        """Remove a request from active scheduling while retaining accounting."""

        index = int(request_index)
        if index not in self.request_states:
            raise KeyError(index)
        self.controller.invalidate_request(index)
        self._active = [current for current in self._active if current != index]

    def reactivate(self, request_index: int) -> None:
        index = int(request_index)
        if index not in self.request_states:
            raise KeyError(index)
        if index in self._active:
            return
        if len(self._active) >= self.max_num_seqs:
            raise RuntimeError("SpecRhythm scheduler has no free request slots")
        self._active.append(index)

    def remove(self, request_index: int) -> None:
        index = int(request_index)
        self.preempt(index)
        del self.request_states[index]

    def schedule(
        self,
        *,
        projected_wait_ms: float,
        context_len: int,
        draft_token_budget: int | None = None,
        batch_size: int | None = None,
    ) -> SpecRhythmSchedule:
        """Build a dual-batch execution plan and one global roofline budget."""

        execution = self.controller.build_plan(self._active)
        budget = self.shaper.shape(
            plan_id=execution.plan_id,
            normal_request_indices=execution.normal_draft_request_indices,
            eager_request_indices=execution.eager_candidate_indices,
            states=self.request_states,
            projected_wait_ms=projected_wait_ms,
            context_len=context_len,
            draft_token_budget=draft_token_budget,
            batch_size=batch_size,
        )
        return SpecRhythmSchedule(execution=execution, budget=budget)

    def finish_verification(self, request_index: int, **kwargs) -> SpecRhythmProposalTicket | None:
        return self.controller.finish_verification(request_index, **kwargs)


__all__ = [
    "PipelinePhase",
    "ProposalLifecycle",
    "SpecRhythmBudgetPlan",
    "SpecRhythmBudgetShaper",
    "SpecRhythmExecutionPlan",
    "SpecRhythmPipelineController",
    "SpecRhythmSchedule",
    "SpecRhythmScheduler",
    "SpecRhythmProposalTicket",
    "SpecRhythmRuntimeState",
]

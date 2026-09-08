# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.spec_decode.pearl.spec_rhythm import (
    SpecRhythmBudgetShaper,
    SpecRhythmScheduler,
    PipelinePhase,
)


def test_scheduler_admits_balanced_batches_and_shapes_global_budget():
    scheduler = SpecRhythmScheduler(
        SpecRhythmBudgetShaper(min_gamma=1, max_gamma=4), max_num_seqs=4
    )
    scheduler.admit(0)
    scheduler.admit(1)
    assert scheduler.request_states[0].home_batch_id != scheduler.request_states[1].home_batch_id
    first = scheduler.schedule(projected_wait_ms=0, context_len=16, draft_token_budget=8)
    assert first.execution.phase is PipelinePhase.WARMUP
    assert first.budget.allocated_draft_tokens <= first.budget.verification_roof
    scheduler.preempt(1)
    assert scheduler.active_request_indices == (0,)
    scheduler.reactivate(1)
    assert scheduler.active_request_indices == (0, 1)


def test_scheduler_remove_reclaims_slot():
    scheduler = SpecRhythmScheduler(
        SpecRhythmBudgetShaper(min_gamma=1, max_gamma=2), max_num_seqs=1
    )
    scheduler.admit(4)
    scheduler.remove(4)
    scheduler.admit(5)
    assert scheduler.active_request_indices == (5,)

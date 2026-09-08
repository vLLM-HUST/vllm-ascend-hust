# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.spec_decode.pearl.spec_rhythm import (
    PipelinePhase,
    ProposalLifecycle,
    SpecRhythmBudgetShaper,
    SpecRhythmPipelineController,
    SpecRhythmRuntimeState,
)


def _states(count=4):
    return {
        index: SpecRhythmRuntimeState(
            request_index=index,
            home_batch_id=index % 2,
            slo_tpot_ms=40.0 if index < 2 else 150.0,
        )
        for index in range(count)
    }


def test_progress_gap_matches_paper_equation():
    state = SpecRhythmRuntimeState(
        request_index=0,
        home_batch_id=0,
        slo_tpot_ms=40.0,
        delivered_tokens=4,
        decode_elapsed_ms=150.0,
    )
    assert state.projected_progress_gap(90.0) == 2
    assert state.projected_progress_gap(0.0) == 0


def test_budget_shaper_honors_batch_roof_and_draft_window():
    states = _states()
    states[0].decode_elapsed_ms = 300.0
    states[1].decode_elapsed_ms = 300.0
    shaper = SpecRhythmBudgetShaper(
        min_gamma=1,
        max_gamma=5,
        roofline={"batch:2": 7},
    )
    plan = shaper.shape(
        plan_id=3,
        normal_request_indices=[1, 3],
        eager_request_indices=[0, 2],
        states=states,
        projected_wait_ms=80.0,
        context_len=256,
        draft_token_budget=9,
    )
    assert sum(plan.normal_budgets.values()) <= 7
    assert sum(plan.eager_budgets.values()) <= 7
    assert plan.allocated_draft_tokens <= 9
    assert plan.eager_priorities[0] > plan.eager_priorities[2]
    assert plan.eager_budgets.get(0, 0) >= plan.eager_budgets.get(2, 0)


def test_budget_shaper_preserves_prefix_by_allocating_integer_depths():
    states = _states(2)
    states[0].decode_elapsed_ms = 500.0
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=4)
    plan = shaper.shape(
        plan_id=0,
        normal_request_indices=[1],
        eager_request_indices=[0],
        states=states,
        projected_wait_ms=100.0,
        context_len=32,
        draft_token_budget=5,
    )
    assert 1 <= plan.normal_budgets[1] <= 4
    assert 1 <= plan.eager_budgets[0] <= 4
    assert plan.allocated_draft_tokens == 5


def test_budget_plan_rejects_normal_plus_eager_global_roof_overrun():
    from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmBudgetPlan

    with pytest.raises(ValueError, match="global target verification roofline"):
        SpecRhythmBudgetPlan(
            plan_id=0,
            normal_budgets={0: 3},
            eager_budgets={1: 3},
            progress_gaps={0: 0, 1: 0},
            eager_priorities={1: 0.0},
            verification_roof=5,
            draft_token_budget=6,
            allocated_draft_tokens=6,
        )


def test_dual_batch_pipeline_warms_up_then_alternates():
    states = _states()
    controller = SpecRhythmPipelineController(states)
    warmup = controller.build_plan([0, 1, 2, 3])
    assert warmup.phase is PipelinePhase.WARMUP
    assert warmup.target_request_indices == ()
    assert warmup.normal_draft_request_indices == (0, 2)

    tickets = [controller.new_ticket(index, gamma=4, eager=False) for index in (0, 2)]
    controller.publish(tickets)
    first = controller.build_plan([0, 1, 2, 3])
    assert first.phase is PipelinePhase.STEADY
    assert first.target_request_indices == (0, 2)
    assert first.normal_draft_request_indices == (1, 3)
    assert first.eager_candidate_indices == (0, 2)


def test_full_acceptance_promotes_matching_eager_continuation():
    states = _states(2)
    controller = SpecRhythmPipelineController(states)
    normal = controller.new_ticket(0, gamma=3, eager=False)
    controller.publish([normal])
    eager = controller.new_ticket(0, gamma=2, eager=True)
    controller.publish([eager])

    promoted = controller.finish_verification(
        0,
        fully_accepted=True,
        proposed_tokens=3,
        accepted_tokens=3,
        delivered_tokens=3,
        draft_confidence=0.8,
        ema_alpha=0.2,
    )
    assert promoted is eager
    assert eager.lifecycle is ProposalLifecycle.AVAILABLE
    assert controller.ready[0] is eager
    assert states[0].prefix_epoch == eager.required_prefix_epoch == 1


def test_rejection_discards_eager_continuation_and_advances_epoch():
    states = _states(2)
    controller = SpecRhythmPipelineController(states)
    normal = controller.new_ticket(0, gamma=4, eager=False)
    controller.publish([normal])
    eager = controller.new_ticket(0, gamma=3, eager=True)
    controller.publish([eager])

    promoted = controller.finish_verification(
        0,
        fully_accepted=False,
        proposed_tokens=4,
        accepted_tokens=1,
        delivered_tokens=2,
        draft_confidence=0.7,
        ema_alpha=0.2,
    )
    assert promoted is None
    assert eager.lifecycle is ProposalLifecycle.INVALIDATED
    assert 0 not in controller.ready
    assert 0 not in controller.staged_eager
    assert states[0].prefix_epoch == 1


def test_stale_mailbox_is_rejected_before_state_mutation():
    states = _states(1)
    controller = SpecRhythmPipelineController(states)
    ticket = controller.new_ticket(0, gamma=2, eager=False)
    controller.publish([ticket])
    states[0].prefix_epoch += 1
    with pytest.raises(RuntimeError, match="stale proposal"):
        controller.finish_verification(
            0,
            fully_accepted=True,
            proposed_tokens=2,
            accepted_tokens=2,
            delivered_tokens=2,
            draft_confidence=None,
            ema_alpha=0.2,
        )


def test_inactive_request_invalidates_ready_and_staged_payloads():
    states = _states(2)
    controller = SpecRhythmPipelineController(states)
    ready = controller.new_ticket(0, gamma=2, eager=False)
    controller.publish([ready])
    controller.build_plan([1])
    assert ready.lifecycle is ProposalLifecycle.INVALIDATED
    assert controller.ready == {}

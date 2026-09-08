# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmBudgetPlan
from vllm_ascend.spec_decode.pearl.tree import (
    build_tree_attention_mask,
    build_tree_speculation_plan,
    make_spine_first_parents,
    select_tree_candidates,
    SpecRhythmTreeCoordinator,
    tree_budget_from_spec_rhythm,
)


def test_tree_mask_exposes_prefix_root_and_ancestors_only():
    mask = build_tree_attention_mask(2, 2, prefix_len=3, max_model_len=8)
    assert mask.shape == (5, 8)
    assert not mask[0, :4].any()
    assert not mask[1, :5].any()
    assert not mask[2, 5]
    assert mask[3, 4]
    assert mask[3, 5]
    assert not mask[3, 6]
    assert not mask[4, 4]
    assert mask[4, 6]


def test_tree_plan_positions_and_budget():
    plan = build_tree_speculation_plan(2, 3, prefix_len=4, max_model_len=12, candidate_budget=4)
    assert plan.positions.tolist() == [4, 5, 6, 7, 5, 6, 7]
    assert plan.candidate_budget == 4
    assert torch.equal(plan.parent_indices, make_spine_first_parents(2, 3))


def test_selection_preserves_ancestor_chain():
    parents = make_spine_first_parents(2, 3)
    tokens = torch.arange(6)
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 8.0])
    selected = select_tree_candidates(tokens, parents, scores, budget=3)
    assert selected.indices.tolist() == [0, 1, 5]
    assert selected.token_ids.tolist() == [0, 1, 5]


def test_selection_rejects_invalid_parent_order():
    with pytest.raises(ValueError):
        select_tree_candidates(torch.arange(2), torch.tensor([1, -1]), torch.ones(2), 1)


def test_tree_budget_reads_normal_and_eager_allocations():
    plan = SpecRhythmBudgetPlan(
        plan_id=1,
        normal_budgets={2: 3},
        eager_budgets={5: 2},
        progress_gaps={2: 3, 5: 2},
        eager_priorities={5: 1.0},
        verification_roof=8,
        draft_token_budget=8,
        allocated_draft_tokens=5,
    )
    assert tree_budget_from_spec_rhythm(plan, 2, 2, 3) == 3
    assert tree_budget_from_spec_rhythm(plan, 5, 2, 3) == 2


def test_tree_coordinator_uses_budget_to_choose_depth():
    plan = SpecRhythmBudgetPlan(
        plan_id=1,
        normal_budgets={2: 3},
        eager_budgets={},
        progress_gaps={2: 3},
        eager_priorities={},
        verification_roof=4,
        draft_token_budget=4,
        allocated_draft_tokens=3,
    )
    tree = SpecRhythmTreeCoordinator(width=2, max_depth=3).for_request(
        plan, 2, prefix_len=2, max_model_len=8
    )
    assert tree is not None
    assert (tree.width, tree.depth, tree.candidate_budget) == (2, 2, 3)

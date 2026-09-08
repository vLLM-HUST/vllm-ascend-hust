# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the experimental PEARL protocol building blocks."""

import pytest
import torch

from vllm_ascend.spec_decode.pearl import (
    PearlPhase,
    PearlProposalBatch,
    PearlRequestState,
    PearlTopology,
    PearlVerificationBatch,
    advance_request_states,
)


def test_topology_creates_disjoint_draft_target_and_verification_groups():
    topology = PearlTopology.from_tensor_parallel_sizes(draft_tp_size=2, target_tp_size=3)

    assert topology.draft_ranks == (0, 1)
    assert topology.target_ranks == (2, 3, 4)
    assert topology.verification_ranks == (0, 2, 3, 4)
    assert topology.correction_ranks == (0, 1, 2)
    assert topology.world_size == 5
    assert topology.is_verification_rank(0)
    assert not topology.is_verification_rank(1)
    topology.validate_world_size(5)


@pytest.mark.parametrize(
    ("draft_ranks", "target_ranks"),
    [
        ((), (0,)),
        ((0,), ()),
        ((0, 1), (1, 2)),
        ((-1,), (0,)),
    ],
)
def test_topology_rejects_invalid_rank_assignments(draft_ranks, target_ranks):
    with pytest.raises(ValueError):
        PearlTopology(draft_ranks=draft_ranks, target_ranks=target_ranks)


def test_proposal_tensor_round_trip_preserves_variable_windows():
    proposals = PearlProposalBatch(
        request_slots=(5, 9),
        candidate_token_ids=((10, 11, 12), (20,)),
    )

    decoded = PearlProposalBatch.from_tensors(*proposals.to_tensors("cpu"))

    assert decoded == proposals


def test_verification_tensor_round_trip_preserves_corrections():
    verifications = PearlVerificationBatch(
        request_slots=(5, 9),
        accepted_prefix_lengths=(3, 0),
        correction_token_ids=(None, 42),
        finished=(False, True),
    )

    decoded = PearlVerificationBatch.from_tensors(*verifications.to_tensors("cpu"))

    assert decoded == verifications


def test_verification_requires_a_correction_for_rejection():
    proposals = PearlProposalBatch(request_slots=(5,), candidate_token_ids=((10, 11),))
    verifications = PearlVerificationBatch(
        request_slots=(5,),
        accepted_prefix_lengths=(1,),
        correction_token_ids=(None,),
        finished=(False,),
    )

    with pytest.raises(ValueError, match="requires a correction"):
        verifications.validate_against(proposals)


def test_state_advance_commits_accepted_prefix_and_target_correction():
    request_states = {
        5: PearlRequestState(request_slot=5, token_ids=(1, 2)),
        9: PearlRequestState(request_slot=9, token_ids=(3,)),
    }
    proposals = PearlProposalBatch(
        request_slots=(5, 9),
        candidate_token_ids=((10, 11, 12), (20, 21)),
    )
    verifications = PearlVerificationBatch(
        request_slots=(5, 9),
        accepted_prefix_lengths=(3, 1),
        correction_token_ids=(None, 99),
        finished=(False, True),
    )

    next_states = advance_request_states(request_states, proposals, verifications)

    assert next_states[5].token_ids == (1, 2, 10, 11, 12)
    assert next_states[5].accepted_draft_tokens == 3
    assert next_states[5].phase is PearlPhase.PIPELINED
    assert next_states[9].token_ids == (3, 20, 99)
    assert next_states[9].accepted_draft_tokens == 1
    assert next_states[9].phase is PearlPhase.FINISHED


def test_state_rejects_updates_after_completion():
    state = PearlRequestState(request_slot=5, token_ids=(1,), phase=PearlPhase.FINISHED)

    with pytest.raises(ValueError, match="finished request"):
        state.apply_verification((10,), 0, 42, False)


def test_proposal_decode_rejects_invalid_tensor_shape():
    proposals = PearlProposalBatch(request_slots=(5,), candidate_token_ids=((10, 11),))
    header, request_slots, candidate_lengths, candidate_token_ids = proposals.to_tensors("cpu")

    with pytest.raises(ValueError, match="candidate_token_ids has shape"):
        PearlProposalBatch.from_tensors(
            header,
            request_slots,
            candidate_lengths,
            torch.empty((1, 3), dtype=torch.int64),
        )

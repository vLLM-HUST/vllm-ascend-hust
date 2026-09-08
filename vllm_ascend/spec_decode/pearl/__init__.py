# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental building blocks for disaggregated PEARL decoding on Ascend."""

from vllm_ascend.spec_decode.pearl.api import (
    PEARLConfig,
    PEARLEngine,
    PEARLModelGroupConfig,
    SamplingParams,
    logger,
)
from vllm_ascend.spec_decode.pearl.protocol import (
    PearlProposalBatch,
    PearlVerificationBatch,
    broadcast_proposals,
    broadcast_verifications,
)
from vllm_ascend.spec_decode.pearl.runtime import PearlRoundExecutor
from vllm_ascend.spec_decode.pearl.spec_rhythm import (
    PipelinePhase,
    ProposalLifecycle,
    SpecRhythmBudgetShaper,
    SpecRhythmExecutionPlan,
    SpecRhythmPipelineController,
    SpecRhythmProposalTicket,
    SpecRhythmRuntimeState,
)
from vllm_ascend.spec_decode.pearl.state import PearlPhase, PearlRequestState, advance_request_states
from vllm_ascend.spec_decode.pearl.topology import PearlProcessGroups, PearlTopology
from vllm_ascend.spec_decode.pearl.verifier import PearlTargetVerifier
from vllm_ascend.spec_decode.pearl.vocab import PearlVocabProjection

__all__ = [
    "PEARLConfig",
    "PEARLEngine",
    "PEARLModelGroupConfig",
    "PearlPhase",
    "PearlProcessGroups",
    "PearlProposalBatch",
    "PearlRequestState",
    "PearlRoundExecutor",
    "PearlTopology",
    "PearlTargetVerifier",
    "PearlVerificationBatch",
    "PearlVocabProjection",
    "SamplingParams",
    "PipelinePhase",
    "ProposalLifecycle",
    "SpecRhythmBudgetShaper",
    "SpecRhythmExecutionPlan",
    "SpecRhythmPipelineController",
    "SpecRhythmProposalTicket",
    "SpecRhythmRuntimeState",
    "advance_request_states",
    "broadcast_proposals",
    "broadcast_verifications",
    "logger",
]

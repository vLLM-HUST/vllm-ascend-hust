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
    SpecRhythmSchedule,
    SpecRhythmScheduler,
)
from vllm_ascend.spec_decode.pearl.state import PearlPhase, PearlRequestState, advance_request_states
from vllm_ascend.spec_decode.pearl.topology import PearlProcessGroups, PearlTopology
from vllm_ascend.spec_decode.pearl.verifier import PearlTargetVerifier
from vllm_ascend.spec_decode.pearl.vocab import PearlVocabProjection
from vllm_ascend.spec_decode.pearl.distill import (
    PearlDistillationConfig,
    PearlDistillationMetrics,
    load_pearl_distillation_checkpoint,
    pearl_distillation_loss,
    save_pearl_distillation_checkpoint,
    train_pearl_distillation_step,
)
from vllm_ascend.spec_decode.pearl.tree import (
    TreeCandidateSelection,
    TreeSpeculationPlan,
    build_tree_attention_mask,
    build_tree_speculation_plan,
    make_spine_first_parents,
    select_tree_candidates,
    SpecRhythmTreeCoordinator,
    tree_budget_from_spec_rhythm,
)
from vllm_ascend.spec_decode.pearl.mc2 import (
    MC2Capability,
    capability_dict,
    detect_mc2_capability,
    matmul_allreduce_add_rmsnorm_or_fallback,
)
from vllm_ascend.spec_decode.pearl.capabilities import collect_specslo_capabilities

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
    "PearlDistillationConfig",
    "PearlDistillationMetrics",
    "load_pearl_distillation_checkpoint",
    "pearl_distillation_loss",
    "save_pearl_distillation_checkpoint",
    "train_pearl_distillation_step",
    "TreeCandidateSelection",
    "TreeSpeculationPlan",
    "build_tree_attention_mask",
    "build_tree_speculation_plan",
    "make_spine_first_parents",
    "select_tree_candidates",
    "tree_budget_from_spec_rhythm",
    "MC2Capability",
    "capability_dict",
    "detect_mc2_capability",
    "matmul_allreduce_add_rmsnorm_or_fallback",
    "collect_specslo_capabilities",
    "SamplingParams",
    "PipelinePhase",
    "ProposalLifecycle",
    "SpecRhythmBudgetShaper",
    "SpecRhythmExecutionPlan",
    "SpecRhythmPipelineController",
    "SpecRhythmProposalTicket",
    "SpecRhythmRuntimeState",
    "SpecRhythmSchedule",
    "SpecRhythmScheduler",
    "SpecRhythmTreeCoordinator",
    "advance_request_states",
    "broadcast_proposals",
    "broadcast_verifications",
    "logger",
]

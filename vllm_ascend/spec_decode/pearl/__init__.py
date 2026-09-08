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
from vllm_ascend.spec_decode.pearl.runtime import (
    PearlDualBatchResult,
    PearlDualModelScheduler,
    PearlRoundExecutor,
)
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
    collect_pearl_teacher_trace,
    load_pearl_distillation_checkpoint,
    pearl_distillation_loss,
    save_pearl_distillation_checkpoint,
    train_pearl_distillation_step,
    write_pearl_teacher_trace,
)
from vllm_ascend.spec_decode.pearl.tree import (
    TreeCandidateSelection,
    TreeSpeculationPlan,
    TreeVerificationOutput,
    build_tree_attention_mask,
    build_tree_speculation_plan,
    make_spine_first_parents,
    select_tree_candidates,
    SpecRhythmTreeCoordinator,
    tree_budget_from_spec_rhythm,
    verify_greedy_tree,
    verify_greedy_tree_batch,
)
from vllm_ascend.spec_decode.pearl.mc2 import (
    MC2Capability,
    capability_dict,
    detect_mc2_capability,
    matmul_allreduce_add_rmsnorm_or_fallback,
    resolve_hccl_comm_name,
)
from vllm_ascend.spec_decode.pearl.capabilities import collect_specslo_capabilities
from vllm_ascend.spec_decode.tree_kv import (
    TreeKVCompactionPlan,
    build_tree_kv_compaction_plan,
    move_kv_cache_slots,
)

__all__ = [
    "PEARLConfig",
    "PEARLEngine",
    "PEARLModelGroupConfig",
    "PearlPhase",
    "PearlProcessGroups",
    "PearlProposalBatch",
    "PearlRequestState",
    "PearlRoundExecutor",
    "PearlDualBatchResult",
    "PearlDualModelScheduler",
    "PearlTopology",
    "PearlTargetVerifier",
    "PearlVerificationBatch",
    "PearlVocabProjection",
    "PearlDistillationConfig",
    "PearlDistillationMetrics",
    "collect_pearl_teacher_trace",
    "load_pearl_distillation_checkpoint",
    "pearl_distillation_loss",
    "save_pearl_distillation_checkpoint",
    "train_pearl_distillation_step",
    "write_pearl_teacher_trace",
    "TreeCandidateSelection",
    "TreeSpeculationPlan",
    "TreeVerificationOutput",
    "build_tree_attention_mask",
    "build_tree_speculation_plan",
    "make_spine_first_parents",
    "select_tree_candidates",
    "tree_budget_from_spec_rhythm",
    "verify_greedy_tree",
    "verify_greedy_tree_batch",
    "MC2Capability",
    "capability_dict",
    "detect_mc2_capability",
    "matmul_allreduce_add_rmsnorm_or_fallback",
    "resolve_hccl_comm_name",
    "TreeKVCompactionPlan",
    "build_tree_kv_compaction_plan",
    "move_kv_cache_slots",
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

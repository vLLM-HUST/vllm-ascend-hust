# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Ascend port of nano-PEARL's persistent two-model decode loop.

Run this module under ``torchrun``.  Unlike the OpenAI-compatible bridge,
all ranks form one HCCL world and retain their own model KV cache across every
PEARL round.  The round ordering matches nano-PEARL:

``pre-verify -> gamma draft tokens -> packed target verification -> rollback``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from vllm_ascend.cpu_binding import bind_cpus
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.spec_decode.pearl.native_cache import NativeCacheAllocation, NativePrefixCache
from vllm_ascend.spec_decode.pearl.native_graph import NativeACLGraphRunner
from vllm_ascend.spec_decode.pearl.native_model import (
    PAGED_ATTENTION_BLOCK_SIZE,
    NativeTPContext,
    build_native_model,
    load_native_model_weights,
)
from vllm_ascend.spec_decode.pearl.qwen_pair import validate_model_pair
from vllm_ascend.spec_decode.pearl.spec_rhythm import (
    SpecRhythmBudgetShaper,
    SpecRhythmPipelineController,
    SpecRhythmProposalTicket,
    SpecRhythmRuntimeState,
)
from vllm_ascend.spec_decode.pearl.topology import PearlProcessGroups, PearlTopology

AUTO_GAMMA_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
AUTO_GAMMA_WARMUP_STEPS = 5
AUTO_GAMMA_PROFILE_STEPS = 30
AUTO_GAMMA_PROFILE_SEQUENCE_LENGTH = 256
TARGET_VERIFICATION_GRAPH_BUCKETS = 8
PREEMPTIVE_SCHEDULING_EXPLORATION_ROUNDS = 8
PREEMPTIVE_SCHEDULING_PRIOR_ROUNDS = 8
PREEMPTIVE_SCHEDULING_RECENT_ROUNDS = 32

logger = logging.getLogger("vllm_ascend.spec_decode.pearl.native")


def _set_default_npu_environment(target_tp_size: int | None = None) -> None:
    os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
    os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    if target_tp_size == 3:
        # CANN's deterministic AIV kernel is faster for the 3-7 MiB TP3
        # all-reduces used by packed PEARL verification, and keeps greedy
        # decoding reproducible across independent worker launches.
        os.environ.setdefault("HCCL_DETERMINISTIC", "true")


@dataclass
class PearlPipelineState:
    """Logical sequence state replicated by the draft and target model groups."""

    token_ids: list[int]
    prompt_length: int
    pre_verify: bool = True
    accepted_draft_tokens: int = 0
    verified_draft_tokens: int = 0
    verification_rounds: int = 0
    committed_length: int | None = None
    temperature: float = 0.0
    draft_temperature: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = False
    slo_tpot_ms: float | None = None
    slo_class: str | None = None
    finished_decode_elapsed_ms: float | None = None
    num_acc_tokens: list[int] | None = None
    cur_acc_tokens: int = 0
    pending_window_size: int = 0
    continuation_epoch: int = 0
    request_id: str | int | None = None
    arrival_ts: float | None = None

    def __post_init__(self) -> None:
        if self.committed_length is None:
            self.committed_length = len(self.token_ids)
        if self.num_acc_tokens is None:
            self.num_acc_tokens = []
        if self.temperature < 0 or self.draft_temperature < 0 or self.max_tokens <= 0:
            raise ValueError("PEARL sampling requires non-negative temperature and positive max_tokens.")
        if self.slo_tpot_ms is not None and self.slo_tpot_ms <= 0:
            raise ValueError("PEARL TPOT SLO must be positive when supplied.")
        if self.pending_window_size < 0 or self.continuation_epoch < 0:
            raise ValueError("PEARL pending-window state must be non-negative.")

    def clone(self) -> PearlPipelineState:
        return PearlPipelineState(
            token_ids=list(self.token_ids),
            prompt_length=self.prompt_length,
            pre_verify=self.pre_verify,
            accepted_draft_tokens=self.accepted_draft_tokens,
            verified_draft_tokens=self.verified_draft_tokens,
            verification_rounds=self.verification_rounds,
            committed_length=self.committed_length,
            temperature=self.temperature,
            draft_temperature=self.draft_temperature,
            max_tokens=self.max_tokens,
            ignore_eos=self.ignore_eos,
            slo_tpot_ms=self.slo_tpot_ms,
            slo_class=self.slo_class,
            finished_decode_elapsed_ms=self.finished_decode_elapsed_ms,
            num_acc_tokens=list(self.num_acc_tokens or ()),
            cur_acc_tokens=self.cur_acc_tokens,
            pending_window_size=self.pending_window_size,
            continuation_epoch=self.continuation_epoch,
            request_id=self.request_id,
            arrival_ts=self.arrival_ts,
        )

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.prompt_length :]

    @property
    def committed_completion_token_ids(self) -> list[int]:
        assert self.committed_length is not None
        return self.token_ids[self.prompt_length : self.committed_length]

    def apply_target_verification(
        self,
        *,
        gamma: int,
        accepted: int,
        correction_token_id: int | None,
        next_round_token_ids: list[int],
        verification_size: int | None = None,
    ) -> None:
        """Apply nano-PEARL's target-side append/rollback transition."""
        expected = self._verification_size(gamma, verification_size)
        self._validate_verification(
            gamma,
            expected,
            accepted,
            correction_token_id,
            next_round_token_ids,
        )
        was_pre_verify = self.pre_verify
        assert self.committed_length is not None
        self.committed_length += accepted + (accepted < expected)
        self.accepted_draft_tokens += accepted
        self.verified_draft_tokens += expected
        self.verification_rounds += 1
        self.continuation_epoch += 1
        self._record_acceptance(expected=expected, accepted=accepted)
        if accepted == expected:
            self.token_ids.extend(next_round_token_ids)
            self.pre_verify = False
            self.pending_window_size = len(next_round_token_ids)
            return

        assert correction_token_id is not None
        if not was_pre_verify:
            rollout = expected - accepted
            if rollout > 1:
                del self.token_ids[-(rollout - 1) :]
        self.token_ids.append(correction_token_id)
        self.pre_verify = True
        self.pending_window_size = 0

    def apply_draft_verification(
        self,
        *,
        gamma: int,
        accepted: int,
        correction_token_id: int | None,
        next_round_token_ids: list[int],
        verification_size: int | None = None,
    ) -> None:
        """Apply the mirrored draft-side rollback after a target verdict."""
        expected = self._verification_size(gamma, verification_size)
        self._validate_verification(
            gamma,
            expected,
            accepted,
            correction_token_id,
            next_round_token_ids,
        )
        was_pre_verify = self.pre_verify
        assert self.committed_length is not None
        self.committed_length += accepted + (accepted < expected)
        self.accepted_draft_tokens += accepted
        self.verified_draft_tokens += expected
        self.verification_rounds += 1
        self.continuation_epoch += 1
        self._record_acceptance(expected=expected, accepted=accepted)
        if accepted == expected:
            self.pre_verify = False
            self.pending_window_size = len(next_round_token_ids)
            return

        assert correction_token_id is not None
        del self.token_ids[-len(next_round_token_ids) :]
        if not was_pre_verify:
            rollout = expected - accepted
            if rollout > 1:
                del self.token_ids[-(rollout - 1) :]
        self.token_ids.append(correction_token_id)
        self.pre_verify = True
        self.pending_window_size = 0

    def _verification_size(self, gamma: int, verification_size: int | None) -> int:
        if self.pre_verify:
            expected = 1
        elif verification_size is not None:
            expected = int(verification_size)
        elif self.pending_window_size:
            expected = self.pending_window_size
        else:
            # Compatibility for states constructed before variable per-request
            # windows were introduced.
            expected = gamma
        if expected <= 0 or expected > gamma:
            raise ValueError("PEARL verification size must be in [1, gamma].")
        return expected

    def _validate_verification(
        self,
        gamma: int,
        expected: int,
        accepted: int,
        correction_token_id: int | None,
        next_round_token_ids: list[int],
    ) -> None:
        if not 0 < len(next_round_token_ids) <= gamma:
            raise ValueError("Every PEARL next-round window must contain 1..gamma draft tokens.")
        if not 0 <= accepted <= expected:
            raise ValueError("PEARL accepted length is outside the verification window.")
        if accepted == expected and correction_token_id is not None:
            raise ValueError("A fully accepted PEARL window cannot include a correction token.")
        if accepted < expected and correction_token_id is None:
            raise ValueError("A rejected PEARL window requires a target correction token.")

    def _record_acceptance(self, *, expected: int, accepted: int) -> None:
        assert self.num_acc_tokens is not None
        if accepted == expected:
            self.cur_acc_tokens += accepted
        else:
            # Upstream MAT treats the target correction as part of the output
            # segment terminated by this rejection.
            self.num_acc_tokens.append(self.cur_acc_tokens + accepted + 1)
            self.cur_acc_tokens = 0

    @property
    def acceptance_lengths(self) -> list[int]:
        if self.verified_draft_tokens == 0:
            return []
        return [*(self.num_acc_tokens or ()), self.cur_acc_tokens]


@dataclass(frozen=True)
class NativeSamplingParams:
    """Per-request sampling controls supported by upstream nano-PEARL."""

    temperature: float = 1.0
    draft_temperature: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = False
    slo_tpot_ms: float | None = None
    slo_class: str | None = None
    spec_rhythm_max_gamma: int | None = None
    request_id: str | int | None = None
    arrival_ts: float | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0 or self.draft_temperature < 0:
            raise ValueError("PEARL and draft temperatures must be non-negative.")
        if self.max_tokens <= 0:
            raise ValueError("PEARL max_tokens must be positive.")
        if self.slo_tpot_ms is not None and self.slo_tpot_ms <= 0:
            raise ValueError("PEARL TPOT SLO must be positive when supplied.")
        if self.spec_rhythm_max_gamma is not None and self.spec_rhythm_max_gamma <= 0:
            raise ValueError("SpecRhythm per-request max gamma must be positive.")
        if self.arrival_ts is not None and not math.isfinite(self.arrival_ts):
            raise ValueError("SpecRhythm request arrival timestamp must be finite.")


# Match upstream's public name while retaining a native-specific explicit name.
SamplingParams = NativeSamplingParams


@dataclass
class NativeSpecRhythmDevicePayload:
    """Rank-local tensor views for one guarded SpecRhythm proposal."""

    ticket: SpecRhythmProposalTicket
    verification_tokens: torch.Tensor
    next_tokens: torch.Tensor
    verification_size: int
    draft_confidence: float

    def validate_for(self, state: PearlPipelineState) -> None:
        if self.ticket.required_prefix_epoch != state.continuation_epoch:
            raise RuntimeError(
                "SpecRhythm refused a stale device mailbox proposal: "
                f"request={self.ticket.request_index}, "
                f"required_epoch={self.ticket.required_prefix_epoch}, "
                f"current_epoch={state.continuation_epoch}."
            )
        if self.verification_tokens.shape != (self.verification_size,):
            raise RuntimeError("SpecRhythm mailbox verification tensor has an invalid shape.")
        if self.verification_size <= 0:
            raise RuntimeError("SpecRhythm mailbox verification width must be positive.")
        expected_size = (
            1
            if state.pre_verify
            else (state.pending_window_size or self.ticket.gamma)
        )
        if self.verification_size != expected_size:
            raise RuntimeError(
                "SpecRhythm mailbox verification width does not match the request prefix."
            )
        if self.next_tokens.shape != (self.ticket.gamma,):
            raise RuntimeError("SpecRhythm mailbox continuation tensor has an invalid shape.")
        if not math.isfinite(self.draft_confidence) or not 0.0 <= self.draft_confidence <= 1.0:
            raise RuntimeError("SpecRhythm mailbox confidence must be finite and in [0, 1].")


@dataclass(frozen=True)
class NativePearlConfig:
    draft_model: str
    target_model: str
    draft_tp_size: int
    target_tp_size: int
    gamma: int
    max_model_len: int
    max_tokens: int
    draft_dtype: str = "auto"
    target_dtype: str = "auto"
    max_num_seqs: int = 1
    prefill_chunk_size: int | None = None
    auto_gamma_profile_sequence_length: int = AUTO_GAMMA_PROFILE_SEQUENCE_LENGTH
    max_num_queued_seqs: int | None = None
    max_num_batched_tokens: int = 16384
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = PAGED_ATTENTION_BLOCK_SIZE
    num_kvcache_blocks: int = -1
    max_aclgraph_entries: int = 32
    target_verification_graph_buckets: int = TARGET_VERIFICATION_GRAPH_BUCKETS
    target_verification_graph_post_counts: tuple[tuple[int, tuple[int, ...]], ...] = ()
    enable_prefix_caching: bool = True
    enable_continuous_batching: bool = False
    enable_preemptive_scheduling: bool = False
    enable_spec_rhythm: bool = False
    spec_rhythm_min_gamma: int = 1
    spec_rhythm_max_eager_tokens: int = 0
    spec_rhythm_urgency_threshold: float = 0.75
    spec_rhythm_acceptance_floor: float = 0.4
    spec_rhythm_acceptance_ema_alpha: float = 0.2
    spec_rhythm_roofline: Mapping[str, int] | None = None
    spec_rhythm_draft_token_budget: int | None = None
    pad_finished_requests: bool = False
    draft_use_paged_attention: bool = False
    target_use_paged_attention: bool = False
    draft_use_production_rope: bool = True
    target_use_production_rope: bool = True
    precompile_decode_graphs: bool = False
    enable_cpu_binding: bool = True
    profile_decode_steps: int = 0
    stop_after_profiled_decode_steps: bool = False
    enforce_eager: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_verification_graph_post_counts",
            _normalize_target_graph_post_counts(
                self.target_verification_graph_post_counts
            ),
        )
        if self.draft_tp_size <= 0 or self.target_tp_size <= 0:
            raise ValueError("Draft and target TP sizes must be positive.")
        supported_dtypes = {"auto", "bfloat16", "float16"}
        if self.draft_dtype not in supported_dtypes or self.target_dtype not in supported_dtypes:
            raise ValueError("PEARL model dtype must be auto, bfloat16, or float16.")
        if self.gamma == 0 or self.gamma < -1:
            raise ValueError("PEARL gamma must be positive, or -1 for automatic selection.")
        if self.max_model_len <= 0 or self.max_tokens <= 0 or self.max_num_seqs <= 0:
            raise ValueError("PEARL model, generation, and batch limits must be positive.")
        if self.prefill_chunk_size is not None and not 0 < self.prefill_chunk_size <= self.max_num_seqs:
            raise ValueError("PEARL prefill_chunk_size must be in [1, max_num_seqs].")
        if self.auto_gamma_profile_sequence_length <= 0:
            raise ValueError("PEARL auto-gamma profile sequence length must be positive.")
        if self.max_num_queued_seqs is not None and self.max_num_queued_seqs < self.max_num_seqs:
            raise ValueError("PEARL max_num_queued_seqs must be at least max_num_seqs.")
        if self.enable_preemptive_scheduling and not self.enable_continuous_batching:
            raise ValueError("PEARL preemptive scheduling requires continuous batching.")
        if self.enable_spec_rhythm and not (
            self.enable_continuous_batching and self.enable_preemptive_scheduling
        ):
            raise ValueError(
                "SpecRhythm requires PEARL continuous batching and preemptive scheduling."
            )
        if self.enable_spec_rhythm and self.gamma == -1:
            raise ValueError("SpecRhythm requires a fixed maximum PEARL gamma.")
        if self.spec_rhythm_min_gamma <= 0 or (
            self.gamma != -1 and self.spec_rhythm_min_gamma > self.gamma
        ):
            raise ValueError("SpecRhythm min gamma must be in [1, gamma].")
        if self.spec_rhythm_max_eager_tokens < 0 or (
            self.gamma != -1 and self.spec_rhythm_max_eager_tokens > self.gamma
        ):
            raise ValueError("SpecRhythm eager-token cap must be in [0, gamma].")
        if not 0 <= self.spec_rhythm_urgency_threshold:
            raise ValueError("SpecRhythm urgency threshold must be non-negative.")
        if not 0 <= self.spec_rhythm_acceptance_floor <= 1:
            raise ValueError("SpecRhythm acceptance floor must be in [0, 1].")
        if not 0 < self.spec_rhythm_acceptance_ema_alpha <= 1:
            raise ValueError("SpecRhythm acceptance EMA alpha must be in (0, 1].")
        if self.spec_rhythm_roofline is not None:
            object.__setattr__(
                self,
                "spec_rhythm_roofline",
                {str(key): int(value) for key, value in self.spec_rhythm_roofline.items()},
            )
            if any(value <= 0 for value in self.spec_rhythm_roofline.values()):
                raise ValueError("SpecRhythm roofline values must be positive token budgets.")
        if self.spec_rhythm_draft_token_budget is not None and self.spec_rhythm_draft_token_budget <= 0:
            raise ValueError("SpecRhythm draft-token budget must be positive when supplied.")
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("PEARL max_num_batched_tokens must be at least max_model_len.")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("PEARL gpu_memory_utilization must be in (0, 1].")
        if self.kvcache_block_size != PAGED_ATTENTION_BLOCK_SIZE:
            raise ValueError(f"Native Ascend PEARL requires kvcache_block_size={PAGED_ATTENTION_BLOCK_SIZE}.")
        if self.num_kvcache_blocks == 0 or self.num_kvcache_blocks < -1:
            raise ValueError("PEARL num_kvcache_blocks must be positive, or -1 for automatic sizing.")
        if self.max_aclgraph_entries <= 0:
            raise ValueError("PEARL max_aclgraph_entries must be positive.")
        if self.target_verification_graph_buckets <= 0:
            raise ValueError("PEARL target_verification_graph_buckets must be positive.")
        if self.precompile_decode_graphs:
            if self.gamma == -1:
                raise ValueError("PEARL decode graph precompilation requires a fixed gamma.")
            if not self.draft_use_paged_attention or not self.target_use_paged_attention:
                raise ValueError("PEARL decode graph precompilation requires paged attention on both models.")
            graph_batch_sizes = [self.max_num_seqs]
            if self.enable_continuous_batching and self.max_num_seqs > 1:
                graph_batch_sizes.append(max(1, self.max_num_seqs // 2))
            graph_shapes = _target_graph_precompile_shapes(
                graph_batch_sizes,
                self.gamma,
                self.target_verification_graph_buckets,
                self.target_verification_graph_post_counts,
            )
            if len(graph_shapes) > self.max_aclgraph_entries:
                raise ValueError("PEARL decode graph precompilation exceeds max_aclgraph_entries.")
        if self.profile_decode_steps < 0:
            raise ValueError("PEARL profile_decode_steps must be non-negative.")
        if self.stop_after_profiled_decode_steps and self.profile_decode_steps == 0:
            raise ValueError("PEARL profiling-only execution requires profile_decode_steps to be positive.")
        if self.seed is not None and self.seed < 0:
            raise ValueError("PEARL sampling seed must be non-negative.")


class NativePearlEngine:
    """One rank of nano-PEARL's persistent HCCL runtime."""

    def __init__(self, config: NativePearlConfig) -> None:
        self.config = config
        self.gamma = config.gamma
        # Bind this spawn worker before HCCL creates any communicator. Creating
        # the world group on the inherited default device and switching later
        # leaves MC2 resource allocation associated with the wrong NPU.
        self.local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        torch.npu.set_device(self.local_rank)
        # torch.distributed's env:// rendezvous is established by torchrun.
        if not dist.is_initialized():
            _set_default_npu_environment(config.target_tp_size)
            dist.init_process_group(backend="hccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # Match the production vLLM-Ascend model runner so NPU kernels may
        # retain optimized internal layouts (for example FRACTAL_NZ weights).
        torch.npu.config.allow_internal_format = True
        init_device_properties_triton()
        self.device = torch.device("npu")

        self.topology = PearlTopology.from_tensor_parallel_sizes(
            config.draft_tp_size,
            config.target_tp_size,
        )
        self.groups = PearlProcessGroups.create(self.topology, backend="hccl")
        if self.world_size != self.topology.world_size:
            raise ValueError(f"PEARL needs {self.topology.world_size} ranks, received {self.world_size}.")

        self.is_draft = self.groups.is_draft_worker
        self.model_group = self.groups.model_group
        self.model_context = NativeTPContext(
            group=self.model_group,
            rank=self.rank if self.is_draft else self.rank - config.draft_tp_size,
            size=config.draft_tp_size if self.is_draft else config.target_tp_size,
            leader_rank=self.topology.draft_leader_rank if self.is_draft else self.topology.target_leader_rank,
        )

        projection = validate_model_pair(config.draft_model, config.target_model)
        self.draft_vocab_size = projection.draft_vocab_size
        self.target_vocab_size = projection.target_vocab_size
        draft_model_config = AutoConfig.from_pretrained(config.draft_model)
        target_model_config = AutoConfig.from_pretrained(config.target_model)
        draft_model_config.pearl_use_production_rope = config.draft_use_production_rope
        target_model_config.pearl_use_production_rope = config.target_use_production_rope
        for model_config, dtype_name in (
            (draft_model_config, config.draft_dtype),
            (target_model_config, config.target_dtype),
        ):
            if dtype_name != "auto":
                model_config.torch_dtype = getattr(torch, dtype_name)
        if _normalize_eos_tokens(draft_model_config.eos_token_id) != _normalize_eos_tokens(
            target_model_config.eos_token_id
        ):
            raise ValueError("Native PEARL requires identical draft and target EOS token IDs.")
        model_path = config.draft_model if self.is_draft else config.target_model
        model_config = draft_model_config if self.is_draft else target_model_config
        self.model = build_native_model(
            model_config,
            self.model_context,
            config.max_model_len,
            config.max_num_seqs,
            configure_cache=False,
        )
        load_native_model_weights(self.model, model_path)
        num_cache_blocks = self._resolve_num_cache_blocks()
        self.model.configure_cache(
            config.max_model_len,
            self._cache_sequence_capacity,
            config.kvcache_block_size,
            num_cache_blocks,
        )
        attention = self.model.layers[0].self_attn
        assert attention.key_cache is not None
        self.prefix_cache = NativePrefixCache(
            num_blocks=attention.key_cache.shape[0],
            blocks_per_sequence=attention.blocks_per_sequence,
            block_size=attention.block_size,
        )
        self.cache_allocation: NativeCacheAllocation | None = None
        self.cache_block_tables: torch.Tensor | None = None
        self.model.eval()
        if config.seed is not None:
            torch.manual_seed(config.seed)
            torch.npu.manual_seed_all(config.seed)
        self.graph_runner = NativeACLGraphRunner(
            self.model,
            enabled=not config.enforce_eager,
            max_graph_tokens=max(512, config.max_num_seqs * max(1, self.gamma)),
            max_graph_entries=config.max_aclgraph_entries,
        )
        self.last_worker_decode_phase_seconds: dict[str, float] = {}
        self.last_worker_decode_profile_seconds: dict[str, float] = {}
        self.last_worker_profiled_decode_steps = 0
        self.last_worker_decode_counters: dict[str, int] = {}
        self.greedy_verification_layouts: dict[
            tuple[int, tuple[int, ...]],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(config.draft_model)
        # validate_model_pair above permits a target-only vocabulary suffix and
        # verifies that every draft token keeps the same target token ID.
        self.eos_token_ids = _normalize_eos_tokens(target_model_config.eos_token_id)
        self.gamma_profiles: dict[int, int] = {}
        if config.gamma == -1:
            self.gamma_profiles = self._profile_auto_gammas()
        dist.barrier()
        if config.enable_cpu_binding:
            try:
                bind_cpus(self.local_rank)
            except Exception as error:
                logger.warning("Bind cpus failed in PEARL rank%s: %s. Skip CPU binding.", self.local_rank, error)
        self.precompiled_decode_batch_sizes: frozenset[int] = frozenset()
        if config.precompile_decode_graphs:
            self._precompile_decode_graphs()
        dist.barrier()

    def graph_metrics(self) -> dict[str, int | float]:
        """Return this worker's cumulative ACLGraph counters."""
        metrics: dict[str, int | float] = {
            "rank": self.rank,
            "is_draft_rank": int(self.is_draft),
            "aclgraph_entries": len(self.graph_runner.entries) + len(self.graph_runner.draft_entries),
            "aclgraph_captures": self.graph_runner.capture_count,
            "aclgraph_capture_attempts": self.graph_runner.capture_attempt_count,
            "aclgraph_replays": self.graph_runner.replay_count,
            "aclgraph_failed_captures": self.graph_runner.failed_capture_count,
            "aclgraph_capacity_fallbacks": self.graph_runner.capacity_fallback_count,
            "aclgraph_shape_fallbacks": self.graph_runner.shape_fallback_count,
        }
        metrics.update(
            {f"worker_{phase}_seconds": seconds for phase, seconds in self.last_worker_decode_phase_seconds.items()}
        )
        metrics["worker_profiled_decode_steps"] = self.last_worker_profiled_decode_steps
        metrics.update(
            {
                f"worker_profile_{phase}_seconds": seconds
                for phase, seconds in self.last_worker_decode_profile_seconds.items()
            }
        )
        metrics.update({f"worker_{name}": value for name, value in self.last_worker_decode_counters.items()})
        return metrics

    def configure_decode_profiling(
        self,
        profile_decode_steps: int,
        stop_after_profiled_decode_steps: bool = False,
    ) -> None:
        """Change decode profiling between requests without reloading the models."""
        if profile_decode_steps < 0:
            raise ValueError("PEARL profile_decode_steps must be non-negative.")
        if stop_after_profiled_decode_steps and profile_decode_steps == 0:
            raise ValueError("PEARL profiling-only execution requires profile_decode_steps to be positive.")
        self.config = replace(
            self.config,
            profile_decode_steps=profile_decode_steps,
            stop_after_profiled_decode_steps=stop_after_profiled_decode_steps,
        )

    def _resolve_num_cache_blocks(self) -> int:
        if self.config.num_kvcache_blocks > 0:
            return self.config.num_kvcache_blocks
        attention = self.model.layers[0].self_attn
        bytes_per_element = attention.qkv_proj.weight.element_size()
        bytes_per_block = (
            2
            * self.config.kvcache_block_size
            * attention.num_kv_heads
            * attention.head_dim
            * bytes_per_element
            * len(self.model.layers)
        )
        free_memory, total_memory = torch.npu.mem_get_info(self.local_rank)
        used_memory = total_memory - free_memory
        cache_budget = max(0, int(total_memory * self.config.gpu_memory_utilization) - used_memory)
        max_required_blocks = (
            (self.config.max_model_len + self.config.kvcache_block_size - 1)
            // self.config.kvcache_block_size
            * self._cache_sequence_capacity
        )
        num_blocks = min(max_required_blocks, cache_budget // bytes_per_block)
        if num_blocks < self._cache_sequence_capacity:
            raise MemoryError(
                "PEARL cannot reserve one KV cache page per configured sequence; "
                "lower max_num_seqs or increase available NPU memory."
            )
        return num_blocks

    @property
    def _cache_sequence_capacity(self) -> int:
        return self.config.max_num_queued_seqs or self.config.max_num_seqs

    def _allocate_cache(
        self,
        prompts: list[list[int]],
        *,
        enable_prefix_caching: bool,
    ) -> NativeCacheAllocation:
        allocation = self.prefix_cache.allocate(
            prompts,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.cache_allocation = allocation
        self.cache_block_tables = torch.tensor(
            allocation.block_tables,
            dtype=torch.int32,
            device=self.device,
        )
        return allocation

    def _ensure_cache_capacity(
        self,
        sequence_ids: list[int],
        positions: list[int],
    ) -> None:
        if self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before extending it.")
        updates = self.prefix_cache.ensure_capacity(sequence_ids, positions)
        if not updates:
            return
        update_sequences, update_logical_blocks, update_block_ids = zip(*updates)
        self.cache_block_tables[
            torch.tensor(update_sequences, dtype=torch.long, device=self.device),
            torch.tensor(update_logical_blocks, dtype=torch.long, device=self.device),
        ] = torch.tensor(update_block_ids, dtype=torch.int32, device=self.device)

    def _release_cache(self) -> None:
        self.prefix_cache.release()
        self.cache_allocation = None
        self.cache_block_tables = None

    def _cache_slot_mapping(
        self,
        sequence_ids: list[int],
        positions: list[int],
    ) -> list[int]:
        if self.cache_allocation is None:
            raise RuntimeError("Allocate the native PEARL KV cache before building slot mappings.")
        block_size = self.config.kvcache_block_size
        return [
            self.cache_allocation.block_tables[sequence_id][position // block_size] * block_size + position % block_size
            for sequence_id, position in zip(sequence_ids, positions)
        ]

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: list[int],
        sampling_params: NativeSamplingParams | None = None,
    ) -> dict[str, Any] | None:
        results = self.generate_batch([prompt_token_ids], sampling_params)
        return results[0] if results is not None else None

    @torch.inference_mode()
    def generate_batch(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: NativeSamplingParams | Sequence[NativeSamplingParams] | None = None,
        *,
        max_rounds: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Generate a packed batch, optionally scheduling a prefilled request queue."""
        continuous_batching = (
            self.config.enable_continuous_batching
            and max_rounds is None
            and len(prompt_token_ids) > self.config.max_num_seqs
        )
        if not prompt_token_ids or (len(prompt_token_ids) > self.config.max_num_seqs and not continuous_batching):
            raise ValueError("PEARL batch size must fit max_num_seqs unless continuous batching is enabled.")
        if any(len(prompt) > self.config.max_num_batched_tokens for prompt in prompt_token_ids):
            raise ValueError("A PEARL prompt exceeds max_num_batched_tokens.")
        if continuous_batching and len(prompt_token_ids) > self._cache_sequence_capacity:
            raise ValueError("PEARL continuous requests exceed max_num_queued_seqs.")
        request_params = _normalize_sampling_params(len(prompt_token_ids), sampling_params, self.config.max_tokens)
        if max_rounds is not None and max_rounds <= 0:
            raise ValueError("PEARL max_rounds must be positive when supplied.")
        self.gamma = self._auto_select_gamma(prompt_token_ids) if self.config.gamma == -1 else self.config.gamma
        for prompt, params in zip(prompt_token_ids, request_params):
            if not prompt:
                raise ValueError("PEARL generation requires non-empty prompts.")
            completion_capacity = params.max_tokens if max_rounds is None else 1 + (max_rounds + 1) * self.gamma
            if len(prompt) + completion_capacity + self.gamma > self.config.max_model_len:
                raise ValueError("Prompt plus PEARL completion exceeds max_model_len.")

        all_tokens = [[int(token_id) for token_id in prompt] for prompt in prompt_token_ids]
        initial_batch_size = min(len(all_tokens), self.config.max_num_seqs)
        prefill_tokens = all_tokens if continuous_batching else all_tokens[:initial_batch_size]
        prefill_params = request_params if continuous_batching else request_params[:initial_batch_size]
        prefill_chunk_size = self.config.prefill_chunk_size or self.config.max_num_seqs
        if any(
            sum(len(prompt) for prompt in prefill_tokens[start : start + prefill_chunk_size])
            > self.config.max_num_batched_tokens
            for start in range(0, len(prefill_tokens), prefill_chunk_size)
        ):
            raise ValueError("A PEARL prefill chunk exceeds max_num_batched_tokens.")
        self.graph_runner.set_expected_fia_batch_size(initial_batch_size)
        self._allocate_cache(
            prefill_tokens,
            enable_prefix_caching=self.config.enable_prefix_caching,
        )
        draft_states = [
            PearlPipelineState(
                tokens,
                len(tokens),
                temperature=params.temperature,
                draft_temperature=params.draft_temperature,
                max_tokens=params.max_tokens,
                ignore_eos=params.ignore_eos,
                slo_tpot_ms=params.slo_tpot_ms,
                slo_class=params.slo_class,
                request_id=params.request_id,
                arrival_ts=params.arrival_ts,
            )
            for tokens, params in zip(prefill_tokens, prefill_params)
        ]
        target_states = [state.clone() for state in draft_states]
        torch.npu.synchronize()
        prefill_started = time.perf_counter()
        target_tokens: list[int] = []
        for start in range(0, len(prefill_tokens), prefill_chunk_size):
            end = start + prefill_chunk_size
            target_tokens.extend(
                self._prefill_and_sample_target_batch(
                    prefill_tokens[start:end],
                    target_states[start:end],
                    list(range(start, min(end, len(prefill_tokens)))),
                )
            )
        torch.npu.synchronize()
        prefill_elapsed = time.perf_counter() - prefill_started
        for draft_state, target_state, target_token in zip(draft_states, target_states, target_tokens):
            draft_state.token_ids.append(target_token)
            target_state.token_ids.append(target_token)
            assert draft_state.committed_length is not None and target_state.committed_length is not None
            draft_state.committed_length += 1
            target_state.committed_length += 1

        self._capture_decode_graphs(
            prefill_tokens[:initial_batch_size],
            target_tokens[:initial_batch_size],
        )
        torch.npu.synchronize()
        started = time.perf_counter()
        if self.config.enable_spec_rhythm:
            return self._generate_spec_rhythm_decode(
                draft_states=draft_states,
                target_states=target_states,
                request_params=prefill_params,
                initial_batch_size=initial_batch_size,
                continuous_batching=continuous_batching,
                prefill_elapsed=prefill_elapsed,
                started=started,
                max_rounds=max_rounds,
            )
        npu_profiler = None
        npu_profile_dir = os.getenv("VLLM_ASCEND_PEARL_NPU_PROFILE_DIR")
        if npu_profile_dir:
            npu_profile_rank = int(
                os.getenv(
                    "VLLM_ASCEND_PEARL_NPU_PROFILE_RANK",
                    str(self.topology.target_leader_rank),
                )
            )
            if not 0 <= npu_profile_rank < self.topology.world_size:
                raise ValueError(
                    "VLLM_ASCEND_PEARL_NPU_PROFILE_RANK must identify a PEARL worker."
                )
        if npu_profile_dir and self.rank == npu_profile_rank:
            import torch_npu

            npu_profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    npu_profile_dir,
                    worker_name=(
                        f"pearl-{'draft' if self.is_draft else 'target'}-rank-{self.rank}"
                    ),
                ),
                record_shapes=True,
            )
            npu_profiler.start()
        round_count = 0
        completed_states: dict[int, PearlPipelineState] = {}
        completed_request_states: dict[int, PearlPipelineState] = {}
        active_request_indices = list(range(initial_batch_size))
        next_request_index = initial_batch_size
        recent_scheduling_token_gains: list[deque[int]] = [
            deque(maxlen=PREEMPTIVE_SCHEDULING_RECENT_ROUNDS) for _ in all_tokens
        ]
        decode_phase_seconds = {
            "draft": 0.0,
            "target": 0.0,
            "exchange": 0.0,
            "verify": 0.0,
            "broadcast": 0.0,
            "state_update": 0.0,
            "refill": 0.0,
        }
        decode_profile_seconds = {
            "draft_compute": 0.0,
            "draft_to_target_communication": 0.0,
            "target_compute": 0.0,
            "target_verdict": 0.0,
            "target_to_draft_communication": 0.0,
            "wait_sync": 0.0,
            "state_update": 0.0,
        }
        profiled_decode_steps = 0
        target_verification_tokens = 0
        target_model_tokens = 0
        scheduled_sequence_slots = 0
        useful_sequence_slots = 0
        verification_shape_rounds: dict[tuple[int, int], int] = {}
        while True:
            local_states = draft_states if self.is_draft else target_states
            if continuous_batching:
                for request_index, state in completed_request_states.items():
                    local_states[request_index] = state.clone()
                candidate_indices = (
                    range(len(local_states))
                    if self.config.enable_preemptive_scheduling
                    else active_request_indices
                )
                finished_requests = [
                    request_index
                    for request_index in candidate_indices
                    if request_index not in completed_request_states
                    and _finished(local_states[request_index], self.eos_token_ids)
                ]
                for request_index in finished_requests:
                    local_states[request_index].finished_decode_elapsed_ms = (
                        time.perf_counter() - started
                    ) * 1000.0
                    completed_request_states[request_index] = local_states[request_index].clone()
                if self.config.enable_preemptive_scheduling:
                    unfinished_request_indices = [
                        request_index
                        for request_index in range(len(local_states))
                        if request_index not in completed_request_states
                    ]
                    active_request_indices = _select_preemptive_continuous_indices(
                        local_states,
                        unfinished_request_indices,
                        initial_batch_size,
                        recent_scheduling_token_gains,
                        decode_elapsed_ms=(time.perf_counter() - started) * 1000.0,
                        slo_aware=self.config.enable_spec_rhythm,
                    )
                elif finished_requests:
                    finished_set = set(finished_requests)
                    active_request_indices = [
                        request_index for request_index in active_request_indices if request_index not in finished_set
                    ]
                    replacement_count = min(
                        len(finished_requests),
                        len(all_tokens) - next_request_index,
                    )
                    active_request_indices.extend(range(next_request_index, next_request_index + replacement_count))
                    next_request_index += replacement_count
                if len(completed_request_states) == len(all_tokens):
                    break

            if max_rounds is None:
                if self.config.pad_finished_requests and not continuous_batching:
                    _restore_completed_states(
                        local_states,
                        completed_states,
                        self.eos_token_ids,
                    )
                if continuous_batching:
                    active_indices = _continuous_bucket_indices(
                        active_request_indices,
                        completed_request_states,
                        initial_batch_size,
                    )
                else:
                    unfinished_indices = [
                        index for index, state in enumerate(local_states) if not _finished(state, self.eos_token_ids)
                    ]
                    active_indices = (
                        list(range(len(local_states)))
                        if unfinished_indices and self.config.pad_finished_requests
                        else unfinished_indices
                    )
            elif round_count < max_rounds:
                active_indices = list(range(len(local_states)))
            else:
                active_indices = []
            if not active_indices:
                break
            if continuous_batching:
                self.graph_runner.set_expected_fia_batch_size(len(active_indices))
                scheduled_sequence_slots += len(active_indices)
                useful_sequence_slots += sum(
                    request_index not in completed_request_states
                    for request_index in active_indices
                )
            # Request order is not semantically observable, so canonicalize it
            # by verification width. This reduces full-batch target FIA shapes
            # from every binary 1/gamma permutation to one shape per count of
            # pre-verify requests, allowing the bounded graph cache to cover
            # the hot path without any token or KV padding.
            active_indices = _canonical_active_indices(local_states, active_indices)
            pre_verify = [local_states[index].pre_verify for index in active_indices]
            verification_sizes = [1 if value else self.gamma for value in pre_verify]
            verification_shape = (len(active_indices), sum(not value for value in pre_verify))
            verification_shape_rounds[verification_shape] = (
                verification_shape_rounds.get(verification_shape, 0) + 1
            )
            profile_this_round = round_count < self.config.profile_decode_steps
            if profile_this_round:
                torch.npu.synchronize()
            phase_started = time.perf_counter()
            verification_tensor, next_window_tensor = self._draft_round_device_batch(
                draft_states,
                active_indices,
            )
            if profile_this_round:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            decode_phase_seconds["draft"] += phase_elapsed
            if profile_this_round and self.is_draft:
                decode_profile_seconds["draft_compute"] += phase_elapsed
            phase_started = time.perf_counter()
            target_token_windows, target_logits = self._target_round_outputs_batch(
                target_states,
                active_indices,
            )
            if profile_this_round:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            decode_phase_seconds["target"] += phase_elapsed
            if profile_this_round and not self.is_draft:
                decode_profile_seconds["target_compute"] += phase_elapsed
            target_verification_tokens += sum(verification_sizes)
            target_model_tokens += sum(
                _bucket_target_verification_widths(
                    pre_verify,
                    self.gamma,
                    self.config.target_verification_graph_buckets,
                    self.config.target_verification_graph_post_counts,
                )[0]
            )
            if profile_this_round and self.groups.is_verification_worker:
                wait_started = time.perf_counter()
                dist.barrier(group=self.groups.verification_group)
                torch.npu.synchronize()
                decode_profile_seconds["wait_sync"] += time.perf_counter() - wait_started
            phase_started = time.perf_counter()
            draft_message = self._exchange_draft_device_windows(
                verification_tensor,
                next_window_tensor,
                verification_sizes,
            )
            if profile_this_round and self.groups.is_verification_worker:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            decode_phase_seconds["exchange"] += phase_elapsed
            if profile_this_round and self.groups.is_verification_worker:
                decode_profile_seconds["draft_to_target_communication"] += phase_elapsed
            temperatures = [target_states[index].temperature for index in active_indices]
            phase_started = time.perf_counter()
            verdict = self._verify_target_tokens_batch(
                target_token_windows,
                target_logits,
                draft_message,
                verification_sizes,
                temperatures,
            )
            if profile_this_round:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            decode_phase_seconds["verify"] += phase_elapsed
            if profile_this_round and not self.is_draft:
                decode_profile_seconds["target_verdict"] += phase_elapsed
            replicated_target_verdict = all(temperature == 0 for temperature in temperatures)
            if profile_this_round:
                if replicated_target_verdict:
                    participates_in_correction = self.rank in self.topology.correction_ranks
                    correction_group = self.groups.correction_group
                else:
                    participates_in_correction = True
                    correction_group = None
                if participates_in_correction:
                    wait_started = time.perf_counter()
                    dist.barrier(group=correction_group)
                    torch.npu.synchronize()
                    decode_profile_seconds["wait_sync"] += time.perf_counter() - wait_started
            phase_started = time.perf_counter()
            accepted, corrections, synchronized_next = self._broadcast_device_round_result(
                verdict,
                draft_message,
                sum(verification_sizes),
                len(active_indices),
                next_window_tensor,
                replicated_target_verdict=replicated_target_verdict,
                profile_phase_seconds=decode_profile_seconds if profile_this_round else None,
            )
            decode_phase_seconds["broadcast"] += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            committed_lengths_before_update = [
                local_states[sequence_index].committed_length
                for sequence_index in active_indices
            ]
            for batch_index, sequence_index in enumerate(active_indices):
                if self.is_draft:
                    draft_states[sequence_index].token_ids.extend(synchronized_next[batch_index])
                    draft_states[sequence_index].apply_draft_verification(
                        gamma=self.gamma,
                        accepted=accepted[batch_index],
                        correction_token_id=corrections[batch_index],
                        next_round_token_ids=synchronized_next[batch_index],
                    )
                else:
                    target_states[sequence_index].apply_target_verification(
                        gamma=self.gamma,
                        accepted=accepted[batch_index],
                        correction_token_id=corrections[batch_index],
                        next_round_token_ids=synchronized_next[batch_index],
                    )
                if continuous_batching and sequence_index not in completed_request_states:
                    committed_before = committed_lengths_before_update[batch_index]
                    committed_after = local_states[sequence_index].committed_length
                    assert committed_before is not None and committed_after is not None
                    recent_scheduling_token_gains[sequence_index].append(
                        committed_after - committed_before
                    )
            phase_elapsed = time.perf_counter() - phase_started
            decode_phase_seconds["state_update"] += phase_elapsed
            if profile_this_round:
                decode_profile_seconds["state_update"] += phase_elapsed
                profiled_decode_steps += 1
            round_count += 1
            if npu_profiler is not None:
                npu_profiler.step()
            if (
                self.config.stop_after_profiled_decode_steps
                and profiled_decode_steps >= self.config.profile_decode_steps
            ):
                break

        if npu_profiler is not None:
            npu_profiler.stop()
        torch.npu.synchronize()
        decode_elapsed = time.perf_counter() - started
        self.last_worker_decode_phase_seconds = dict(decode_phase_seconds)
        self.last_worker_decode_profile_seconds = dict(decode_profile_seconds)
        self.last_worker_profiled_decode_steps = profiled_decode_steps
        self.last_worker_decode_counters = {
            "target_verification_tokens": target_verification_tokens,
            "target_model_tokens": target_model_tokens,
            "target_padding_tokens": target_model_tokens - target_verification_tokens,
            "scheduled_sequence_slots": scheduled_sequence_slots,
            "useful_sequence_slots": useful_sequence_slots,
            "padded_sequence_slots": scheduled_sequence_slots - useful_sequence_slots,
            **{
                f"verification_shape_b{batch_size}_post{post_verify_count}_rounds": rounds
                for (batch_size, post_verify_count), rounds in sorted(verification_shape_rounds.items())
            },
        }
        elapsed = prefill_elapsed + decode_elapsed
        results = []
        if continuous_batching:
            result_states = [
                (
                    state,
                    self.cache_allocation.num_cached_tokens[request_index],
                )
                for request_index, state in enumerate(
                    _continuous_result_states(local_states, completed_request_states)
                )
            ]
        else:
            result_states = []
            for sequence_index, state in enumerate(target_states):
                if self.config.pad_finished_requests and sequence_index in completed_states:
                    state = completed_states[sequence_index]
                result_states.append((state, self.cache_allocation.num_cached_tokens[sequence_index]))
        for state, num_cached_tokens in result_states:
            acceptance_lengths = state.acceptance_lengths
            completion_token_ids = _truncate_completion(
                state.committed_completion_token_ids,
                self.eos_token_ids,
                state.max_tokens if max_rounds is None else len(state.committed_completion_token_ids),
                state.ignore_eos if max_rounds is None else True,
            )
            request_decode_elapsed_ms = (
                state.finished_decode_elapsed_ms
                if state.finished_decode_elapsed_ms is not None
                else decode_elapsed * 1000.0
            )
            observed_tpot_ms = request_decode_elapsed_ms / max(
                1, len(completion_token_ids)
            )
            slo_attained = (
                None
                if state.slo_tpot_ms is None
                else observed_tpot_ms <= state.slo_tpot_ms
            )
            results.append(
                {
                    "completion_token_ids": completion_token_ids,
                    "request_id": state.request_id,
                    "arrival_ts": state.arrival_ts,
                    "accepted_draft_tokens": state.accepted_draft_tokens,
                    "verified_draft_tokens": state.verified_draft_tokens,
                    "verification_rounds": state.verification_rounds,
                    "acceptance_rate": (
                        state.accepted_draft_tokens / state.verified_draft_tokens
                        if state.verified_draft_tokens
                        else 0.0
                    ),
                    "num_acc_tokens": acceptance_lengths,
                    "mean_accept_tokens": (
                        sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0.0
                    ),
                    "temperature": state.temperature,
                    "draft_temperature": state.draft_temperature,
                    "max_tokens": state.max_tokens,
                    "ignore_eos": state.ignore_eos,
                    "slo_tpot_ms": state.slo_tpot_ms,
                    "slo_class": state.slo_class,
                    "observed_tpot_ms": observed_tpot_ms,
                    "slo_attained": slo_attained,
                    "slo_goodput_tokens": (
                        len(completion_token_ids) if slo_attained is not False else 0
                    ),
                    "round_count": round_count,
                    "decode_phase_seconds": dict(decode_phase_seconds),
                    "elapsed_seconds": elapsed,
                    "prefill_elapsed_seconds": prefill_elapsed,
                    "decode_elapsed_seconds": decode_elapsed,
                    "cached_prompt_tokens": num_cached_tokens,
                    "gamma": self.gamma,
                    **self.graph_metrics(),
                }
            )
        self._release_cache()
        return results if self.rank == self.topology.target_leader_rank else None

    def _generate_spec_rhythm_decode(
        self,
        *,
        draft_states: list[PearlPipelineState],
        target_states: list[PearlPipelineState],
        request_params: Sequence[NativeSamplingParams],
        initial_batch_size: int,
        continuous_batching: bool,
        prefill_elapsed: float,
        started: float,
        max_rounds: int | None,
    ) -> list[dict[str, Any]] | None:
        """Execute the guarded dual-batch pipeline from the SpecRhythm paper."""

        runtime_states = {
            index: SpecRhythmRuntimeState(
                request_index=index,
                home_batch_id=index % 2,
                slo_tpot_ms=params.slo_tpot_ms,
                slo_class=params.slo_class,
                max_gamma=params.spec_rhythm_max_gamma,
            )
            for index, params in enumerate(request_params)
        }
        controller = SpecRhythmPipelineController(runtime_states)
        shaper = SpecRhythmBudgetShaper(
            min_gamma=self.config.spec_rhythm_min_gamma,
            max_gamma=self.gamma,
            acceptance_floor=self.config.spec_rhythm_acceptance_floor,
            acceptance_ema_alpha=self.config.spec_rhythm_acceptance_ema_alpha,
            roofline=self.config.spec_rhythm_roofline,
        )
        local_states = draft_states if self.is_draft else target_states
        active_indices: list[int] = []
        pending_admission = list(range(len(local_states)))
        completed_states: dict[int, PearlPipelineState] = {}
        payloads: dict[int, NativeSpecRhythmDevicePayload] = {}
        last_cycle_ms = 0.0
        round_count = 0
        phase_seconds = {
            "draft": 0.0,
            "target": 0.0,
            "exchange": 0.0,
            "verify": 0.0,
            "broadcast": 0.0,
            "state_update": 0.0,
            "refill": 0.0,
        }
        counters = {
            "spec_rhythm_warmup_steps": 0,
            "spec_rhythm_steady_steps": 0,
            "spec_rhythm_drain_steps": 0,
            "spec_rhythm_normal_proposals": 0,
            "spec_rhythm_eager_proposals": 0,
            "spec_rhythm_eager_promoted": 0,
            "spec_rhythm_eager_invalidated": 0,
            "spec_rhythm_allocated_draft_tokens": 0,
            "spec_rhythm_verified_tokens": 0,
        }
        decode_profile_seconds = {
            "draft_compute": 0.0,
            "draft_to_target_communication": 0.0,
            "target_compute": 0.0,
            "target_verdict": 0.0,
            "target_to_draft_communication": 0.0,
            "wait_sync": 0.0,
            "state_update": 0.0,
        }
        profiled_decode_steps = 0
        npu_profiler = None
        npu_profile_dir = os.getenv("VLLM_ASCEND_PEARL_NPU_PROFILE_DIR")
        if npu_profile_dir:
            npu_profile_rank = int(
                os.getenv(
                    "VLLM_ASCEND_PEARL_NPU_PROFILE_RANK",
                    str(self.topology.target_leader_rank),
                )
            )
            if not 0 <= npu_profile_rank < self.topology.world_size:
                raise ValueError(
                    "VLLM_ASCEND_PEARL_NPU_PROFILE_RANK must identify a PEARL worker."
                )
            if self.rank == npu_profile_rank:
                import torch_npu

                npu_profiler = torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    schedule=torch_npu.profiler.schedule(
                        wait=0, warmup=1, active=3, repeat=1
                    ),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        npu_profile_dir,
                        worker_name=(
                            f"spec-rhythm-{'draft' if self.is_draft else 'target'}-"
                            f"rank-{self.rank}"
                        ),
                    ),
                    record_shapes=True,
                )
                npu_profiler.start()

        def synchronized_wall_time() -> float:
            value = torch.tensor(
                [time.time() if self.rank == self.topology.target_leader_rank else 0.0],
                dtype=torch.float64,
                device=self.device,
            )
            dist.broadcast(value, src=self.topology.target_leader_rank)
            return float(value.cpu().item())

        def admit_ready(now: float) -> None:
            while len(active_indices) < initial_batch_size:
                request_index = next(
                    (
                        index
                        for index in pending_admission
                        if request_params[index].arrival_ts is None
                        or request_params[index].arrival_ts <= now
                    ),
                    None,
                )
                if request_index is None:
                    break
                pending_admission.remove(request_index)
                home_counts = [
                    sum(
                        runtime_states[index].home_batch_id == home
                        for index in active_indices
                    )
                    for home in (0, 1)
                ]
                runtime_states[request_index].home_batch_id = (
                    0 if home_counts[0] <= home_counts[1] else 1
                )
                arrival = request_params[request_index].arrival_ts
                if arrival is not None:
                    runtime_states[request_index].decode_elapsed_ms = max(
                        runtime_states[request_index].decode_elapsed_ms,
                        (now - arrival) * 1000.0,
                    )
                active_indices.append(request_index)

        def invalidate_request(request_index: int) -> None:
            controller.invalidate_request(request_index)
            for proposal_id, payload in tuple(payloads.items()):
                if payload.ticket.request_index == request_index:
                    payloads.pop(proposal_id, None)

        admit_ready(synchronized_wall_time())
        while (active_indices or pending_admission) and (
            max_rounds is None or round_count < max_rounds
        ):
            if not active_indices:
                next_arrival = min(
                    float(request_params[index].arrival_ts)
                    for index in pending_admission
                    if request_params[index].arrival_ts is not None
                )
                delay = max(0.0, next_arrival - synchronized_wall_time())
                if delay:
                    # Poll in short intervals so a malformed far-future trace
                    # cannot make a worker appear hung to the controller.
                    time.sleep(min(delay, 0.05))
                admit_ready(synchronized_wall_time())
                continue
            cycle_started = time.perf_counter()
            plan = controller.build_plan(active_indices)
            if plan.phase.value == "complete":
                raise RuntimeError(
                    "SpecRhythm has active requests but no verifiable or draftable proposal."
                )
            counters[f"spec_rhythm_{plan.phase.value}_steps"] += 1
            target_indices = list(plan.target_request_indices)
            target_payloads: list[NativeSpecRhythmDevicePayload] = []
            for request_index in target_indices:
                ticket = controller.ready[request_index]
                payload = payloads.get(ticket.proposal_id)
                if payload is None:
                    raise RuntimeError("SpecRhythm controller referenced a missing device payload.")
                payload.validate_for(local_states[request_index])
                target_payloads.append(payload)

            eager_candidates: list[int] = []
            if self.config.spec_rhythm_max_eager_tokens:
                projected_wait_ms = max(last_cycle_ms * 2.0, 1e-6)
                eager_candidates = [
                    index
                    for index in plan.eager_candidate_indices
                    if runtime_states[index].urgency(projected_wait_ms)
                    >= self.config.spec_rhythm_urgency_threshold
                    and runtime_states[index].expected_acceptance_benefit
                    >= self.config.spec_rhythm_acceptance_floor
                ]
            normal_indices = list(plan.normal_draft_request_indices)
            work_indices: list[int] = []
            work_budgets: list[int] = []
            work_is_eager: list[bool] = []
            if normal_indices or eager_candidates:
                projected_wait_ms = max(last_cycle_ms * 2.0, 1e-6)
                budget_plan = shaper.shape(
                    plan_id=plan.plan_id,
                    normal_request_indices=normal_indices,
                    eager_request_indices=eager_candidates,
                    states=runtime_states,
                    projected_wait_ms=projected_wait_ms,
                    context_len=max(len(local_states[index].token_ids) for index in active_indices),
                    draft_token_budget=self.config.spec_rhythm_draft_token_budget,
                    batch_size=len(active_indices),
                )
                normal_budgets = dict(budget_plan.normal_budgets)
                eager_budgets = {
                    index: min(value, self.config.spec_rhythm_max_eager_tokens)
                    for index, value in budget_plan.eager_budgets.items()
                    if min(value, self.config.spec_rhythm_max_eager_tokens) > 0
                }
                work_indices = [*normal_budgets, *eager_budgets]
                work_budgets = [
                    *normal_budgets.values(),
                    *eager_budgets.values(),
                ]
                work_is_eager = [False] * len(normal_budgets) + [True] * len(eager_budgets)
                counters["spec_rhythm_allocated_draft_tokens"] += sum(work_budgets)

            target_payload_by_index = {
                payload.ticket.request_index: payload for payload in target_payloads
            }
            work_verification_sizes = []
            work_verification_prefixes: list[torch.Tensor | None] = []
            for request_index, eager in zip(work_indices, work_is_eager):
                if eager:
                    parent = target_payload_by_index[request_index]
                    work_verification_sizes.append(parent.ticket.gamma)
                    work_verification_prefixes.append(parent.next_tokens[1:])
                else:
                    work_verification_sizes.append(
                        1
                        if local_states[request_index].pre_verify
                        else (
                            local_states[request_index].pending_window_size
                            or self.gamma
                        )
                    )
                    work_verification_prefixes.append(None)
            tickets = [
                controller.new_ticket(index, gamma=budget, eager=eager)
                for index, budget, eager in zip(
                    work_indices, work_budgets, work_is_eager
                )
            ]

            profile_this_round = round_count < self.config.profile_decode_steps
            if profile_this_round:
                torch.npu.synchronize()
            phase_started = time.perf_counter()
            draft_verification, draft_next, draft_confidence = (
                self._draft_spec_rhythm_device_batch(
                    draft_states,
                    work_indices,
                    work_budgets,
                    verification_sizes=work_verification_sizes,
                    verification_prefixes=work_verification_prefixes,
                )
                if work_indices
                else (None, None, None)
            )
            if self.is_draft and draft_next is not None:
                for row, (request_index, budget) in enumerate(
                    zip(work_indices, work_budgets)
                ):
                    draft_states[request_index].token_ids.extend(
                        int(value)
                        for value in draft_next[row, :budget].detach().cpu().tolist()
                    )
            if profile_this_round:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            phase_seconds["draft"] += phase_elapsed
            if profile_this_round and self.is_draft:
                decode_profile_seconds["draft_compute"] += phase_elapsed

            current_verification_sizes = [
                payload.verification_size for payload in target_payloads
            ]
            phase_started = time.perf_counter()
            target_tokens, target_logits = (
                self._target_round_outputs_batch(
                    target_states,
                    target_indices,
                    current_verification_sizes,
                )
                if target_indices
                else (None, None)
            )
            if profile_this_round:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            phase_seconds["target"] += phase_elapsed
            if profile_this_round and not self.is_draft:
                decode_profile_seconds["target_compute"] += phase_elapsed

            if profile_this_round and self.groups.is_verification_worker:
                wait_started = time.perf_counter()
                dist.barrier(group=self.groups.verification_group)
                torch.npu.synchronize()
                decode_profile_seconds["wait_sync"] += (
                    time.perf_counter() - wait_started
                )
            phase_started = time.perf_counter()
            exchanged, confidences = self._exchange_spec_rhythm_device_proposals(
                draft_verification,
                draft_next,
                draft_confidence,
                tickets,
                work_verification_sizes,
            )
            if tickets:
                new_payloads = self._materialize_spec_rhythm_payloads(
                    tickets=tickets,
                    verification_sizes=work_verification_sizes,
                    local_verification=draft_verification,
                    local_next_windows=draft_next,
                    exchanged_message=exchanged,
                    draft_confidences=confidences,
                )
                controller.publish(tickets)
                payloads.update(
                    (payload.ticket.proposal_id, payload) for payload in new_payloads
                )
                counters["spec_rhythm_normal_proposals"] += sum(
                    not value for value in work_is_eager
                )
                counters["spec_rhythm_eager_proposals"] += sum(work_is_eager)
            if profile_this_round and self.groups.is_verification_worker:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            phase_seconds["exchange"] += phase_elapsed
            if profile_this_round and self.groups.is_verification_worker:
                decode_profile_seconds["draft_to_target_communication"] += phase_elapsed

            finished_this_cycle: list[int] = []
            if target_indices:
                temperatures = [target_states[index].temperature for index in target_indices]
                if self.is_draft:
                    current_message = None
                else:
                    verification_tensor = torch.cat(
                        [payload.verification_tokens for payload in target_payloads]
                    )
                    continuation_tensor = torch.full(
                        (len(target_payloads), self.gamma),
                        -1,
                        dtype=torch.long,
                        device=self.device,
                    )
                    for row, payload in enumerate(target_payloads):
                        continuation_tensor[row, : payload.ticket.gamma] = payload.next_tokens
                    current_message = torch.cat(
                        (verification_tensor, continuation_tensor.flatten())
                    )
                phase_started = time.perf_counter()
                verdict = self._verify_target_tokens_batch(
                    target_tokens,
                    target_logits,
                    current_message,
                    current_verification_sizes,
                    temperatures,
                )
                if profile_this_round:
                    torch.npu.synchronize()
                phase_elapsed = time.perf_counter() - phase_started
                phase_seconds["verify"] += phase_elapsed
                if profile_this_round and not self.is_draft:
                    decode_profile_seconds["target_verdict"] += phase_elapsed
                current_next = torch.full(
                    (len(target_payloads), self.gamma),
                    -1,
                    dtype=torch.long,
                    device=self.device,
                )
                for row, payload in enumerate(target_payloads):
                    current_next[row, : payload.ticket.gamma] = payload.next_tokens
                replicated_target_verdict = all(
                    temperature == 0 for temperature in temperatures
                )
                if profile_this_round:
                    if replicated_target_verdict:
                        participates_in_correction = (
                            self.rank in self.topology.correction_ranks
                        )
                        correction_group = self.groups.correction_group
                    else:
                        participates_in_correction = True
                        correction_group = None
                    if participates_in_correction:
                        wait_started = time.perf_counter()
                        dist.barrier(group=correction_group)
                        torch.npu.synchronize()
                        decode_profile_seconds["wait_sync"] += (
                            time.perf_counter() - wait_started
                        )
                phase_started = time.perf_counter()
                accepted, corrections, synchronized_next = self._broadcast_device_round_result(
                    verdict,
                    current_message,
                    sum(current_verification_sizes),
                    len(target_indices),
                    current_next if self.is_draft else None,
                    replicated_target_verdict=replicated_target_verdict,
                    next_window_sizes=[
                        payload.ticket.gamma for payload in target_payloads
                    ],
                    profile_phase_seconds=(
                        decode_profile_seconds if profile_this_round else None
                    ),
                )
                phase_seconds["broadcast"] += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                for row, request_index in enumerate(target_indices):
                    payload = target_payloads[row]
                    expected = current_verification_sizes[row]
                    fully_accepted = accepted[row] == expected
                    eager_ticket = controller.staged_eager.get(request_index)
                    if self.is_draft:
                        if eager_ticket is not None and not fully_accepted:
                            del draft_states[request_index].token_ids[-eager_ticket.gamma :]
                        draft_states[request_index].apply_draft_verification(
                            gamma=self.gamma,
                            accepted=accepted[row],
                            correction_token_id=corrections[row],
                            next_round_token_ids=synchronized_next[row],
                            verification_size=expected,
                        )
                    else:
                        target_states[request_index].apply_target_verification(
                            gamma=self.gamma,
                            accepted=accepted[row],
                            correction_token_id=corrections[row],
                            next_round_token_ids=synchronized_next[row],
                            verification_size=expected,
                        )
                    delivered = accepted[row] + int(not fully_accepted)
                    promoted = controller.finish_verification(
                        request_index,
                        fully_accepted=fully_accepted,
                        proposed_tokens=expected,
                        accepted_tokens=accepted[row],
                        delivered_tokens=delivered,
                        draft_confidence=payload.draft_confidence,
                        ema_alpha=self.config.spec_rhythm_acceptance_ema_alpha,
                    )
                    payloads.pop(payload.ticket.proposal_id, None)
                    if eager_ticket is not None:
                        if promoted is None:
                            counters["spec_rhythm_eager_invalidated"] += 1
                            payloads.pop(eager_ticket.proposal_id, None)
                        else:
                            counters["spec_rhythm_eager_promoted"] += 1
                    counters["spec_rhythm_verified_tokens"] += expected
                    if _finished(local_states[request_index], self.eos_token_ids):
                        finished_this_cycle.append(request_index)
                phase_seconds["state_update"] += time.perf_counter() - phase_started
                if profile_this_round:
                    decode_profile_seconds["state_update"] += (
                        time.perf_counter() - phase_started
                    )
            else:
                # Warmup has no target verdict/correction collective. Keep every
                # rank at the same service-step boundary before rotating roles.
                dist.barrier()

            if profile_this_round:
                profiled_decode_steps += 1
            if npu_profiler is not None:
                npu_profiler.step()

            cycle_elapsed = time.perf_counter() - cycle_started
            elapsed_tensor = torch.tensor(
                [cycle_elapsed if self.rank == self.topology.target_leader_rank else 0.0],
                dtype=torch.float64,
                device=self.device,
            )
            dist.broadcast(elapsed_tensor, src=self.topology.target_leader_rank)
            last_cycle_ms = float(elapsed_tensor.cpu().item()) * 1000.0
            for index in active_indices:
                runtime_states[index].add_decode_time(last_cycle_ms)

            if finished_this_cycle:
                finished_set = set(finished_this_cycle)
                for request_index in finished_this_cycle:
                    local_states[request_index].finished_decode_elapsed_ms = runtime_states[
                        request_index
                    ].decode_elapsed_ms
                    completed_states[request_index] = local_states[request_index].clone()
                    invalidate_request(request_index)
                active_indices = [
                    index for index in active_indices if index not in finished_set
                ]
            refill_started = time.perf_counter()
            admit_ready(synchronized_wall_time())
            phase_seconds["refill"] += time.perf_counter() - refill_started
            round_count += 1
            if (
                self.config.stop_after_profiled_decode_steps
                and profiled_decode_steps >= self.config.profile_decode_steps
            ):
                break

        for request_index in active_indices:
            invalidate_request(request_index)
        if npu_profiler is not None:
            npu_profiler.stop()
        torch.npu.synchronize()
        decode_elapsed = time.perf_counter() - started
        self.last_worker_decode_phase_seconds = dict(phase_seconds)
        self.last_worker_decode_profile_seconds = dict(decode_profile_seconds)
        self.last_worker_profiled_decode_steps = profiled_decode_steps
        self.last_worker_decode_counters = dict(counters)
        elapsed = prefill_elapsed + decode_elapsed
        if continuous_batching:
            result_states = [
                completed_states.get(index, local_states[index])
                for index in range(len(local_states))
            ]
        else:
            result_states = target_states
        results: list[dict[str, Any]] = []
        for sequence_index, state in enumerate(result_states):
            completion_token_ids = _truncate_completion(
                state.committed_completion_token_ids,
                self.eos_token_ids,
                state.max_tokens
                if max_rounds is None
                else len(state.committed_completion_token_ids),
                state.ignore_eos if max_rounds is None else True,
            )
            acceptance_lengths = state.acceptance_lengths
            request_decode_elapsed_ms = (
                state.finished_decode_elapsed_ms
                if state.finished_decode_elapsed_ms is not None
                else runtime_states[sequence_index].decode_elapsed_ms
            )
            observed_tpot_ms = request_decode_elapsed_ms / max(
                1, len(completion_token_ids)
            )
            slo_attained = (
                None
                if state.slo_tpot_ms is None
                else observed_tpot_ms <= state.slo_tpot_ms
            )
            results.append(
                {
                    "completion_token_ids": completion_token_ids,
                    "accepted_draft_tokens": state.accepted_draft_tokens,
                    "verified_draft_tokens": state.verified_draft_tokens,
                    "verification_rounds": state.verification_rounds,
                    "acceptance_rate": (
                        state.accepted_draft_tokens / state.verified_draft_tokens
                        if state.verified_draft_tokens
                        else 0.0
                    ),
                    "num_acc_tokens": acceptance_lengths,
                    "mean_accept_tokens": (
                        sum(acceptance_lengths) / len(acceptance_lengths)
                        if acceptance_lengths
                        else 0.0
                    ),
                    "temperature": state.temperature,
                    "draft_temperature": state.draft_temperature,
                    "max_tokens": state.max_tokens,
                    "ignore_eos": state.ignore_eos,
                    "slo_tpot_ms": state.slo_tpot_ms,
                    "slo_class": state.slo_class,
                    "request_id": request_params[sequence_index].request_id,
                    "arrival_ts": request_params[sequence_index].arrival_ts,
                    "observed_tpot_ms": observed_tpot_ms,
                    "slo_attained": slo_attained,
                    "slo_goodput_tokens": (
                        len(completion_token_ids) if slo_attained is not False else 0
                    ),
                    "round_count": round_count,
                    "decode_phase_seconds": dict(phase_seconds),
                    "spec_rhythm": dict(counters),
                    "elapsed_seconds": elapsed,
                    "prefill_elapsed_seconds": prefill_elapsed,
                    "decode_elapsed_seconds": decode_elapsed,
                    "cached_prompt_tokens": self.cache_allocation.num_cached_tokens[
                        sequence_index
                    ],
                    "gamma": self.gamma,
                    **self.graph_metrics(),
                }
            )
        self._release_cache()
        return results if self.rank == self.topology.target_leader_rank else None

    @torch.inference_mode()
    def generate_target_ar_batch(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: NativeSamplingParams | Sequence[NativeSamplingParams] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Generate a static batch autoregressively on the target model only."""
        if not prompt_token_ids or len(prompt_token_ids) > self.config.max_num_seqs:
            raise ValueError("PEARL batch size must be between one and max_num_seqs.")
        if sum(len(prompt) for prompt in prompt_token_ids) > self.config.max_num_batched_tokens:
            raise ValueError("PEARL prompts exceed max_num_batched_tokens.")
        request_params = _normalize_sampling_params(len(prompt_token_ids), sampling_params, self.config.max_tokens)
        for prompt, params in zip(prompt_token_ids, request_params):
            if not prompt:
                raise ValueError("Target AR generation requires non-empty prompts.")
            if len(prompt) + params.max_tokens > self.config.max_model_len:
                raise ValueError("Prompt plus target completion exceeds max_model_len.")

        if self.is_draft:
            return None

        initial_tokens = [[int(token_id) for token_id in prompt] for prompt in prompt_token_ids]
        self._allocate_cache(
            initial_tokens,
            enable_prefix_caching=self.config.enable_prefix_caching,
        )
        states = [
            PearlPipelineState(
                tokens,
                len(tokens),
                temperature=params.temperature,
                draft_temperature=params.draft_temperature,
                max_tokens=params.max_tokens,
                ignore_eos=params.ignore_eos,
                slo_tpot_ms=params.slo_tpot_ms,
                slo_class=params.slo_class,
                request_id=params.request_id,
                arrival_ts=params.arrival_ts,
            )
            for tokens, params in zip(initial_tokens, request_params)
        ]
        torch.npu.synchronize()
        prefill_started = time.perf_counter()
        prefill_chunk_size = self.config.prefill_chunk_size or self.config.max_num_seqs
        target_tokens: list[int] = []
        for start in range(0, len(initial_tokens), prefill_chunk_size):
            end = start + prefill_chunk_size
            target_tokens.extend(
                self._target_ar_prefill(
                    initial_tokens[start:end],
                    states[start:end],
                    list(range(start, min(end, len(initial_tokens)))),
                )
            )
        torch.npu.synchronize()
        prefill_elapsed = time.perf_counter() - prefill_started
        for state, target_token in zip(states, target_tokens):
            state.token_ids.append(target_token)
            assert state.committed_length is not None
            state.committed_length += 1
        self._capture_target_ar_graph(states)

        torch.npu.synchronize()
        started = time.perf_counter()
        while True:
            active_indices = [index for index, state in enumerate(states) if not _finished(state, self.eos_token_ids)]
            if not active_indices:
                break
            input_token_ids = [states[index].token_ids[-1] for index in active_indices]
            positions = [len(states[index].token_ids) - 1 for index in active_indices]
            next_tokens = self._run_packed_sample(
                input_token_ids,
                active_indices,
                positions,
                [states[index].temperature for index in active_indices],
            )
            if any(states[index].temperature > 0 for index in active_indices):
                dist.broadcast(
                    next_tokens,
                    src=self.topology.target_leader_rank,
                    group=self.groups.target_group,
                )
            for sequence_index, token_id in zip(active_indices, next_tokens.cpu().tolist()):
                states[sequence_index].token_ids.append(int(token_id))
                assert states[sequence_index].committed_length is not None
                states[sequence_index].committed_length += 1

        torch.npu.synchronize()
        decode_elapsed = time.perf_counter() - started
        elapsed = prefill_elapsed + decode_elapsed
        results = []
        for sequence_index, state in enumerate(states):
            results.append(
                {
                    "completion_token_ids": _truncate_completion(
                        state.committed_completion_token_ids,
                        self.eos_token_ids,
                        state.max_tokens,
                        state.ignore_eos,
                    ),
                    "request_id": state.request_id,
                    "arrival_ts": state.arrival_ts,
                    "accepted_draft_tokens": 0,
                    "verified_draft_tokens": 0,
                    "acceptance_rate": 0.0,
                    "num_acc_tokens": [],
                    "mean_accept_tokens": 0.0,
                    "temperature": state.temperature,
                    "draft_temperature": state.draft_temperature,
                    "max_tokens": state.max_tokens,
                    "ignore_eos": state.ignore_eos,
                    "slo_tpot_ms": state.slo_tpot_ms,
                    "slo_class": state.slo_class,
                    "observed_tpot_ms": (
                        decode_elapsed * 1000.0
                        / max(1, len(state.committed_completion_token_ids))
                    ),
                    "slo_attained": (
                        None
                        if state.slo_tpot_ms is None
                        else (
                            decode_elapsed * 1000.0
                            / max(1, len(state.committed_completion_token_ids))
                            <= state.slo_tpot_ms
                        )
                    ),
                    "slo_goodput_tokens": (
                        len(state.committed_completion_token_ids)
                        if state.slo_tpot_ms is None
                        or (
                            decode_elapsed * 1000.0
                            / max(1, len(state.committed_completion_token_ids))
                            <= state.slo_tpot_ms
                        )
                        else 0
                    ),
                    "elapsed_seconds": elapsed,
                    "prefill_elapsed_seconds": prefill_elapsed,
                    "decode_elapsed_seconds": decode_elapsed,
                    "cached_prompt_tokens": self.cache_allocation.num_cached_tokens[sequence_index],
                    "gamma": 0,
                    **self.graph_metrics(),
                }
            )
        self._release_cache()
        return results if self.rank == self.topology.target_leader_rank else None

    def _run_packed_hidden(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if self.cache_allocation is None:
            raise RuntimeError("Allocate the native PEARL KV cache before running the model.")
        input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=self.device)
        return self._run_device_packed_hidden(
            input_ids,
            sequence_ids,
            positions,
            use_aclgraph,
            logit_indices,
            use_fused_infer_attention,
        )

    def _run_device_packed_hidden(
        self,
        input_ids: torch.Tensor,
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before running the model.")
        position_tensor, attention_metadata = self._prepare_attention_metadata(
            sequence_ids,
            positions,
            use_fused_infer_attention,
        )
        if use_aclgraph:
            hidden_states = self.graph_runner(input_ids, position_tensor, attention_metadata)
        else:
            hidden_states = self.model(input_ids, position_tensor, attention_metadata)
        if logit_indices is not None:
            hidden_states = hidden_states[logit_indices]
        return hidden_states

    def _prepare_attention_metadata(
        self,
        sequence_ids: list[int],
        positions: list[int],
        use_fused_infer_attention: bool,
    ) -> tuple[torch.Tensor, Any]:
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before running the model.")
        self._ensure_cache_capacity(sequence_ids, positions)
        slot_mapping = self._cache_slot_mapping(sequence_ids, positions)
        return self.model.make_attention_metadata(
            sequence_ids,
            positions,
            self.cache_block_tables,
            slot_mapping,
            use_fused_infer_attention,
        )

    def _run_packed_model(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        hidden_states = self._run_packed_hidden(
            input_token_ids,
            sequence_ids,
            positions,
            use_aclgraph,
            logit_indices,
            use_fused_infer_attention,
        )
        return self.model.compute_logits(hidden_states)

    def _run_packed_greedy(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if logit_indices is not None or not use_aclgraph:
            hidden_states = self._run_packed_hidden(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph,
                logit_indices,
                use_fused_infer_attention,
            )
            return self.model.compute_greedy_tokens(hidden_states, self.draft_vocab_size)
        input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=self.device)
        return self._run_device_packed_greedy(
            input_ids,
            sequence_ids,
            positions,
            use_aclgraph=True,
            use_fused_infer_attention=use_fused_infer_attention,
        )

    def _run_packed_sample(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        temperatures: list[float],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if all(temperature == 0 for temperature in temperatures):
            return self._run_packed_greedy(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph,
                logit_indices,
                use_fused_infer_attention,
            )
        logits = self._run_packed_model(
            input_token_ids,
            sequence_ids,
            positions,
            use_aclgraph,
            logit_indices,
            use_fused_infer_attention,
        )[:, : self.draft_vocab_size]
        return _sample_logits(logits, temperatures)

    def _run_device_packed_greedy(
        self,
        input_ids: torch.Tensor,
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before running the model.")
        position_tensor, attention_metadata = self._prepare_attention_metadata(
            sequence_ids,
            positions,
            use_fused_infer_attention,
        )
        if use_aclgraph:
            return self.graph_runner.run_greedy(
                input_ids,
                position_tensor,
                attention_metadata,
                self.draft_vocab_size,
            )
        hidden_states = self.model(input_ids, position_tensor, attention_metadata)
        return self.model.compute_greedy_tokens(hidden_states, self.draft_vocab_size)

    def _run_device_packed_greedy_with_confidence(
        self,
        input_ids: torch.Tensor,
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        use_fused_infer_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run greedy drafting and retain the confidence used by SpecRhythm."""

        hidden_states = self._run_device_packed_hidden(
            input_ids,
            sequence_ids,
            positions,
            use_aclgraph=use_aclgraph,
            use_fused_infer_attention=use_fused_infer_attention,
        )
        return self.model.compute_greedy_tokens_with_confidence(
            hidden_states, self.draft_vocab_size
        )

    def _run_device_packed_sample_with_confidence(
        self,
        input_ids: torch.Tensor,
        sequence_ids: list[int],
        positions: list[int],
        temperatures: Sequence[float],
        *,
        use_aclgraph: bool = True,
        use_fused_infer_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample draft proposals and retain confidence for SpecRhythm.

        Draft temperature is an optional proposal policy.  The target still
        owns verification and correction, so a greedy target remains exact
        even when this proposal is stochastic.
        """

        if len(temperatures) != input_ids.shape[0] or any(value <= 0 for value in temperatures):
            raise ValueError("stochastic draft temperatures must be positive and row-aligned")
        hidden_states = self._run_device_packed_hidden(
            input_ids,
            sequence_ids,
            positions,
            use_aclgraph=use_aclgraph,
            use_fused_infer_attention=use_fused_infer_attention,
        )
        logits = self.model.compute_logits(hidden_states)[:, : self.draft_vocab_size]
        temperature_tensor = torch.tensor(
            temperatures, dtype=torch.float32, device=logits.device
        ).unsqueeze(1)
        probabilities = torch.softmax(logits.float() / temperature_tensor, dim=-1)
        tokens = _sample_logits(logits, temperatures)
        return tokens, probabilities.max(dim=-1).values

    def _target_ar_prefill(
        self,
        prompts: list[list[int]],
        states: list[PearlPipelineState],
        sequence_ids: list[int] | None = None,
    ) -> list[int]:
        if sequence_ids is None:
            sequence_ids = list(range(len(prompts)))
        if len(sequence_ids) != len(prompts):
            raise ValueError("Every target AR prefill prompt requires one cache sequence ID.")
        input_token_ids: list[int] = []
        packed_sequence_ids: list[int] = []
        positions: list[int] = []
        last_token_indices: list[int] = []
        for sequence_id, prompt in zip(sequence_ids, prompts):
            assert self.cache_allocation is not None
            cached_tokens = self.cache_allocation.num_cached_tokens[sequence_id]
            input_token_ids.extend(prompt[cached_tokens:])
            packed_sequence_ids.extend([sequence_id] * (len(prompt) - cached_tokens))
            positions.extend(range(cached_tokens, len(prompt)))
            last_token_indices.append(len(input_token_ids) - 1)
        token_ids = self._run_packed_sample(
            input_token_ids,
            packed_sequence_ids,
            positions,
            [state.temperature for state in states],
            use_aclgraph=False,
            logit_indices=last_token_indices,
        )
        if any(state.temperature > 0 for state in states):
            dist.broadcast(
                token_ids,
                src=self.topology.target_leader_rank,
                group=self.groups.target_group,
            )
        return [int(token_id) for token_id in token_ids.cpu().tolist()]

    def _capture_target_ar_graph(self, states: list[PearlPipelineState]) -> None:
        if not self.config.enforce_eager:
            sequence_ids = list(range(len(states)))
            input_ids = torch.tensor(
                [state.token_ids[-1] for state in states],
                dtype=torch.long,
                device=self.device,
            )
            self._run_device_packed_greedy(
                input_ids,
                sequence_ids,
                [len(state.token_ids) - 1 for state in states],
            )
        dist.barrier(group=self.groups.target_group)

    def _prefill_and_sample_target_batch(
        self,
        prompts: list[list[int]],
        states: list[PearlPipelineState],
        sequence_ids: list[int] | None = None,
    ) -> list[int]:
        if sequence_ids is None:
            sequence_ids = list(range(len(prompts)))
        if len(sequence_ids) != len(prompts):
            raise ValueError("Every PEARL prefill prompt requires one cache sequence ID.")
        input_token_ids: list[int] = []
        packed_sequence_ids: list[int] = []
        positions: list[int] = []
        last_token_indices: list[int] = []
        for sequence_id, prompt in zip(sequence_ids, prompts):
            assert self.cache_allocation is not None
            cached_tokens = self.cache_allocation.num_cached_tokens[sequence_id]
            input_token_ids.extend(prompt[cached_tokens:])
            packed_sequence_ids.extend([sequence_id] * (len(prompt) - cached_tokens))
            positions.extend(range(cached_tokens, len(prompt)))
            last_token_indices.append(len(input_token_ids) - 1)
        if self.is_draft:
            # The target sample is the common committed frontier, but the draft
            # prefill still runs so its persistent KV cache is populated.
            self._run_packed_hidden(
                input_token_ids,
                packed_sequence_ids,
                positions,
                use_aclgraph=False,
                logit_indices=last_token_indices,
            )
            token_ids = torch.zeros(len(prompts), dtype=torch.long, device=self.device)
        else:
            token_ids = self._run_packed_sample(
                input_token_ids,
                packed_sequence_ids,
                positions,
                [state.temperature for state in states],
                use_aclgraph=False,
                logit_indices=last_token_indices,
            )
        dist.broadcast(token_ids, src=self.topology.target_leader_rank)
        return [int(token_id) for token_id in token_ids.cpu().tolist()]

    def _auto_select_gamma(self, prompts: list[list[int]]) -> int:
        if not self.gamma_profiles:
            raise RuntimeError("PEARL auto-gamma profiles were not initialized.")
        batch_size = len(prompts)
        bucket = next(
            (size for size in self.gamma_profiles if size >= batch_size),
            max(self.gamma_profiles),
        )
        return self.gamma_profiles[bucket]

    def _profile_auto_gammas(self) -> dict[int, int]:
        profile_length = min(
            self.config.auto_gamma_profile_sequence_length,
            self.config.max_model_len - 1,
        )
        if profile_length <= 0:
            raise ValueError("PEARL max_model_len is too small for automatic gamma profiling.")
        batch_sizes = [
            batch_size
            for batch_size in AUTO_GAMMA_BATCH_SIZES
            if batch_size <= self.config.max_num_seqs
            and batch_size * profile_length <= self.config.max_num_batched_tokens
        ]
        if not batch_sizes:
            raise ValueError("PEARL batch limits cannot fit the automatic gamma profile.")

        profiles: dict[int, int] = {}
        for batch_size in batch_sizes:
            prompts = [[0] * profile_length for _ in range(batch_size)]
            self._allocate_cache(prompts, enable_prefix_caching=False)
            input_token_ids = [token for prompt in prompts for token in prompt]
            sequence_ids = [index for index in range(batch_size) for _ in range(profile_length)]
            positions = list(range(profile_length)) * batch_size
            last_token_indices = [(index + 1) * profile_length - 1 for index in range(batch_size)]
            decode_tokens = self._run_packed_greedy(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph=False,
                logit_indices=last_token_indices,
            )
            decode_positions = [profile_length] * batch_size
            decode_sequence_ids = list(range(batch_size))
            for _ in range(AUTO_GAMMA_WARMUP_STEPS):
                self._run_device_packed_greedy(
                    decode_tokens,
                    decode_sequence_ids,
                    decode_positions,
                )
            torch.npu.synchronize()
            dist.barrier()
            started = time.perf_counter()
            for _ in range(AUTO_GAMMA_PROFILE_STEPS):
                self._run_device_packed_greedy(
                    decode_tokens,
                    decode_sequence_ids,
                    decode_positions,
                )
            torch.npu.synchronize()
            elapsed = time.perf_counter() - started
            local_speed = AUTO_GAMMA_PROFILE_STEPS / elapsed
            speeds = torch.zeros(2, dtype=torch.float32, device=self.device)
            if self.rank == self.topology.draft_leader_rank:
                speeds[0] = local_speed
            if self.rank == self.topology.target_leader_rank:
                speeds[1] = local_speed
            dist.all_reduce(speeds)
            draft_speed, target_speed = (float(value) for value in speeds.cpu().tolist())
            profiles[batch_size] = _gamma_from_decode_speeds(draft_speed, target_speed)
            self._release_cache()
        return profiles

    def _capture_decode_graphs(self, prompts: list[list[int]], target_tokens: list[int]) -> None:
        if self.config.enforce_eager or self.gamma > 16:
            return
        if len(prompts) in self.precompiled_decode_batch_sizes:
            dist.barrier()
            return
        sequence_ids = list(range(len(prompts)))
        first_decode_positions = [len(prompt) for prompt in prompts]
        input_ids = torch.tensor(target_tokens, dtype=torch.long, device=self.device)
        target_use_fia = not self.config.target_use_paged_attention
        if not self.is_draft:
            self._run_device_packed_greedy(
                input_ids,
                sequence_ids,
                first_decode_positions,
                use_fused_infer_attention=target_use_fia,
            )
        if not self.is_draft and self.gamma > 1:
            packed_tokens = [token_id for token_id in target_tokens for _ in range(self.gamma)]
            packed_sequence_ids = [sequence_id for sequence_id in sequence_ids for _ in range(self.gamma)]
            packed_positions = [
                position for prompt in prompts for position in range(len(prompt) + 1, len(prompt) + self.gamma + 1)
            ]
            packed_input_ids = torch.tensor(packed_tokens, dtype=torch.long, device=self.device)
            self._run_device_packed_greedy(
                packed_input_ids,
                packed_sequence_ids,
                packed_positions,
                use_fused_infer_attention=target_use_fia,
            )
        dist.barrier()

    def _precompile_decode_graphs(self) -> None:
        """Build stable paged-attention graphs before serving requests."""
        batch_sizes = [self.config.max_num_seqs]
        if self.config.enable_continuous_batching and self.config.max_num_seqs > 1:
            batch_sizes.append(max(1, self.config.max_num_seqs // 2))
        batch_sizes = sorted(set(batch_sizes))
        prompts = [[0] for _ in range(max(batch_sizes))]
        self._allocate_cache(prompts, enable_prefix_caching=False)
        try:
            states = [
                PearlPipelineState(
                    [0],
                    prompt_length=1,
                    temperature=0.0,
                    max_tokens=self.config.max_tokens,
                    ignore_eos=True,
                )
                for _ in prompts
            ]
            # Capturing against an uninitialized KV cache is not replay-stable
            # on CANN, so initialize a real one-token prefix on every worker.
            self._prefill_and_sample_target_batch(prompts, states)
            for state in states:
                state.token_ids.append(0)
                assert state.committed_length is not None
                state.committed_length += 1

            if self.is_draft:
                for batch_size in batch_sizes:
                    self.graph_runner.set_expected_fia_batch_size(batch_size)
                    batch_states = states[:batch_size]
                    active_indices = list(range(batch_size))
                    # Capture plus one init-time replay keeps graph setup out of
                    # the first measured request.
                    self._draft_round_device_batch(batch_states, active_indices)
                    self._draft_round_device_batch(batch_states, active_indices)
            else:
                # Normal paged replay may pad into an existing larger graph.
                # Ascending compilation guarantees an exact entry per bucket.
                for _, batch_size, post_verify_count in _target_graph_precompile_shapes(
                    batch_sizes,
                    self.gamma,
                    self.config.target_verification_graph_buckets,
                    self.config.target_verification_graph_post_counts,
                ):
                    pre_verify = [False] * post_verify_count + [True] * (
                        batch_size - post_verify_count
                    )
                    model_widths, _ = _bucket_target_verification_widths(
                        pre_verify,
                        self.gamma,
                        self.config.target_verification_graph_buckets,
                        self.config.target_verification_graph_post_counts,
                    )
                    input_token_ids: list[int] = []
                    sequence_ids: list[int] = []
                    positions: list[int] = []
                    for sequence_id, model_width in enumerate(model_widths):
                        input_token_ids.extend([0] * model_width)
                        sequence_ids.extend([sequence_id] * model_width)
                        positions.extend(range(1, model_width + 1))
                    input_ids = torch.tensor(
                        input_token_ids,
                        dtype=torch.long,
                        device=self.device,
                    )
                    self.graph_runner.set_expected_fia_batch_size(batch_size)
                    self._run_device_packed_greedy(
                        input_ids,
                        sequence_ids,
                        positions,
                        use_fused_infer_attention=False,
                    )
                    self._run_device_packed_greedy(
                        input_ids,
                        sequence_ids,
                        positions,
                        use_fused_infer_attention=False,
                    )
            torch.npu.synchronize()
            self.precompiled_decode_batch_sizes = frozenset(batch_sizes)
        finally:
            self._release_cache()

    def _draft_round_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
    ) -> tuple[list[list[int]], list[list[int]]]:
        if not self.is_draft:
            return [], []
        was_pre_verify = [states[index].pre_verify for index in active_indices]
        input_ids = torch.tensor(
            [states[index].token_ids[-1] for index in active_indices],
            dtype=torch.long,
            device=self.device,
        )
        first_positions = [len(states[index].token_ids) - 1 for index in active_indices]
        draft_use_fia = not getattr(self.config, "draft_use_paged_attention", False)
        if (
            not self.config.enforce_eager
            and self.gamma <= 16
            and hasattr(self, "graph_runner")
            and hasattr(self.graph_runner, "run_draft_greedy")
        ):
            position_tensors = []
            attention_metadatas = []
            for step in range(self.gamma):
                step_positions = [position + step for position in first_positions]
                position_tensor, attention_metadata = self._prepare_attention_metadata(
                    active_indices,
                    step_positions,
                    use_fused_infer_attention=draft_use_fia,
                )
                position_tensors.append(position_tensor)
                attention_metadatas.append(attention_metadata)
            draft_tensor = self.graph_runner.run_draft_greedy(
                input_ids,
                position_tensors,
                attention_metadatas,
                self.draft_vocab_size,
            )
            draft_windows = draft_tensor.cpu().tolist()
        else:
            draft_steps: list[torch.Tensor] = []
            for step in range(self.gamma):
                positions = [position + step for position in first_positions]
                input_ids = self._run_device_packed_greedy(
                    input_ids,
                    active_indices,
                    positions,
                    use_aclgraph=not self.config.enforce_eager and self.gamma <= 16,
                    use_fused_infer_attention=draft_use_fia and self.gamma <= 16,
                )
                # ACLGraph replays reuse one persistent output buffer. Preserve
                # each proposal before the next replay overwrites that buffer.
                draft_steps.append(input_ids.clone())
            draft_windows = torch.stack(draft_steps, dim=1).cpu().tolist()
        for sequence_index, token_ids in zip(active_indices, draft_windows):
            states[sequence_index].token_ids.extend(int(token_id) for token_id in token_ids)
        next_windows = [states[index].token_ids[-self.gamma :] for index in active_indices]
        verification_windows = []
        for sequence_index, is_pre_verify, next_window in zip(active_indices, was_pre_verify, next_windows):
            state = states[sequence_index]
            verification_windows.append(
                [next_window[0]] if is_pre_verify else state.token_ids[-2 * self.gamma + 1 : -self.gamma + 1]
            )
        return verification_windows, next_windows

    def _draft_round_device_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
        draft_budgets: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Produce packed proposals without an intermediate NPU-to-CPU copy."""
        if not self.is_draft:
            return None, None
        budgets = (
            [self.gamma] * len(active_indices)
            if draft_budgets is None
            else [int(value) for value in draft_budgets]
        )
        if len(budgets) != len(active_indices) or any(
            value <= 0 or value > self.gamma for value in budgets
        ):
            raise ValueError("PEARL draft budgets must contain one value in [1, gamma] per request.")
        was_pre_verify = [states[index].pre_verify for index in active_indices]
        input_ids = torch.tensor(
            [states[index].token_ids[-1] for index in active_indices],
            dtype=torch.long,
            device=self.device,
        )
        first_positions = [len(states[index].token_ids) - 1 for index in active_indices]
        draft_temperatures = [states[index].draft_temperature for index in active_indices]
        draft_use_fia = not getattr(self.config, "draft_use_paged_attention", False)
        if (
            not self.config.enforce_eager
            and self.gamma <= 16
            and all(value == 0 for value in draft_temperatures)
            and all(value == self.gamma for value in budgets)
            and hasattr(self, "graph_runner")
            and hasattr(self.graph_runner, "run_draft_greedy")
        ):
            position_tensors = []
            attention_metadatas = []
            for step in range(self.gamma):
                step_positions = [position + step for position in first_positions]
                position_tensor, attention_metadata = self._prepare_attention_metadata(
                    active_indices,
                    step_positions,
                    use_fused_infer_attention=draft_use_fia,
                )
                position_tensors.append(position_tensor)
                attention_metadatas.append(attention_metadata)
            next_windows = self.graph_runner.run_draft_greedy(
                input_ids,
                position_tensors,
                attention_metadatas,
                self.draft_vocab_size,
            )
        else:
            next_windows = torch.full(
                (len(active_indices), self.gamma),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            for step in range(max(budgets)):
                active_rows = [
                    row for row, budget in enumerate(budgets) if step < budget
                ]
                row_tensor = torch.tensor(
                    active_rows, dtype=torch.long, device=self.device
                )
                step_sequence_ids = [active_indices[row] for row in active_rows]
                positions = [first_positions[row] + step for row in active_rows]
                step_input = input_ids.index_select(0, row_tensor)
                step_temperatures = [draft_temperatures[row] for row in active_rows]
                if all(value == 0 for value in step_temperatures):
                    step_output = self._run_device_packed_greedy(
                        step_input,
                        step_sequence_ids,
                        positions,
                        use_aclgraph=not self.config.enforce_eager and self.gamma <= 16,
                        use_fused_infer_attention=draft_use_fia and self.gamma <= 16,
                    )
                else:
                    step_output, _ = self._run_device_packed_sample_with_confidence(
                        step_input,
                        step_sequence_ids,
                        positions,
                        step_temperatures,
                        use_aclgraph=False,
                        use_fused_infer_attention=draft_use_fia and self.gamma <= 16,
                    )
                # Graph replays may return a persistent output buffer, so copy
                # both the proposal column and the next-step input explicitly.
                step_output = step_output.clone()
                next_windows[row_tensor, step] = step_output
                input_ids = input_ids.clone()
                input_ids.index_copy_(0, row_tensor, step_output)

        verification_sizes = [
            1
            if pre_verify
            else (states[index].pending_window_size or self.gamma)
            for index, pre_verify in zip(active_indices, was_pre_verify)
        ]
        verification_size = sum(verification_sizes)
        current_indices: list[int] = []
        previous_indices: list[int] = []
        previous_tokens: list[int] = []
        offset = 0
        for sequence_index, pre_verify, size in zip(
            active_indices,
            was_pre_verify,
            verification_sizes,
        ):
            if not pre_verify and size > 1:
                previous_indices.extend(range(offset, offset + size - 1))
                previous_tokens.extend(states[sequence_index].token_ids[-(size - 1) :])
            current_indices.append(offset + size - 1)
            offset += size
        verification = torch.empty(
            verification_size,
            dtype=torch.long,
            device=self.device,
        )
        if previous_indices:
            verification[torch.tensor(previous_indices, dtype=torch.long, device=self.device)] = torch.tensor(
                previous_tokens, dtype=torch.long, device=self.device
            )
        verification[torch.tensor(current_indices, dtype=torch.long, device=self.device)] = next_windows[:, 0]
        return verification, next_windows

    def _draft_spec_rhythm_device_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
        draft_budgets: Sequence[int],
        *,
        verification_sizes: Sequence[int] | None = None,
        verification_prefixes: Sequence[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Draft variable windows and report mean softmax confidence per row."""

        if not self.is_draft:
            return None, None, None
        budgets = [int(value) for value in draft_budgets]
        if len(budgets) != len(active_indices) or any(
            value <= 0 or value > self.gamma for value in budgets
        ):
            raise ValueError("SpecRhythm draft budgets must be in [1, gamma].")
        input_ids = torch.tensor(
            [states[index].token_ids[-1] for index in active_indices],
            dtype=torch.long,
            device=self.device,
        )
        first_positions = [len(states[index].token_ids) - 1 for index in active_indices]
        draft_temperatures = [states[index].draft_temperature for index in active_indices]
        next_windows = torch.full(
            (len(active_indices), self.gamma),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        confidence_sums = torch.zeros(
            len(active_indices), dtype=torch.float32, device=self.device
        )
        draft_use_fia = not getattr(self.config, "draft_use_paged_attention", False)
        for step in range(max(budgets)):
            active_rows = [row for row, budget in enumerate(budgets) if step < budget]
            row_tensor = torch.tensor(active_rows, dtype=torch.long, device=self.device)
            sequence_ids = [active_indices[row] for row in active_rows]
            positions = [first_positions[row] + step for row in active_rows]
            step_input = input_ids.index_select(0, row_tensor)
            step_temperatures = [draft_temperatures[row] for row in active_rows]
            if all(value == 0 for value in step_temperatures):
                step_tokens, step_confidence = self._run_device_packed_greedy_with_confidence(
                    step_input,
                    sequence_ids,
                    positions,
                    use_aclgraph=not self.config.enforce_eager and self.gamma <= 16,
                    use_fused_infer_attention=draft_use_fia and self.gamma <= 16,
                )
            else:
                step_tokens, step_confidence = self._run_device_packed_sample_with_confidence(
                    step_input,
                    sequence_ids,
                    positions,
                    step_temperatures,
                    use_aclgraph=False,
                    use_fused_infer_attention=draft_use_fia and self.gamma <= 16,
                )
            step_tokens = step_tokens.clone()
            next_windows[row_tensor, step] = step_tokens
            confidence_sums.index_add_(0, row_tensor, step_confidence.float())
            input_ids = input_ids.clone()
            input_ids.index_copy_(0, row_tensor, step_tokens)

        confidence = confidence_sums / torch.tensor(
            budgets, dtype=torch.float32, device=self.device
        )
        actual_verification_sizes = (
            [
                1
                if states[index].pre_verify
                else (states[index].pending_window_size or self.gamma)
                for index in active_indices
            ]
            if verification_sizes is None
            else [int(value) for value in verification_sizes]
        )
        prefixes = (
            [None] * len(active_indices)
            if verification_prefixes is None
            else list(verification_prefixes)
        )
        if len(actual_verification_sizes) != len(active_indices) or len(prefixes) != len(
            active_indices
        ):
            raise ValueError(
                "SpecRhythm verification metadata must contain one row per request."
            )
        verification_parts: list[torch.Tensor] = []
        for row, (sequence_index, size, prefix) in enumerate(
            zip(active_indices, actual_verification_sizes, prefixes)
        ):
            if not 0 < size <= self.gamma:
                raise ValueError("SpecRhythm verification sizes must be in [1, gamma].")
            if prefix is None:
                prefix = torch.tensor(
                    states[sequence_index].token_ids[-(size - 1) :]
                    if size > 1
                    else [],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                prefix = prefix.reshape(-1).to(device=self.device, dtype=torch.long)
            if prefix.numel() != size - 1:
                raise ValueError(
                    "SpecRhythm eager verification prefix does not match its parent window."
                )
            verification_parts.append(torch.cat((prefix, next_windows[row, :1])))
        verification = torch.cat(verification_parts)
        return verification, next_windows, confidence

    def _exchange_spec_rhythm_device_proposals(
        self,
        verification_window: torch.Tensor | None,
        next_windows: torch.Tensor | None,
        confidences: torch.Tensor | None,
        tickets: Sequence[SpecRhythmProposalTicket],
        verification_sizes: Sequence[int],
    ) -> tuple[torch.Tensor | None, list[float]]:
        """Broadcast a self-describing variable-offset proposal envelope."""

        count = len(tickets)
        if count == 0:
            return None, []
        if len(verification_sizes) != count:
            raise RuntimeError("SpecRhythm proposal metadata has inconsistent row counts.")
        metadata_width = 7
        verification_size = sum(verification_sizes)
        message_size = metadata_width * count + verification_size + count * self.gamma
        if self.is_draft:
            if confidences is None or confidences.shape != (count,):
                raise RuntimeError("SpecRhythm draft confidence tensor has an invalid shape.")
            quantized_confidences = [
                int(round(float(value) * 1_000_000))
                for value in confidences.detach().cpu().tolist()
            ]
        else:
            quantized_confidences = [0] * count

        if self.groups.is_verification_worker:
            if self.rank == self.topology.draft_leader_rank:
                if verification_window is None or verification_window.shape != (verification_size,):
                    raise RuntimeError("SpecRhythm verification payload has an invalid shape.")
                if next_windows is None or next_windows.shape != (count, self.gamma):
                    raise RuntimeError("SpecRhythm continuation payload has an invalid shape.")
                metadata = torch.tensor(
                    [
                        value
                        for ticket, size, confidence in zip(
                            tickets, verification_sizes, quantized_confidences
                        )
                        for value in (
                            ticket.proposal_id,
                            ticket.request_index,
                            ticket.home_batch_id,
                            size,
                            ticket.gamma,
                            ticket.required_prefix_epoch,
                            confidence,
                        )
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
                message = torch.cat((metadata, verification_window, next_windows.flatten()))
            else:
                message = torch.empty(message_size, dtype=torch.long, device=self.device)
            dist.broadcast(
                message,
                src=self.topology.draft_leader_rank,
                group=self.groups.verification_group,
            )
            metadata_values = message[: metadata_width * count].reshape(
                count, metadata_width
            ).cpu().tolist()
            expected = [
                [
                    ticket.proposal_id,
                    ticket.request_index,
                    ticket.home_batch_id,
                    int(size),
                    ticket.gamma,
                    ticket.required_prefix_epoch,
                ]
                for ticket, size in zip(tickets, verification_sizes)
            ]
            if [row[:6] for row in metadata_values] != expected:
                raise RuntimeError("SpecRhythm received a misrouted or stale HCCL mailbox envelope.")
            quantized_confidences = [int(row[6]) for row in metadata_values]
        else:
            message = None
        return message, [value / 1_000_000.0 for value in quantized_confidences]

    def _materialize_spec_rhythm_payloads(
        self,
        *,
        tickets: Sequence[SpecRhythmProposalTicket],
        verification_sizes: Sequence[int],
        local_verification: torch.Tensor | None,
        local_next_windows: torch.Tensor | None,
        exchanged_message: torch.Tensor | None,
        draft_confidences: Sequence[float],
    ) -> list[NativeSpecRhythmDevicePayload]:
        count = len(tickets)
        metadata_size = count * 7
        verification_size = sum(verification_sizes)
        if self.is_draft:
            if local_verification is None or local_next_windows is None:
                raise RuntimeError("SpecRhythm draft worker lost a local proposal payload.")
            verification = local_verification
            continuations = local_next_windows
        else:
            if exchanged_message is None:
                raise RuntimeError("SpecRhythm target worker did not receive a proposal payload.")
            verification = exchanged_message[
                metadata_size : metadata_size + verification_size
            ]
            continuations = exchanged_message[
                metadata_size + verification_size :
            ].reshape(count, self.gamma)
        payloads: list[NativeSpecRhythmDevicePayload] = []
        offset = 0
        for row, (ticket, size, confidence) in enumerate(
            zip(tickets, verification_sizes, draft_confidences)
        ):
            payloads.append(
                NativeSpecRhythmDevicePayload(
                    ticket=ticket,
                    verification_tokens=verification[offset : offset + size],
                    next_tokens=continuations[row, : ticket.gamma],
                    verification_size=int(size),
                    draft_confidence=float(confidence),
                )
            )
            offset += size
        return payloads

    def _target_round_outputs_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
        verification_sizes: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.is_draft:
            return None, None
        input_token_ids: list[int] = []
        sequence_ids: list[int] = []
        positions: list[int] = []
        actual_verification_sizes = (
            [
                1
                if states[index].pre_verify
                else (states[index].pending_window_size or self.gamma)
                for index in active_indices
            ]
            if verification_sizes is None
            else [int(value) for value in verification_sizes]
        )
        if len(actual_verification_sizes) != len(active_indices) or any(
            value <= 0 or value > self.gamma for value in actual_verification_sizes
        ):
            raise ValueError("PEARL target verification sizes must be in [1, gamma].")
        model_widths, valid_token_indices = _bucket_variable_target_verification_widths(
            actual_verification_sizes,
            self.gamma,
            self.config.target_verification_graph_buckets,
            getattr(self.config, "target_verification_graph_post_counts", ()),
        )
        for sequence_index, valid_width, model_width in zip(
            active_indices, actual_verification_sizes, model_widths
        ):
            state = states[sequence_index]
            start_position = len(state.token_ids) - valid_width
            tokens = state.token_ids[-valid_width:]
            if model_width > valid_width:
                tokens = [*tokens, *([tokens[-1]] * (model_width - valid_width))]
            input_token_ids.extend(tokens)
            sequence_ids.extend([sequence_index] * model_width)
            positions.extend(range(start_position, start_position + model_width))
        temperatures = [
            states[index].temperature
            for index, model_width in zip(active_indices, model_widths)
            for _ in range(model_width)
        ]
        # Speculative execution uses FIA graphs; the decode-only paged-attention
        # graph can be selected for target verification on model/topology pairs
        # whose dynamic FIA graph-task updates are not replay-stable.
        use_aclgraph = not self.config.enforce_eager and self.gamma <= 16
        target_use_fia = not self.config.target_use_paged_attention
        if all(temperature == 0 for temperature in temperatures):
            target_tokens = self._run_packed_greedy(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph=use_aclgraph,
                use_fused_infer_attention=target_use_fia and self.gamma <= 16,
            )
            if len(valid_token_indices) != len(input_token_ids):
                target_tokens = target_tokens.index_select(
                    0,
                    torch.tensor(valid_token_indices, dtype=torch.long, device=self.device),
                )
            return target_tokens, None
        logits = self._run_packed_model(
            input_token_ids,
            sequence_ids,
            positions,
            use_aclgraph=use_aclgraph,
            use_fused_infer_attention=target_use_fia and self.gamma <= 16,
        )
        if len(valid_token_indices) != len(input_token_ids):
            logits = logits.index_select(
                0,
                torch.tensor(valid_token_indices, dtype=torch.long, device=self.device),
            )
        return None, logits[:, : self.draft_vocab_size]

    def _exchange_draft_windows(
        self,
        verification_windows: list[list[int]],
        next_windows: list[list[int]],
        verification_sizes: list[int],
    ) -> torch.Tensor | None:
        verification_size = sum(verification_sizes)
        continuation_size = self.gamma * len(verification_sizes)
        if self.groups.is_verification_worker:
            if self.rank == self.topology.draft_leader_rank:
                if [len(window) for window in verification_windows] != verification_sizes:
                    raise RuntimeError("PEARL draft window has an unexpected shape.")
                if any(len(window) != self.gamma for window in next_windows):
                    raise RuntimeError("PEARL continuation window has an unexpected shape.")
                message = torch.tensor(
                    [token for window in verification_windows for token in window]
                    + [token for window in next_windows for token in window],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                message = torch.empty(verification_size + continuation_size, dtype=torch.long, device=self.device)
            dist.broadcast(message, src=self.topology.draft_leader_rank, group=self.groups.verification_group)
            return message
        return None

    def _exchange_draft_device_windows(
        self,
        verification_window: torch.Tensor | None,
        next_windows: torch.Tensor | None,
        verification_sizes: list[int],
        next_window_sizes: Sequence[int] | None = None,
    ) -> torch.Tensor | None:
        verification_size = sum(verification_sizes)
        continuation_size = self.gamma * len(verification_sizes)
        if not self.groups.is_verification_worker:
            return None
        if self.rank == self.topology.draft_leader_rank:
            if verification_window is None or verification_window.shape != (verification_size,):
                raise RuntimeError("PEARL draft verification window has an unexpected shape.")
            if next_windows is None or next_windows.shape != (len(verification_sizes), self.gamma):
                raise RuntimeError("PEARL draft continuation window has an unexpected shape.")
            if next_window_sizes is not None and (
                len(next_window_sizes) != len(verification_sizes)
                or any(value <= 0 or value > self.gamma for value in next_window_sizes)
            ):
                raise RuntimeError("PEARL draft continuation lengths are invalid.")
            message = torch.cat((verification_window, next_windows.flatten()))
        else:
            message = torch.empty(
                verification_size + continuation_size,
                dtype=torch.long,
                device=self.device,
            )
        dist.broadcast(
            message,
            src=self.topology.draft_leader_rank,
            group=self.groups.verification_group,
        )
        return message

    def _verify_target_tokens_batch(
        self,
        target_tokens: torch.Tensor | None,
        target_logits: torch.Tensor | None,
        draft_message: torch.Tensor | None,
        verification_sizes: list[int],
        temperatures: list[float],
    ) -> torch.Tensor | None:
        verification_size = sum(verification_sizes)
        packed_temperatures = [
            temperature for temperature, size in zip(temperatures, verification_sizes) for _ in range(size)
        ]
        greedy = all(temperature == 0 for temperature in packed_temperatures)
        if self.is_draft or (not greedy and self.rank != self.topology.target_leader_rank):
            return None
        if draft_message is None:
            raise RuntimeError("A PEARL target rank did not receive draft verification tokens.")
        if greedy:
            if target_tokens is None or target_tokens.shape != (verification_size,):
                raise RuntimeError("PEARL target and draft verification windows differ in length.")
            layout_key = (self.gamma, tuple(verification_sizes))
            layout = self.greedy_verification_layouts.get(layout_key)
            if layout is None:
                layout = _build_verification_layout(
                    verification_sizes,
                    self.gamma,
                    target_tokens.device,
                )
                self.greedy_verification_layouts[layout_key] = layout
            return _build_greedy_verdict_with_layout(
                target_tokens,
                draft_message[:verification_size],
                *layout,
                self.gamma,
            )
        if target_logits is None or target_logits.shape[0] != verification_size:
            raise RuntimeError("PEARL target logits do not match the packed verification window.")
        return _build_stochastic_verdict(
            target_logits,
            draft_message[:verification_size],
            verification_sizes,
            self.gamma,
            packed_temperatures,
        )

    def _broadcast_round_result(
        self,
        verdict: torch.Tensor | None,
        draft_message: torch.Tensor | None,
        verification_size: int,
        batch_size: int,
        *,
        replicated_target_verdict: bool,
    ) -> tuple[list[int], list[int | None], list[list[int]]]:
        continuation_size = batch_size * self.gamma
        is_target_rank = self.rank in self.topology.target_ranks
        if replicated_target_verdict and is_target_rank:
            if verdict is None or draft_message is None:
                raise RuntimeError("A PEARL target rank did not produce a replicated round result.")
            result = torch.cat((verdict.flatten(), draft_message[verification_size:]))
        else:
            result = torch.empty(batch_size * 2 + continuation_size, dtype=torch.long, device=self.device)
        if replicated_target_verdict:
            if self.rank in self.topology.correction_ranks:
                dist.broadcast(
                    result,
                    src=self.topology.target_leader_rank,
                    group=self.groups.correction_group,
                )
        else:
            if self.rank == self.topology.target_leader_rank:
                if verdict is None or draft_message is None:
                    raise RuntimeError("The PEARL target leader did not produce a round result.")
                result = torch.cat((verdict.flatten(), draft_message[verification_size:]))
            dist.broadcast(result, src=self.topology.target_leader_rank)
        values = [int(value) for value in result.cpu().tolist()]
        verdict_values = values[: batch_size * 2]
        continuation_values = values[batch_size * 2 :]
        accepted = verdict_values[::2]
        corrections = [None if value == -1 else value for value in verdict_values[1::2]]
        next_windows = [
            continuation_values[index : index + self.gamma] for index in range(0, continuation_size, self.gamma)
        ]
        return accepted, corrections, next_windows

    def _broadcast_device_round_result(
        self,
        verdict: torch.Tensor | None,
        draft_message: torch.Tensor | None,
        verification_size: int,
        batch_size: int,
        local_next_windows: torch.Tensor | None,
        *,
        replicated_target_verdict: bool,
        next_window_sizes: Sequence[int] | None = None,
        profile_phase_seconds: dict[str, float] | None = None,
    ) -> tuple[list[int], list[int | None], list[list[int]]]:
        """Synchronize only the verdict; proposal continuations stay local."""
        is_target_rank = self.rank in self.topology.target_ranks
        if is_target_rank:
            if verdict is None:
                raise RuntimeError("A PEARL target rank did not produce a round verdict.")
            verdict_result = verdict.flatten()
        else:
            verdict_result = torch.empty(batch_size * 2, dtype=torch.long, device=self.device)

        communication_started = time.perf_counter()
        participated_in_broadcast = False
        if replicated_target_verdict:
            if self.rank in self.topology.correction_ranks:
                participated_in_broadcast = True
                dist.broadcast(
                    verdict_result,
                    src=self.topology.target_leader_rank,
                    group=self.groups.correction_group,
                )
        else:
            participated_in_broadcast = True
            dist.broadcast(verdict_result, src=self.topology.target_leader_rank)
        if profile_phase_seconds is not None and participated_in_broadcast:
            torch.npu.synchronize()
            profile_phase_seconds["target_to_draft_communication"] += (
                time.perf_counter() - communication_started
            )

        if self.is_draft:
            if local_next_windows is None or local_next_windows.shape != (batch_size, self.gamma):
                raise RuntimeError("A PEARL draft rank did not retain its local continuation window.")
            continuation = local_next_windows.flatten()
        else:
            if draft_message is None:
                raise RuntimeError("A PEARL target rank did not receive the draft continuation window.")
            continuation = draft_message[verification_size:]
        materialize_started = time.perf_counter()
        values = torch.cat((verdict_result, continuation)).cpu().tolist()
        if profile_phase_seconds is not None:
            profile_phase_seconds["wait_sync"] += time.perf_counter() - materialize_started
        verdict_values = values[: batch_size * 2]
        continuation_values = values[batch_size * 2 :]
        accepted = [int(value) for value in verdict_values[::2]]
        corrections = [None if value == -1 else int(value) for value in verdict_values[1::2]]
        window_sizes = (
            [self.gamma] * batch_size
            if next_window_sizes is None
            else [int(value) for value in next_window_sizes]
        )
        if len(window_sizes) != batch_size or any(
            value <= 0 or value > self.gamma for value in window_sizes
        ):
            raise RuntimeError("PEARL synchronized continuation lengths are invalid.")
        next_windows = [
            [
                int(value)
                for value in continuation_values[
                    row * self.gamma : row * self.gamma + window_sizes[row]
                ]
            ]
            for row in range(batch_size)
        ]
        return accepted, corrections, next_windows


def _normalize_eos_tokens(eos_token_id: int | list[int] | None) -> frozenset[int]:
    if eos_token_id is None:
        return frozenset()
    if isinstance(eos_token_id, int):
        return frozenset((eos_token_id,))
    return frozenset(int(token_id) for token_id in eos_token_id)


def _canonical_active_indices(
    states: Sequence[PearlPipelineState],
    active_indices: Sequence[int],
) -> list[int]:
    """Group post-verify and pre-verify requests into stable FIA shapes."""
    return sorted(active_indices, key=lambda index: states[index].pre_verify)


def _bucket_target_verification_widths(
    pre_verify: Sequence[bool],
    gamma: int,
    num_buckets: int = TARGET_VERIFICATION_GRAPH_BUCKETS,
    configured_post_counts: Sequence[tuple[int, Sequence[int]]] = (),
) -> tuple[list[int], list[int]]:
    """Quantize target FIA shapes while retaining only real verification rows."""
    if gamma <= 0 or num_buckets <= 0:
        raise ValueError("PEARL verification gamma and graph bucket count must be positive.")
    if not pre_verify:
        return [], []
    post_verify_count = sum(not value for value in pre_verify)
    post_verify_counts = _target_graph_post_counts_for_batch(
        len(pre_verify),
        num_buckets,
        configured_post_counts,
    )
    padded_post_verify_count = next(
        value for value in post_verify_counts if value >= post_verify_count
    )
    padded_pre_verify_count = padded_post_verify_count - post_verify_count
    model_widths: list[int] = []
    valid_token_indices: list[int] = []
    offset = 0
    for is_pre_verify in pre_verify:
        if not is_pre_verify:
            model_width = gamma
            valid_token_indices.extend(range(offset, offset + gamma))
        elif padded_pre_verify_count:
            model_width = gamma
            valid_token_indices.append(offset)
            padded_pre_verify_count -= 1
        else:
            model_width = 1
            valid_token_indices.append(offset)
        model_widths.append(model_width)
        offset += model_width
    return model_widths, valid_token_indices


def _bucket_variable_target_verification_widths(
    verification_sizes: Sequence[int],
    gamma: int,
    num_buckets: int = TARGET_VERIFICATION_GRAPH_BUCKETS,
    configured_post_counts: Sequence[tuple[int, Sequence[int]]] = (),
) -> tuple[list[int], list[int]]:
    """Quantize mixed per-request widths without exposing padded rows."""

    if any(size <= 0 or size > gamma for size in verification_sizes):
        raise ValueError("PEARL verification sizes must be in [1, gamma].")
    post_verify = [size > 1 for size in verification_sizes]
    model_widths, _ = _bucket_target_verification_widths(
        [not value for value in post_verify],
        gamma,
        num_buckets,
        configured_post_counts,
    )
    valid_token_indices: list[int] = []
    offset = 0
    for size, model_width in zip(verification_sizes, model_widths):
        valid_token_indices.extend(range(offset, offset + size))
        offset += model_width
    return model_widths, valid_token_indices


def _target_graph_precompile_shapes(
    batch_sizes: Sequence[int],
    gamma: int,
    num_buckets: int,
    configured_post_counts: Sequence[tuple[int, Sequence[int]]] = (),
) -> list[tuple[int, int, int]]:
    """Return unique target graph sizes with representative request layouts."""
    shape_cases: dict[int, tuple[int, int]] = {}
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError("PEARL graph precompile batch sizes must be positive.")
        post_verify_counts = _target_graph_post_counts_for_batch(
            batch_size,
            num_buckets,
            configured_post_counts,
        )
        for post_verify_count in post_verify_counts:
            pre_verify = [False] * post_verify_count + [True] * (
                batch_size - post_verify_count
            )
            model_widths, _ = _bucket_target_verification_widths(
                pre_verify,
                gamma,
                num_buckets,
                configured_post_counts,
            )
            shape_cases.setdefault(
                sum(model_widths),
                (batch_size, post_verify_count),
            )
    return [
        (num_tokens, batch_size, post_verify_count)
        for num_tokens, (batch_size, post_verify_count) in sorted(shape_cases.items())
    ]


def _normalize_target_graph_post_counts(
    configured_post_counts: Sequence[tuple[int, Sequence[int]]],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Validate and freeze workload-specific target verification graph buckets."""
    normalized: list[tuple[int, tuple[int, ...]]] = []
    configured_batches: set[int] = set()
    for batch_size, raw_post_counts in configured_post_counts:
        if batch_size <= 0:
            raise ValueError("PEARL target graph post-count batch sizes must be positive.")
        if batch_size in configured_batches:
            raise ValueError("PEARL target graph post-count batch sizes must be unique.")
        post_counts = tuple(raw_post_counts)
        if (
            not post_counts
            or post_counts[0] != 0
            or post_counts[-1] != batch_size
            or any(left >= right for left, right in zip(post_counts, post_counts[1:]))
        ):
            raise ValueError(
                "PEARL target graph post counts must be strictly increasing "
                "from zero through their batch size."
            )
        configured_batches.add(batch_size)
        normalized.append((batch_size, post_counts))
    return tuple(sorted(normalized))


def _target_graph_post_counts_for_batch(
    batch_size: int,
    num_buckets: int,
    configured_post_counts: Sequence[tuple[int, Sequence[int]]] = (),
) -> tuple[int, ...]:
    for configured_batch_size, post_counts in configured_post_counts:
        if configured_batch_size == batch_size:
            return tuple(post_counts)
    bucket_width = max(1, (batch_size + num_buckets - 1) // num_buckets)
    values = set(range(0, batch_size + 1, bucket_width))
    values.add(batch_size)
    return tuple(sorted(values))


def _normalize_sampling_params(
    batch_size: int,
    sampling_params: NativeSamplingParams | Sequence[NativeSamplingParams] | None,
    default_max_tokens: int,
) -> list[NativeSamplingParams]:
    if sampling_params is None:
        params = [NativeSamplingParams(temperature=0.0, max_tokens=default_max_tokens)] * batch_size
    elif isinstance(sampling_params, NativeSamplingParams):
        params = [sampling_params] * batch_size
    else:
        params = list(sampling_params)
        if len(params) != batch_size:
            raise ValueError("PEARL requires one SamplingParams value per prompt.")
        if not all(isinstance(value, NativeSamplingParams) for value in params):
            raise TypeError("PEARL sampling_params must contain SamplingParams values.")
    temperatures = [value.temperature for value in params]
    if not (all(value == 0 for value in temperatures) or all(value > 0 for value in temperatures)):
        raise ValueError("A PEARL batch requires temperatures that are either all zero or all non-zero.")
    return params


def _sample_logits(
    logits: torch.Tensor,
    temperatures: Sequence[float],
    *,
    exponential_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply nano-PEARL's greedy or exponential-race target sampler."""
    if logits.ndim != 2 or logits.shape[0] != len(temperatures):
        raise ValueError("PEARL logits and temperatures must have the same batch dimension.")
    if all(temperature == 0 for temperature in temperatures):
        return logits.argmax(dim=-1)
    if not all(temperature > 0 for temperature in temperatures):
        raise ValueError("A PEARL sample requires temperatures that are either all zero or all non-zero.")
    temperature_tensor = torch.tensor(temperatures, dtype=torch.float32, device=logits.device).unsqueeze(1)
    probabilities = torch.softmax(logits.float() / temperature_tensor, dim=-1)
    if exponential_noise is None:
        exponential_noise = torch.empty_like(probabilities).exponential_(1)
    elif exponential_noise.shape != probabilities.shape:
        raise ValueError("PEARL exponential sampling noise must match the logits shape.")
    return probabilities.div(exponential_noise.clamp_min(1e-10)).argmax(dim=-1)


def _build_verification_layout(
    verification_sizes: Sequence[int],
    gamma: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_sizes = torch.tensor(verification_sizes, dtype=torch.long, device=device)
    dense_positions = torch.tensor(
        [row * gamma + column for row, size in enumerate(verification_sizes) for column in range(size)],
        dtype=torch.long,
        device=device,
    )
    packed_positions: list[list[int]] = []
    offset = 0
    for size in verification_sizes:
        row = list(range(offset, offset + size))
        packed_positions.append(row + [row[-1]] * (gamma - size))
        offset += size
    correction_positions = torch.tensor(packed_positions, dtype=torch.long, device=device)
    return expected_sizes, dense_positions, correction_positions


def _build_greedy_verdict_with_layout(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    expected_sizes: torch.Tensor,
    dense_positions: torch.Tensor,
    correction_positions: torch.Tensor,
    gamma: int,
) -> torch.Tensor:
    """Build a greedy verdict without per-request device slice writes."""
    matches = target_tokens == draft_tokens
    dense_size = expected_sizes.shape[0] * gamma
    match_matrix = torch.ones(dense_size, dtype=torch.bool, device=matches.device).scatter(
        0,
        dense_positions,
        matches,
    )
    match_matrix = match_matrix.reshape(-1, gamma)
    all_match = match_matrix.all(dim=1)
    first_mismatch = (~match_matrix).to(dtype=torch.int32).argmax(dim=1)
    accepted = torch.where(all_match, expected_sizes, first_mismatch)
    correction_indices = correction_positions.gather(
        1,
        accepted.clamp(max=gamma - 1).unsqueeze(1),
    ).squeeze(1)
    correction = target_tokens.gather(0, correction_indices)
    correction = torch.where(all_match, torch.full_like(correction, -1), correction)
    return torch.stack((accepted, correction), dim=1)


def _build_greedy_verdict(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    verification_sizes: list[int],
    gamma: int,
) -> torch.Tensor:
    """Find each sequence's accepted greedy prefix without a host sync."""
    if gamma <= 0 or any(size <= 0 or size > gamma for size in verification_sizes):
        raise ValueError("PEARL verification windows must have length in [1, gamma].")
    verification_size = sum(verification_sizes)
    if target_tokens.shape != (verification_size,) or draft_tokens.shape != (verification_size,):
        raise ValueError("PEARL target and draft token tensors must match the packed verification size.")

    matches = target_tokens == draft_tokens
    return _build_verdict_from_acceptance(matches, target_tokens, verification_sizes, gamma)


def _build_stochastic_verdict(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    verification_sizes: list[int],
    gamma: int,
    temperatures: Sequence[float],
    *,
    random_values: torch.Tensor | None = None,
    exponential_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify greedy draft tokens against a sampled target distribution."""
    verification_size = sum(verification_sizes)
    if target_logits.ndim != 2 or target_logits.shape[0] != verification_size:
        raise ValueError("PEARL target logits must match the packed verification size.")
    if draft_tokens.shape != (verification_size,) or len(temperatures) != verification_size:
        raise ValueError("PEARL draft tokens and temperatures must match the packed verification size.")
    if not all(temperature > 0 for temperature in temperatures):
        raise ValueError("Stochastic PEARL verification requires positive target temperatures.")
    if draft_tokens.device.type == "cpu" and draft_tokens.numel() and int(draft_tokens.max()) >= target_logits.shape[1]:
        raise ValueError("A PEARL draft token lies outside the target verification vocabulary.")

    temperature_tensor = torch.tensor(temperatures, dtype=torch.float32, device=target_logits.device).unsqueeze(1)
    probabilities = torch.softmax(target_logits.float() / temperature_tensor, dim=-1)
    candidate_probabilities = probabilities.gather(1, draft_tokens.unsqueeze(1)).squeeze(1)
    if random_values is None:
        random_values = torch.rand_like(candidate_probabilities)
    elif random_values.shape != candidate_probabilities.shape:
        raise ValueError("PEARL verification random values must match the packed window.")
    accepted = random_values <= candidate_probabilities

    correction_logits = target_logits.clone()
    correction_logits.scatter_(1, draft_tokens.unsqueeze(1), -float("inf"))
    correction_tokens = _sample_logits(
        correction_logits,
        temperatures,
        exponential_noise=exponential_noise,
    )
    return _build_verdict_from_acceptance(
        accepted,
        correction_tokens,
        verification_sizes,
        gamma,
    )


def _build_verdict_from_acceptance(
    accepted_tokens: torch.Tensor,
    correction_tokens: torch.Tensor,
    verification_sizes: list[int],
    gamma: int,
) -> torch.Tensor:
    if gamma <= 0 or any(size <= 0 or size > gamma for size in verification_sizes):
        raise ValueError("PEARL verification windows must have length in [1, gamma].")
    verification_size = sum(verification_sizes)
    if accepted_tokens.shape != (verification_size,) or correction_tokens.shape != (verification_size,):
        raise ValueError("PEARL verification tensors must match the packed verification size.")
    match_matrix = torch.ones(
        (len(verification_sizes), gamma),
        dtype=torch.bool,
        device=accepted_tokens.device,
    )
    token_matrix = torch.zeros(
        (len(verification_sizes), gamma),
        dtype=torch.long,
        device=accepted_tokens.device,
    )
    offset = 0
    for row, size in enumerate(verification_sizes):
        match_matrix[row, :size] = accepted_tokens[offset : offset + size]
        token_matrix[row, :size] = correction_tokens[offset : offset + size]
        offset += size
    expected = torch.tensor(verification_sizes, dtype=torch.long, device=accepted_tokens.device)
    all_match = match_matrix.all(dim=1)
    first_mismatch = (~match_matrix).to(dtype=torch.int32).argmax(dim=1)
    accepted = torch.where(all_match, expected, first_mismatch)
    correction = token_matrix.gather(1, accepted.clamp(max=gamma - 1).unsqueeze(1)).squeeze(1)
    correction = torch.where(all_match, torch.full_like(correction, -1), correction)
    return torch.stack((accepted, correction), dim=1)


def _gamma_from_decode_speeds(draft_tokens_per_second: float, target_tokens_per_second: float) -> int:
    if draft_tokens_per_second <= 0 or target_tokens_per_second <= 0:
        raise ValueError("PEARL auto-gamma profiling produced a non-positive decode speed.")
    return max(1, round(draft_tokens_per_second / target_tokens_per_second))


def _finished(state: PearlPipelineState, eos_token_ids: frozenset[int]) -> bool:
    completion_token_ids = state.committed_completion_token_ids
    return len(completion_token_ids) >= state.max_tokens or (
        not state.ignore_eos and any(token_id in eos_token_ids for token_id in completion_token_ids)
    )


def _restore_completed_states(
    states: list[PearlPipelineState],
    completed_states: dict[int, PearlPipelineState],
    eos_token_ids: frozenset[int],
) -> None:
    """Bound static padding by replaying completed rows from their snapshot."""
    for index, state in enumerate(states):
        if index in completed_states:
            states[index] = completed_states[index].clone()
        elif _finished(state, eos_token_ids):
            completed_states[index] = state.clone()


def _continuous_bucket_indices(
    unfinished_indices: list[int],
    completed_states: dict[int, PearlPipelineState],
    initial_batch_size: int,
) -> list[int]:
    """Pad a draining continuous batch to at most two stable graph shapes."""
    if not unfinished_indices:
        return []
    tail_bucket_size = max(1, initial_batch_size // 2)
    bucket_size = initial_batch_size if len(unfinished_indices) > tail_bucket_size else tail_bucket_size
    padding_count = bucket_size - len(unfinished_indices)
    if padding_count <= 0:
        return unfinished_indices
    unfinished = set(unfinished_indices)
    padding_indices = [
        index for index in sorted(completed_states) if index not in unfinished
    ][:padding_count]
    if len(padding_indices) != padding_count:
        raise RuntimeError("PEARL continuous batching cannot fill its ACLGraph tail bucket.")
    return [*unfinished_indices, *padding_indices]


def _select_preemptive_continuous_indices(
    states: list[PearlPipelineState],
    unfinished_indices: list[int],
    max_active: int,
    recent_token_gains: Sequence[Sequence[int]] | None = None,
    *,
    decode_elapsed_ms: float = 0.0,
    slo_aware: bool = False,
) -> list[int]:
    """Prioritize SLO debt, then the largest online remaining-work estimate."""
    observed_tokens = sum(
        max(0, len(states[index].committed_completion_token_ids) - 1)
        for index in unfinished_indices
    )
    observed_rounds = sum(states[index].verification_rounds for index in unfinished_indices)
    global_tokens_per_round = observed_tokens / observed_rounds if observed_rounds else 1.0

    def priority(index: int) -> tuple[float, ...]:
        state = states[index]
        rounds = state.verification_rounds
        if rounds < PREEMPTIVE_SCHEDULING_EXPLORATION_ROUNDS:
            base_priority = (0.0, float(rounds), float(index))
        else:
            generated = min(state.max_tokens, len(state.committed_completion_token_ids))
            observed = max(0, generated - 1)
            smoothed_rate = (
                observed + PREEMPTIVE_SCHEDULING_PRIOR_ROUNDS * global_tokens_per_round
            ) / (rounds + PREEMPTIVE_SCHEDULING_PRIOR_ROUNDS)
            if recent_token_gains is not None and recent_token_gains[index]:
                recent = recent_token_gains[index]
                recent_rate = (
                    sum(recent) + PREEMPTIVE_SCHEDULING_PRIOR_ROUNDS * global_tokens_per_round
                ) / (len(recent) + PREEMPTIVE_SCHEDULING_PRIOR_ROUNDS)
                smoothed_rate = min(smoothed_rate, recent_rate)
            estimated_remaining_rounds = (state.max_tokens - generated) / max(smoothed_rate, 1e-6)
            base_priority = (1.0, -estimated_remaining_rounds, float(rounds), float(index))

        if not slo_aware:
            return base_priority
        if state.slo_tpot_ms is None:
            return (1.0, 0.0, *base_priority)
        generated = max(1, len(state.committed_completion_token_ids))
        expected_elapsed_ms = state.slo_tpot_ms * generated
        urgency = max(0.0, decode_elapsed_ms) / max(expected_elapsed_ms, 1e-6)
        return (0.0, -urgency, *base_priority)

    return sorted(
        unfinished_indices,
        key=priority,
    )[:max_active]


def _continuous_result_states(
    states: list[PearlPipelineState],
    completed_states: dict[int, PearlPipelineState],
) -> list[PearlPipelineState]:
    """Return completed snapshots or the current state after an early profile stop."""
    return [completed_states.get(index, state) for index, state in enumerate(states)]


def _truncate_completion(
    completion_token_ids: list[int],
    eos_token_ids: frozenset[int],
    max_tokens: int,
    ignore_eos: bool = False,
) -> list[int]:
    truncated = completion_token_ids[:max_tokens]
    if ignore_eos:
        return truncated
    return truncated[: next((index + 1 for index, token_id in enumerate(truncated) if token_id in eos_token_ids), None)]


def _load_gsm8k_questions(dataset_path: str, max_samples: int) -> list[str]:
    import pyarrow.parquet as pq

    rows = pq.read_table(dataset_path, columns=["question"]).to_pylist()
    return [str(row["question"]) for row in rows[:max_samples]]


def _prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return list(tokenizer.encode(formatted_prompt, add_special_tokens=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=2)
    parser.add_argument("--draft-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--target-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--mode", choices=("pearl", "target-ar"), default="pearl")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--draft-temperature",
        type=float,
        default=0.0,
        help="Optional proposal temperature; target verification remains authoritative.",
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--slo-tpot-ms", type=float)
    parser.add_argument("--slo-class")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--max-aclgraph-entries", type=int, default=32)
    parser.add_argument(
        "--target-verification-graph-buckets",
        type=int,
        default=TARGET_VERIFICATION_GRAPH_BUCKETS,
    )
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--enable-continuous-batching", action="store_true")
    parser.add_argument("--enable-preemptive-scheduling", action="store_true")
    parser.add_argument("--enable-spec-rhythm", action="store_true")
    parser.add_argument("--spec-rhythm-min-gamma", type=int, default=1)
    parser.add_argument("--spec-rhythm-max-eager-tokens", type=int, default=0)
    parser.add_argument("--spec-rhythm-urgency-threshold", type=float, default=0.75)
    parser.add_argument("--spec-rhythm-acceptance-floor", type=float, default=0.4)
    parser.add_argument("--spec-rhythm-acceptance-ema-alpha", type=float, default=0.2)
    parser.add_argument(
        "--spec-rhythm-roofline",
        type=json.loads,
        help='JSON batch/context candidate-token budgets, e.g. {"64:1": 192}.',
    )
    parser.add_argument("--spec-rhythm-draft-token-budget", type=int)
    parser.add_argument("--spec-rhythm-request-max-gamma", type=int)
    parser.add_argument("--disable-cpu-binding", action="store_true")
    parser.add_argument("--target-use-paged-attention", action="store_true")
    parser.add_argument(
        "--draft-use-production-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--target-use-production-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--repeat-prompt", type=int, default=1)
    parser.add_argument("--gsm8k", help="Path to a GSM8K parquet file.")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if bool(args.prompt) == bool(args.gsm8k):
        raise ValueError("Pass exactly one of --prompt or --gsm8k.")
    if args.repeat_prompt <= 0 or (args.gsm8k and args.repeat_prompt != 1):
        raise ValueError("repeat-prompt must be positive and is only valid with --prompt.")
    config = NativePearlConfig(
        draft_model=args.draft_model,
        target_model=args.target_model,
        draft_tp_size=args.draft_tp_size,
        target_tp_size=args.target_tp_size,
        gamma=args.gamma,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        draft_dtype=args.draft_dtype,
        target_dtype=args.target_dtype,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_aclgraph_entries=args.max_aclgraph_entries,
        target_verification_graph_buckets=args.target_verification_graph_buckets,
        enable_continuous_batching=(
            args.enable_continuous_batching or args.enable_spec_rhythm
        ),
        enable_preemptive_scheduling=(
            args.enable_preemptive_scheduling or args.enable_spec_rhythm
        ),
        enable_spec_rhythm=args.enable_spec_rhythm,
        spec_rhythm_min_gamma=args.spec_rhythm_min_gamma,
        spec_rhythm_max_eager_tokens=args.spec_rhythm_max_eager_tokens,
        spec_rhythm_urgency_threshold=args.spec_rhythm_urgency_threshold,
        spec_rhythm_acceptance_floor=args.spec_rhythm_acceptance_floor,
        spec_rhythm_acceptance_ema_alpha=args.spec_rhythm_acceptance_ema_alpha,
        spec_rhythm_roofline=args.spec_rhythm_roofline,
        spec_rhythm_draft_token_budget=args.spec_rhythm_draft_token_budget,
        target_use_paged_attention=args.target_use_paged_attention,
        draft_use_production_rope=args.draft_use_production_rope,
        target_use_production_rope=args.target_use_production_rope,
        enable_prefix_caching=not args.disable_prefix_caching,
        enable_cpu_binding=not args.disable_cpu_binding,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    engine = NativePearlEngine(config)
    sampling_params = NativeSamplingParams(
        temperature=args.temperature,
        draft_temperature=args.draft_temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        slo_tpot_ms=args.slo_tpot_ms,
        slo_class=args.slo_class,
        spec_rhythm_max_gamma=args.spec_rhythm_request_max_gamma,
    )
    prompts = [args.prompt] * args.repeat_prompt if args.prompt else _load_gsm8k_questions(args.gsm8k, args.max_samples)
    results: list[dict[str, Any]] = []
    total_elapsed = 0.0
    for start in range(0, len(prompts), args.batch_size):
        prompt_batch = prompts[start : start + args.batch_size]
        prompt_token_ids = [_prompt_token_ids(engine.tokenizer, prompt) for prompt in prompt_batch]
        if args.mode == "pearl":
            batch_results = engine.generate_batch(prompt_token_ids, sampling_params)
        else:
            batch_results = engine.generate_target_ar_batch(prompt_token_ids, sampling_params)
        if batch_results is not None:
            total_elapsed += batch_results[0]["elapsed_seconds"]
            for prompt, result in zip(prompt_batch, batch_results):
                result["prompt"] = prompt
                result["text"] = engine.tokenizer.decode(result["completion_token_ids"], skip_special_tokens=True)
                results.append(result)
    if engine.rank == engine.topology.target_leader_rank:
        total_verified = sum(result["verified_draft_tokens"] for result in results)
        total_accepted = sum(result["accepted_draft_tokens"] for result in results)
        generated_token_count = sum(len(result["completion_token_ids"]) for result in results)
        aggregate_mat = sum(result["mean_accept_tokens"] for result in results) / len(results) if results else 0.0
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "samples": [] if args.summary_only else results,
                    "num_samples": len(results),
                    "generated_token_count": generated_token_count,
                    "aggregate_acceptance_rate": total_accepted / total_verified if total_verified else 0.0,
                    "aggregate_mat": aggregate_mat,
                    "decode_throughput_tokens_per_second": (
                        generated_token_count / total_elapsed if total_elapsed else 0.0
                    ),
                },
                ensure_ascii=True,
            )
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

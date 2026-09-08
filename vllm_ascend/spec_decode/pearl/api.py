# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream-compatible nano-PEARL controller backed by Ascend HCCL workers."""

from __future__ import annotations

import atexit
import logging
import os
import socket
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_for_connections
from typing import Any, Mapping

import torch.multiprocessing as mp
from transformers import AutoConfig, AutoTokenizer

from vllm_ascend.spec_decode.pearl.native_engine import (
    TARGET_VERIFICATION_GRAPH_BUCKETS,
    NativePearlConfig,
    NativePearlEngine,
    SamplingParams,
    _normalize_target_graph_post_counts,
    _set_default_npu_environment,
)
from vllm_ascend.spec_decode.pearl.native_model import (
    PAGED_ATTENTION_BLOCK_SIZE,
    SUPPORTED_NATIVE_ARCHITECTURES,
)
from vllm_ascend.spec_decode.pearl.qwen_pair import validate_model_pair

logger = logging.getLogger("vllm_ascend.spec_decode.pearl")


@dataclass(frozen=True)
class PEARLModelGroupConfig:
    """Read-only compatibility view of an upstream nano-PEARL model group."""

    model: str
    tensor_parallel_size: int
    devices: list[int]
    group_name: str
    hf_config: Any
    eos: int | list[int] | None
    master_rank: int


@dataclass(frozen=True)
class PEARLConfig:
    """Configuration surface compatible with upstream nano-PEARL."""

    draft_model_path: str
    target_model_path: str
    draft_tensor_parallel_size: int = 2
    target_tensor_parallel_size: int = 2
    draft_group_name: str = "draft_group"
    target_group_name: str = "target_group"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    prefill_chunk_size: int | None = None
    max_num_queued_seqs: int | None = None
    max_model_len: int = 4096
    draft_dtype: str = "auto"
    target_dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = PAGED_ATTENTION_BLOCK_SIZE
    num_kvcache_blocks: int = -1
    max_aclgraph_entries: int = 32
    target_verification_graph_buckets: int = TARGET_VERIFICATION_GRAPH_BUCKETS
    target_verification_graph_post_counts: tuple[tuple[int, tuple[int, ...]], ...] = ()
    auto_gamma_profile_sequence_length: int = 256
    enforce_eager: bool = False
    gamma: int = -1
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
    seed: int | None = None
    worker_timeout_seconds: float = 300.0
    draft_config: PEARLModelGroupConfig = field(init=False, repr=False)
    target_config: PEARLModelGroupConfig = field(init=False, repr=False)
    eos: int | list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_verification_graph_post_counts",
            _normalize_target_graph_post_counts(
                self.target_verification_graph_post_counts
            ),
        )
        if self.draft_tensor_parallel_size <= 0 or self.target_tensor_parallel_size <= 0:
            raise ValueError("PEARL tensor-parallel sizes must be positive.")
        if self.world_size > 8:
            raise ValueError("Upstream nano-PEARL supports at most eight model workers.")
        if self.max_model_len <= 0 or self.max_num_seqs <= 0:
            raise ValueError("PEARL max_model_len and max_num_seqs must be positive.")
        supported_dtypes = {"auto", "bfloat16", "float16"}
        if self.draft_dtype not in supported_dtypes or self.target_dtype not in supported_dtypes:
            raise ValueError("PEARL model dtype must be auto, bfloat16, or float16.")
        if self.prefill_chunk_size is not None and not 0 < self.prefill_chunk_size <= self.max_num_seqs:
            raise ValueError("PEARL prefill_chunk_size must be in [1, max_num_seqs].")
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
            raise ValueError(
                f"The Ascend PEARL paged-attention backend requires kvcache_block_size={PAGED_ATTENTION_BLOCK_SIZE}."
            )
        if self.num_kvcache_blocks == 0 or self.num_kvcache_blocks < -1:
            raise ValueError("PEARL num_kvcache_blocks must be positive, or -1 for automatic sizing.")
        if self.max_aclgraph_entries <= 0:
            raise ValueError("PEARL max_aclgraph_entries must be positive.")
        if self.target_verification_graph_buckets <= 0:
            raise ValueError("PEARL target_verification_graph_buckets must be positive.")
        if self.auto_gamma_profile_sequence_length <= 0:
            raise ValueError("PEARL auto-gamma profile sequence length must be positive.")
        if self.profile_decode_steps < 0:
            raise ValueError("PEARL profile_decode_steps must be non-negative.")
        if self.stop_after_profiled_decode_steps and self.profile_decode_steps == 0:
            raise ValueError("PEARL profiling-only execution requires profile_decode_steps to be positive.")
        if self.gamma == 0 or self.gamma < -1:
            raise ValueError("PEARL gamma must be positive, or -1 for automatic selection.")
        if self.seed is not None and self.seed < 0:
            raise ValueError("PEARL seed must be non-negative.")
        if self.worker_timeout_seconds <= 0:
            raise ValueError("PEARL worker_timeout_seconds must be positive.")

        draft_config = AutoConfig.from_pretrained(self.draft_model_path)
        target_config = AutoConfig.from_pretrained(self.target_model_path)
        for name, model_config in (("draft", draft_config), ("target", target_config)):
            architecture = model_config.architectures[0]
            if architecture not in SUPPORTED_NATIVE_ARCHITECTURES:
                raise ValueError(
                    f"Unsupported {name} architecture {architecture!r}; expected one of "
                    f"{sorted(SUPPORTED_NATIVE_ARCHITECTURES)}."
                )
        if _eos_set(draft_config.eos_token_id) != _eos_set(target_config.eos_token_id):
            raise ValueError("PEARL draft and target models must use identical EOS token IDs.")
        draft_devices = list(range(self.draft_tensor_parallel_size))
        target_devices = list(
            range(
                self.draft_tensor_parallel_size,
                self.draft_tensor_parallel_size + self.target_tensor_parallel_size,
            )
        )
        object.__setattr__(
            self,
            "draft_config",
            PEARLModelGroupConfig(
                model=self.draft_model_path,
                tensor_parallel_size=self.draft_tensor_parallel_size,
                devices=draft_devices,
                group_name=self.draft_group_name,
                hf_config=draft_config,
                eos=draft_config.eos_token_id,
                master_rank=draft_devices[0],
            ),
        )
        object.__setattr__(
            self,
            "target_config",
            PEARLModelGroupConfig(
                model=self.target_model_path,
                tensor_parallel_size=self.target_tensor_parallel_size,
                devices=target_devices,
                group_name=self.target_group_name,
                hf_config=target_config,
                eos=target_config.eos_token_id,
                master_rank=target_devices[0],
            ),
        )
        object.__setattr__(self, "eos", draft_config.eos_token_id)

    @property
    def world_size(self) -> int:
        return self.draft_tensor_parallel_size + self.target_tensor_parallel_size

    def to_native(self) -> NativePearlConfig:
        return NativePearlConfig(
            draft_model=self.draft_model_path,
            target_model=self.target_model_path,
            draft_tp_size=self.draft_tensor_parallel_size,
            target_tp_size=self.target_tensor_parallel_size,
            gamma=self.gamma,
            max_model_len=self.max_model_len,
            max_tokens=self.max_model_len,
            draft_dtype=self.draft_dtype,
            target_dtype=self.target_dtype,
            max_num_seqs=self.max_num_seqs,
            prefill_chunk_size=self.prefill_chunk_size,
            max_num_queued_seqs=self.max_num_queued_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            gpu_memory_utilization=self.gpu_memory_utilization,
            kvcache_block_size=self.kvcache_block_size,
            num_kvcache_blocks=self.num_kvcache_blocks,
            max_aclgraph_entries=self.max_aclgraph_entries,
            target_verification_graph_buckets=self.target_verification_graph_buckets,
            target_verification_graph_post_counts=self.target_verification_graph_post_counts,
            auto_gamma_profile_sequence_length=self.auto_gamma_profile_sequence_length,
            enable_prefix_caching=self.enable_prefix_caching,
            enable_continuous_batching=self.enable_continuous_batching,
            enable_preemptive_scheduling=self.enable_preemptive_scheduling,
            enable_spec_rhythm=self.enable_spec_rhythm,
            spec_rhythm_min_gamma=self.spec_rhythm_min_gamma,
            spec_rhythm_max_eager_tokens=self.spec_rhythm_max_eager_tokens,
            spec_rhythm_urgency_threshold=self.spec_rhythm_urgency_threshold,
            spec_rhythm_acceptance_floor=self.spec_rhythm_acceptance_floor,
            spec_rhythm_acceptance_ema_alpha=self.spec_rhythm_acceptance_ema_alpha,
            spec_rhythm_roofline=self.spec_rhythm_roofline,
            spec_rhythm_draft_token_budget=self.spec_rhythm_draft_token_budget,
            pad_finished_requests=self.pad_finished_requests,
            draft_use_paged_attention=self.draft_use_paged_attention,
            target_use_paged_attention=self.target_use_paged_attention,
            draft_use_production_rope=self.draft_use_production_rope,
            target_use_production_rope=self.target_use_production_rope,
            precompile_decode_graphs=self.precompile_decode_graphs,
            enable_cpu_binding=self.enable_cpu_binding,
            profile_decode_steps=self.profile_decode_steps,
            stop_after_profiled_decode_steps=self.stop_after_profiled_decode_steps,
            enforce_eager=self.enforce_eager,
            seed=self.seed,
        )


class PEARLEngine:
    """Spawn and control the draft and target HCCL workers like nano-PEARL."""

    def __init__(self, config: PEARLConfig) -> None:
        self.config = config
        validate_model_pair(config.draft_model_path, config.target_model_path)
        # Worker processes must inherit queue mode before importing torch-npu;
        # setting it inside NativePearlEngine is too late for this runtime knob.
        _set_default_npu_environment(config.target_tensor_parallel_size)
        self.tokenizer = AutoTokenizer.from_pretrained(config.draft_model_path, use_fast=True)
        self._requests: list[tuple[int, list[int], SamplingParams]] = []
        self._next_request_id = 0
        self.last_metrics: list[dict[str, Any]] = []
        self.last_worker_metrics: list[dict[str, int | float]] = []
        self.last_worker_metrics_by_chunk: list[dict[str, Any]] = []
        self._closed = False
        self._processes: list[mp.Process] = []
        self._connections: list[Connection] = []
        try:
            self._start_workers()
        except Exception:
            self.exit()
            raise
        atexit.register(self.exit)

    def _start_workers(self) -> None:
        context = mp.get_context("spawn")
        master_port = _reserve_local_port()
        native_config = self.config.to_native()
        for rank in range(self.config.world_size):
            parent_connection, child_connection = context.Pipe(duplex=True)
            process = context.Process(
                target=_pearl_worker,
                args=(native_config, rank, master_port, child_connection),
                daemon=True,
            )
            process.start()
            child_connection.close()
            self._processes.append(process)
            self._connections.append(parent_connection)
        replies = self._receive_all("worker initialization")
        if any(reply[0] != "ready" for reply in replies):
            raise RuntimeError(f"Unexpected PEARL worker initialization replies: {replies!r}")
        self.last_worker_metrics = [reply[2] for reply in replies if len(reply) > 2]

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | int | None = None,
        arrival_ts: float | None = None,
        slo_tpot_ms: float | None = None,
        slo_class: str | None = None,
        per_request_gamma: int | None = None,
    ) -> None:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("PEARL sampling_params must be a SamplingParams value.")
        sampling_params = replace(
            sampling_params,
            request_id=(
                request_id
                if request_id is not None
                else sampling_params.request_id
            ),
            arrival_ts=(
                arrival_ts
                if arrival_ts is not None
                else sampling_params.arrival_ts
            ),
            slo_tpot_ms=(
                slo_tpot_ms
                if slo_tpot_ms is not None
                else sampling_params.slo_tpot_ms
            ),
            slo_class=(
                slo_class if slo_class is not None else sampling_params.slo_class
            ),
            spec_rhythm_max_gamma=(
                per_request_gamma
                if per_request_gamma is not None
                else sampling_params.spec_rhythm_max_gamma
            ),
        )
        if isinstance(prompt, str):
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            token_ids = list(self.tokenizer.encode(formatted_prompt))
        else:
            token_ids = [int(token_id) for token_id in prompt]
        if not token_ids:
            raise ValueError("PEARL requests require a non-empty prompt.")
        if len(token_ids) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError("Prompt plus PEARL completion exceeds max_model_len.")
        self._requests.append((self._next_request_id, token_ids, sampling_params))
        self._next_request_id += 1

    def log(self, content: str) -> None:
        """Emit one controller message from every PEARL worker."""
        self._send_all(("log", str(content), None, None))
        replies = self._receive_all("worker logging")
        if any(reply[0] != "logged" for reply in replies):
            raise RuntimeError(f"Unexpected PEARL worker log replies: {replies!r}")

    def configure_decode_profiling(
        self,
        profile_decode_steps: int,
        stop_after_profiled_decode_steps: bool = False,
    ) -> None:
        """Configure synchronized decode profiling on every persistent worker."""
        if profile_decode_steps < 0:
            raise ValueError("PEARL profile_decode_steps must be non-negative.")
        if stop_after_profiled_decode_steps and profile_decode_steps == 0:
            raise ValueError("PEARL profiling-only execution requires profile_decode_steps to be positive.")
        self._send_all(
            (
                "configure_decode_profiling",
                profile_decode_steps,
                stop_after_profiled_decode_steps,
                None,
            )
        )
        replies = self._receive_all("worker profiling configuration")
        if any(reply[0] != "configured" for reply in replies):
            raise RuntimeError(f"Unexpected PEARL worker profiling replies: {replies!r}")

    def generate(self):
        if self.config.gamma != -1 and any(
            len(prompt) + params.max_tokens + self.config.gamma > self.config.max_model_len
            for _, prompt, params in self._requests
        ):
            raise ValueError("Prompt plus the PEARL verification window exceeds max_model_len.")
        return self._generate("pearl")

    def AR_generate(self):
        output_text, num_tokens, _, elapsed = self._generate("target_ar")
        return output_text, num_tokens, None, elapsed

    def bench_generate(self, num_pearl_steps: int = 100):
        if num_pearl_steps <= 0:
            raise ValueError("num_pearl_steps must be positive.")
        if self.config.gamma != -1 and any(
            len(prompt) + 1 + (num_pearl_steps + 2) * self.config.gamma > self.config.max_model_len
            for _, prompt, _ in self._requests
        ):
            raise ValueError("The fixed PEARL benchmark steps exceed max_model_len.")
        return self._generate("bench", num_pearl_steps=num_pearl_steps)

    def _generate(self, mode: str, *, num_pearl_steps: int | None = None):
        if not self._requests:
            self.last_metrics = []
            self.last_worker_metrics = []
            self.last_worker_metrics_by_chunk = []
            return [], [], (() if mode != "target_ar" else None), 0.0
        requests = self._requests
        self._requests = []
        outputs: list[tuple[int, dict[str, Any]]] = []
        total_elapsed = 0.0
        self.last_worker_metrics_by_chunk = []
        try:
            request_chunks = (
                [requests]
                if mode == "pearl" and getattr(self.config, "enable_continuous_batching", False)
                else self._request_chunks(requests)
            )
            for chunk in request_chunks:
                request_ids = [request_id for request_id, _, _ in chunk]
                prompts = [prompt for _, prompt, _ in chunk]
                params = [params for _, _, params in chunk]
                if mode == "bench":
                    params = [
                        replace(params, max_tokens=self.config.max_model_len, ignore_eos=True) for params in params
                    ]
                self._send_all((mode, prompts, params, num_pearl_steps))
                replies = self._receive_all(f"{mode} generation")
                leader_payloads = [reply[1] for reply in replies if reply[0] == "result" and reply[1] is not None]
                if len(leader_payloads) != 1:
                    raise RuntimeError("PEARL target leader did not return exactly one result batch.")
                batch_results = leader_payloads[0]
                worker_graph_metrics = [reply[2] for reply in replies if len(reply) > 2]
                if worker_graph_metrics:
                    self.last_worker_metrics = worker_graph_metrics
                    self.last_worker_metrics_by_chunk.append(
                        {
                            "batch_size": min(len(chunk), self.config.max_num_seqs),
                            "num_requests": len(chunk),
                            "worker_metrics": worker_graph_metrics,
                        }
                    )
                    aggregate_graph_metrics = {
                        name: max(metrics[name] for metrics in worker_graph_metrics) for name in worker_graph_metrics[0]
                    }
                    for result in batch_results:
                        result.update(aggregate_graph_metrics)
                total_elapsed += batch_results[0]["elapsed_seconds"] if batch_results else 0.0
                outputs.extend(zip(request_ids, batch_results))
        except Exception:
            self._requests = requests + self._requests
            raise

        outputs.sort(key=lambda item: item[0])
        results = [result for _, result in outputs]
        self.last_metrics = results
        output_text = [
            self.tokenizer.decode(result["completion_token_ids"], skip_special_tokens=False) for result in results
        ]
        num_tokens = [len(result["completion_token_ids"]) for result in results]
        num_acc_tokens = tuple(result["num_acc_tokens"] for result in results)
        return output_text, num_tokens, num_acc_tokens, total_elapsed

    def _request_chunks(
        self,
        requests: list[tuple[int, list[int], SamplingParams]],
    ) -> list[list[tuple[int, list[int], SamplingParams]]]:
        chunks: list[list[tuple[int, list[int], SamplingParams]]] = []
        current: list[tuple[int, list[int], SamplingParams]] = []
        current_tokens = 0
        for request in requests:
            prompt_tokens = len(request[1])
            if prompt_tokens > self.config.max_num_batched_tokens:
                raise ValueError("A PEARL prompt exceeds max_num_batched_tokens.")
            if current and (
                len(current) == self.config.max_num_seqs
                or current_tokens + prompt_tokens > self.config.max_num_batched_tokens
            ):
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(request)
            current_tokens += prompt_tokens
        if current:
            chunks.append(current)
        return chunks

    def _send_all(self, message: tuple[Any, ...]) -> None:
        if self._closed:
            raise RuntimeError("The PEARL engine is closed.")
        for connection in self._connections:
            connection.send(message)

    def _receive_all(self, operation: str) -> list[tuple[str, Any]]:
        replies: list[tuple[str, Any] | None] = [None] * len(self._connections)
        pending = {
            connection: (rank, process)
            for rank, (process, connection) in enumerate(zip(self._processes, self._connections))
        }
        deadline = time.monotonic() + self.config.worker_timeout_seconds
        while pending:
            for connection, (rank, process) in tuple(pending.items()):
                if not process.is_alive():
                    raise RuntimeError(f"PEARL rank {rank} exited during {operation} with code {process.exitcode}.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pending_ranks = sorted(rank for rank, _ in pending.values())
                raise TimeoutError(
                    f"PEARL timed out after {self.config.worker_timeout_seconds:g}s during {operation}; "
                    f"pending ranks: {pending_ranks}."
                )
            for connection in wait_for_connections(pending, timeout=min(1.0, remaining)):
                rank, process = pending.pop(connection)
                try:
                    reply = connection.recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError(
                        f"PEARL rank {rank} exited during {operation} with code {process.exitcode}."
                    ) from error
                if reply[0] == "error":
                    raise RuntimeError(f"PEARL rank {rank} failed during {operation}:\n{reply[1]}")
                replies[rank] = reply
        return [reply for reply in replies if reply is not None]

    def exit(self) -> None:
        if self._closed:
            return
        self._closed = True
        for process, connection in zip(self._processes, self._connections):
            if process.is_alive():
                with suppress(BrokenPipeError, EOFError):
                    connection.send(("exit", None, None, None))
        graceful_deadline = time.monotonic() + 30
        for process in self._processes:
            process.join(timeout=max(0.0, graceful_deadline - time.monotonic()))
        live_processes = [process for process in self._processes if process.is_alive()]
        for process in live_processes:
            process.terminate()
        terminate_deadline = time.monotonic() + 5
        for process in live_processes:
            process.join(timeout=max(0.0, terminate_deadline - time.monotonic()))
        for connection in self._connections:
            connection.close()

    def __enter__(self) -> PEARLEngine:
        return self

    def __exit__(self, *_args) -> None:
        self.exit()


def _pearl_worker(
    config: NativePearlConfig,
    rank: int,
    master_port: int,
    connection: Connection,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(config.draft_tp_size + config.target_tp_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    engine: NativePearlEngine | None = None
    try:
        engine = NativePearlEngine(config)
        connection.send(("ready", rank, engine.graph_metrics()))
        while True:
            mode, prompts, sampling_params, num_pearl_steps = connection.recv()
            if mode == "exit":
                break
            if mode == "log":
                logger.info("[PEARL rank %d] %s", rank, prompts)
                connection.send(("logged", rank))
                continue
            if mode == "configure_decode_profiling":
                engine.configure_decode_profiling(int(prompts), bool(sampling_params))
                connection.send(("configured", rank))
                continue
            if mode == "pearl":
                result = engine.generate_batch(prompts, sampling_params)
            elif mode == "target_ar":
                result = engine.generate_target_ar_batch(prompts, sampling_params)
            elif mode == "bench":
                result = engine.generate_batch(
                    prompts,
                    sampling_params,
                    max_rounds=num_pearl_steps,
                )
            else:
                raise ValueError(f"Unknown PEARL worker command {mode!r}.")
            connection.send(("result", result, engine.graph_metrics()))
    except BaseException:
        with suppress(BrokenPipeError, EOFError):
            connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()
        if engine is not None:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _eos_set(eos_token_id: int | list[int] | None) -> frozenset[int]:
    if eos_token_id is None:
        return frozenset()
    if isinstance(eos_token_id, int):
        return frozenset((eos_token_id,))
    return frozenset(int(token_id) for token_id in eos_token_id)


__all__ = ["PEARLConfig", "PEARLEngine", "SamplingParams", "logger"]

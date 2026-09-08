# SPDX-License-Identifier: Apache-2.0
"""Benchmark the native nano-PEARL speculative path with one engine load."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=1)
    parser.add_argument("--draft-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--target-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        help="Limit packed prefill requests without changing the decode batch size.",
    )
    parser.add_argument(
        "--num-pearl-steps",
        type=int,
        help="Run upstream-compatible fixed-step PEARL instead of fixed-token generation.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument(
        "--num-prompts",
        type=int,
        help="Total requests per measurement. Defaults to the batch size.",
    )
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        help="Warm up on only this many prompts. Defaults to one full batch.",
    )
    parser.add_argument(
        "--warmup-prompt-offset",
        type=int,
        default=0,
        help="Start warmup prompts at this dataset offset.",
    )
    parser.add_argument(
        "--warmup-max-tokens",
        type=int,
        help="Limit warmup generation tokens without changing the measured request limit.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of untimed warmup generations to run before measurement.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--max-aclgraph-entries", type=int, default=32)
    parser.add_argument("--target-verification-graph-buckets", type=int, default=8)
    parser.add_argument(
        "--target-verification-graph-post-counts",
        action="append",
        default=[],
        metavar="BATCH:COUNT,COUNT,...",
        help=(
            "Use explicit target verification graph post-verify counts for a "
            "batch size. May be repeated; counts must start at 0 and end at BATCH."
        ),
    )
    parser.add_argument("--auto-gamma-profile-sequence-length", type=int, default=256)
    parser.add_argument(
        "--profile-decode-steps",
        type=int,
        default=0,
        help="Synchronously profile this many decode steps per static chunk.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Stop after the synchronously profiled decode steps.",
    )
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-continuous-batching", action="store_true")
    parser.add_argument(
        "--enable-preemptive-scheduling",
        action="store_true",
        help="Balance decode rounds across all resident continuous requests.",
    )
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
    parser.add_argument("--slo-tpot-ms", type=float)
    parser.add_argument("--slo-class")
    parser.add_argument("--pad-finished-requests", action="store_true")
    parser.add_argument("--draft-use-paged-attention", action="store_true")
    parser.add_argument("--target-use-paged-attention", action="store_true")
    parser.add_argument(
        "--draft-use-production-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM-Ascend's fused rotary operator in the draft model.",
    )
    parser.add_argument(
        "--target-use-production-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM-Ascend's fused rotary operator in the target model.",
    )
    parser.add_argument(
        "--precompile-decode-graphs",
        action="store_true",
        help="Compile fixed paged-attention decode graph buckets during engine initialization.",
    )
    parser.add_argument(
        "--disable-cpu-binding",
        action="store_true",
        help="Disable the production vLLM-Ascend CPU and IRQ affinity policy.",
    )
    parser.add_argument(
        "--output-json",
        help="Also write the final benchmark payload to this path.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--gsm8k", help="Path to a GSM8K parquet or JSONL file.")
    prompt_group.add_argument(
        "--request-manifest",
        help="JSONL rows with prompt and max_tokens, shared with the baseline.",
    )
    return parser


def _load_prompts(
    prompt: str | None,
    gsm8k: str | None,
    request_manifest: str | None,
    max_samples: int,
) -> tuple[list[str], list[int] | None, list[dict] | None]:
    if prompt is not None:
        return [prompt] * max_samples, None, None
    if request_manifest is not None:
        rows = [
            json.loads(line)
            for line in Path(request_manifest).read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(rows) < max_samples:
            raise ValueError(
                f"Request manifest contains {len(rows)} rows, but {max_samples} were requested."
            )
        selected = rows[:max_samples]
        return (
            [str(row["prompt"]) for row in selected],
            [int(row["max_tokens"]) for row in selected],
            selected,
        )
    dataset_path = Path(gsm8k)
    if dataset_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) < max_samples:
            raise ValueError(f"GSM8K contains {len(rows)} rows, but {max_samples} were requested.")
        return [str(row["turns"][0]) for row in rows[:max_samples]], None, None
    import pyarrow.parquet as pq

    rows = pq.read_table(dataset_path, columns=["question"]).to_pylist()
    if len(rows) < max_samples:
        raise ValueError(f"GSM8K contains {len(rows)} rows, but batch size {max_samples} was requested.")
    return [str(row["question"]) for row in rows[:max_samples]], None, None


def _add_requests(engine, prompts, sampling_params) -> None:
    if isinstance(sampling_params, Sequence):
        if len(prompts) != len(sampling_params):
            raise ValueError("PEARL requires one SamplingParams value per prompt.")
        for prompt, params in zip(prompts, sampling_params):
            engine.add_request(prompt, params)
        return
    for prompt in prompts:
        engine.add_request(prompt, sampling_params)


def _parse_target_graph_post_counts(
    values: Sequence[str],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    parsed: list[tuple[int, tuple[int, ...]]] = []
    for value in values:
        try:
            batch_size_text, post_counts_text = value.split(":", 1)
            post_counts = tuple(
                int(item) for item in post_counts_text.split(",") if item
            )
            parsed.append((int(batch_size_text), post_counts))
        except ValueError as error:
            raise ValueError(
                "--target-verification-graph-post-counts must use "
                "BATCH:COUNT,COUNT,..."
            ) from error
    return tuple(parsed)


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("Every speculative batch size must be positive.")
    if args.num_prompts is not None and args.num_prompts <= 0:
        raise ValueError("The speculative prompt count must be positive.")
    if args.warmup_prompts is not None and args.warmup_prompts <= 0:
        raise ValueError("--warmup-prompts must be positive.")
    if args.warmup_prompt_offset < 0:
        raise ValueError("--warmup-prompt-offset must be non-negative.")
    if args.warmup_max_tokens is not None and args.warmup_max_tokens <= 0:
        raise ValueError("--warmup-max-tokens must be positive.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative.")
    if args.profile_decode_steps < 0:
        raise ValueError("--profile-decode-steps must be non-negative.")
    if args.profile_only and args.profile_decode_steps == 0:
        raise ValueError("--profile-only requires --profile-decode-steps to be positive.")
    if args.num_pearl_steps is not None and args.num_pearl_steps <= 0:
        raise ValueError("--num-pearl-steps must be positive.")
    if args.num_pearl_steps is not None and args.enable_continuous_batching:
        raise ValueError("Fixed-step PEARL does not support continuous batching.")
    max_batch_size = max(args.batch_sizes)
    target_graph_post_counts = _parse_target_graph_post_counts(
        args.target_verification_graph_post_counts
    )

    from vllm_ascend.spec_decode.pearl import PEARLConfig, PEARLEngine, SamplingParams

    prompt_count = args.num_prompts or max_batch_size
    max_warmup_count = args.warmup_prompts or max_batch_size
    load_count = (
        max(prompt_count, args.warmup_prompt_offset + max_warmup_count)
        if args.warmup_runs
        else prompt_count
    )
    prompts, request_max_tokens, request_metadata = _load_prompts(
        args.prompt,
        args.gsm8k,
        args.request_manifest,
        load_count,
    )
    config = PEARLConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=args.draft_tp_size,
        target_tensor_parallel_size=args.target_tp_size,
        draft_dtype=args.draft_dtype,
        target_dtype=args.target_dtype,
        max_num_batched_tokens=args.max_model_len * max_batch_size,
        max_num_seqs=max_batch_size,
        prefill_chunk_size=args.prefill_chunk_size,
        max_num_queued_seqs=(
            prompt_count
            if args.enable_continuous_batching or args.enable_spec_rhythm
            else None
        ),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_aclgraph_entries=args.max_aclgraph_entries,
        target_verification_graph_buckets=args.target_verification_graph_buckets,
        target_verification_graph_post_counts=target_graph_post_counts,
        auto_gamma_profile_sequence_length=args.auto_gamma_profile_sequence_length,
        enable_prefix_caching=args.enable_prefix_caching,
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
        pad_finished_requests=args.pad_finished_requests,
        draft_use_paged_attention=args.draft_use_paged_attention,
        target_use_paged_attention=args.target_use_paged_attention,
        draft_use_production_rope=args.draft_use_production_rope,
        target_use_production_rope=args.target_use_production_rope,
        precompile_decode_graphs=args.precompile_decode_graphs,
        enable_cpu_binding=not args.disable_cpu_binding,
        profile_decode_steps=0,
        stop_after_profiled_decode_steps=False,
        enforce_eager=args.enforce_eager,
        gamma=args.gamma,
        seed=args.seed,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    sampling_params = (
        [
            SamplingParams(
                temperature=0.0,
                max_tokens=int(row["max_tokens"]),
                ignore_eos=True,
                request_id=row.get("request_id"),
                arrival_ts=(
                    float(row["arrival_ts"])
                    if row.get("arrival_ts") is not None
                    else None
                ),
                slo_tpot_ms=(
                    float(row["slo_tpot_ms"])
                    if row.get("slo_tpot_ms") is not None
                    else None
                ),
                slo_class=row.get("slo_class"),
                spec_rhythm_max_gamma=(
                    int(row["per_request_gamma"])
                    if row.get("per_request_gamma") is not None
                    else args.spec_rhythm_request_max_gamma
                ),
            )
            for row in request_metadata
        ]
        if request_metadata is not None
        else SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            ignore_eos=True,
            slo_tpot_ms=args.slo_tpot_ms,
            slo_class=args.slo_class,
            spec_rhythm_max_gamma=args.spec_rhythm_request_max_gamma,
        )
    )
    warmup_sampling_params = (
        [
            SamplingParams(
                temperature=0.0,
                max_tokens=min(value, args.warmup_max_tokens or value),
                ignore_eos=True,
            )
            for value in request_max_tokens
        ]
        if request_max_tokens is not None
        else SamplingParams(
            temperature=0.0,
            max_tokens=args.warmup_max_tokens or args.max_tokens,
            ignore_eos=True,
        )
    )

    results = []
    with PEARLEngine(config) as engine:
        first_formatted_prompt = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[0]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        first_prompt_token_ids = list(engine.tokenizer.encode(first_formatted_prompt))
        for batch_size in args.batch_sizes:
            engine.configure_decode_profiling(0)
            measured_prompts = prompts[: args.num_prompts or batch_size]
            measured_sampling_params = (
                sampling_params[: len(measured_prompts)]
                if isinstance(sampling_params, list)
                else sampling_params
            )
            def materialize_arrivals(params):
                if request_metadata is None:
                    return params
                arrival_origin = time.time()
                return [
                    replace(
                        value,
                        arrival_ts=(
                            float(row["arrival_ts"])
                            if row.get("arrival_ts") is not None
                            else arrival_origin
                            + float(row.get("arrival_offset_sec", 0.0))
                        ),
                    )
                    for value, row in zip(
                        params,
                        request_metadata[: len(measured_prompts)],
                    )
                ]
            warmup_count = args.warmup_prompts or batch_size
            warmup_start = args.warmup_prompt_offset
            warmup_prompts = prompts[warmup_start : warmup_start + warmup_count]
            current_warmup_params = (
                warmup_sampling_params[warmup_start : warmup_start + warmup_count]
                if isinstance(warmup_sampling_params, list)
                else warmup_sampling_params
            )
            for _ in range(args.warmup_runs):
                _add_requests(engine, warmup_prompts, current_warmup_params)
                if args.num_pearl_steps is None:
                    engine.generate()
                else:
                    engine.bench_generate(args.num_pearl_steps)

            if args.profile_only:
                engine.configure_decode_profiling(
                    args.profile_decode_steps,
                    stop_after_profiled_decode_steps=True,
                )
                _add_requests(
                    engine,
                    measured_prompts,
                    materialize_arrivals(measured_sampling_params),
                )
                engine.generate()
            warmup_worker_metrics = engine.last_worker_metrics

            engine.configure_decode_profiling(
                args.profile_decode_steps,
                args.profile_only,
            )
            _add_requests(
                engine,
                measured_prompts,
                materialize_arrivals(measured_sampling_params),
            )
            started = time.perf_counter()
            if args.num_pearl_steps is None:
                _, num_tokens, _, inference_elapsed = engine.generate()
            else:
                _, num_tokens, _, inference_elapsed = engine.bench_generate(args.num_pearl_steps)
            e2e_elapsed = time.perf_counter() - started
            output_tokens = sum(num_tokens)
            metrics = engine.last_metrics
            output_token_rows = [metric["completion_token_ids"] for metric in metrics]
            verified_tokens = sum(metric["verified_draft_tokens"] for metric in metrics)
            accepted_tokens = sum(metric["accepted_draft_tokens"] for metric in metrics)
            chunk_metrics = metrics[:1] if args.enable_continuous_batching else metrics[::batch_size]
            decode_phase_seconds = {
                phase: sum(metric["decode_phase_seconds"][phase] for metric in chunk_metrics)
                for phase in chunk_metrics[0]["decode_phase_seconds"]
            }
            results.append(
                {
                    "batch_size": batch_size,
                    "num_prompts": len(measured_prompts),
                    "num_static_chunks": (
                        1 if args.enable_continuous_batching else math.ceil(len(measured_prompts) / batch_size)
                    ),
                    "output_tokens": output_tokens,
                    "inference_elapsed_seconds": inference_elapsed,
                    "e2e_elapsed_seconds": e2e_elapsed,
                    "inference_throughput_tokens_per_second": output_tokens / inference_elapsed,
                    "e2e_throughput_tokens_per_second": output_tokens / e2e_elapsed,
                    "acceptance_rate": accepted_tokens / verified_tokens if verified_tokens else 0.0,
                    "mean_accept_tokens": sum(metric["mean_accept_tokens"] for metric in metrics) / len(metrics),
                    "selected_gamma": metrics[0]["gamma"],
                    "decode_rounds": sum(metric["round_count"] for metric in chunk_metrics),
                    "prefill_elapsed_seconds": sum(
                        metric["prefill_elapsed_seconds"] for metric in chunk_metrics
                    ),
                    "decode_elapsed_seconds": sum(
                        metric["decode_elapsed_seconds"] for metric in chunk_metrics
                    ),
                    "request_verification_rounds": [
                        metric["verification_rounds"] for metric in metrics
                    ],
                    "decode_phase_seconds": decode_phase_seconds,
                    "aclgraph_captures": max(metric["aclgraph_captures"] for metric in metrics),
                    "aclgraph_capture_attempts": max(metric["aclgraph_capture_attempts"] for metric in metrics),
                    "aclgraph_replays": max(metric["aclgraph_replays"] for metric in metrics),
                    "aclgraph_failed_captures": max(metric["aclgraph_failed_captures"] for metric in metrics),
                    "aclgraph_capacity_fallbacks": max(metric["aclgraph_capacity_fallbacks"] for metric in metrics),
                    "aclgraph_shape_fallbacks": max(metric["aclgraph_shape_fallbacks"] for metric in metrics),
                    "worker_aclgraph_metrics": engine.last_worker_metrics,
                    "measured_worker_aclgraph_deltas": _worker_aclgraph_deltas(
                        warmup_worker_metrics,
                        engine.last_worker_metrics,
                    ),
                    "worker_metrics_by_chunk": engine.last_worker_metrics_by_chunk,
                    "decode_profile": _aggregate_decode_profile(
                        engine.last_worker_metrics_by_chunk,
                        batch_size,
                    ),
                    "output_token_ids_sha256": hashlib.sha256(
                        json.dumps(output_token_rows, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "first_output_token_ids": metrics[0]["completion_token_ids"],
                }
            )

    payload = {
        "backend": "nano-pearl-native-speculative",
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "draft_tensor_parallel_size": args.draft_tp_size,
        "target_tensor_parallel_size": args.target_tp_size,
        "draft_dtype": args.draft_dtype,
        "target_dtype": args.target_dtype,
        "gamma": args.gamma,
        "seed": args.seed,
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "TASK_QUEUE_ENABLE",
                "HCCL_OP_EXPANSION_MODE",
                "HCCL_DETERMINISTIC",
            )
        },
        "target_verification_graph_buckets": args.target_verification_graph_buckets,
        "target_verification_graph_post_counts": target_graph_post_counts,
        "auto_gamma_profile_sequence_length": args.auto_gamma_profile_sequence_length,
        "profile_decode_steps": args.profile_decode_steps,
        "profile_only": args.profile_only,
        "profile_shape_warmup": args.profile_only,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "num_kvcache_blocks": args.num_kvcache_blocks,
        "max_aclgraph_entries": args.max_aclgraph_entries,
        "prefill_chunk_size": args.prefill_chunk_size,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_continuous_batching": (
            args.enable_continuous_batching or args.enable_spec_rhythm
        ),
        "enable_preemptive_scheduling": (
            args.enable_preemptive_scheduling or args.enable_spec_rhythm
        ),
        "enable_spec_rhythm": args.enable_spec_rhythm,
        "spec_rhythm_min_gamma": args.spec_rhythm_min_gamma,
        "spec_rhythm_max_eager_tokens": args.spec_rhythm_max_eager_tokens,
        "spec_rhythm_roofline": args.spec_rhythm_roofline,
        "spec_rhythm_draft_token_budget": args.spec_rhythm_draft_token_budget,
        "pad_finished_requests": args.pad_finished_requests,
        "draft_use_paged_attention": args.draft_use_paged_attention,
        "target_use_paged_attention": args.target_use_paged_attention,
        "draft_use_production_rope": args.draft_use_production_rope,
        "target_use_production_rope": args.target_use_production_rope,
        "precompile_decode_graphs": args.precompile_decode_graphs,
        "enable_cpu_binding": not args.disable_cpu_binding,
        "max_tokens": args.max_tokens,
        "request_manifest": args.request_manifest,
        "requested_output_tokens": (
            sum(request_max_tokens[:prompt_count]) if request_max_tokens is not None else None
        ),
        "num_pearl_steps": args.num_pearl_steps,
        "warmup_prompts": args.warmup_prompts,
        "warmup_prompt_offset": args.warmup_prompt_offset,
        "warmup_max_tokens": args.warmup_max_tokens,
        "warmup_runs": args.warmup_runs,
        "first_prompt_token_ids": first_prompt_token_ids,
        "results": results,
    }
    output = json.dumps(payload, ensure_ascii=True)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{output}\n", encoding="utf-8")
    print(output)


def _aggregate_decode_profile(worker_metrics_by_chunk, batch_size: int):
    full_chunks = [chunk for chunk in worker_metrics_by_chunk if chunk["batch_size"] == batch_size]
    totals = {
        "draft_compute": 0.0,
        "draft_to_target_communication": 0.0,
        "target_verify": 0.0,
        "target_to_draft_communication": 0.0,
        "wait_sync_state_update": 0.0,
    }
    component_totals = {
        "target_compute": 0.0,
        "target_verdict": 0.0,
        "wait_sync": 0.0,
        "state_update": 0.0,
    }
    profiled_steps = 0
    profiled_chunks = 0
    for chunk in full_chunks:
        workers = chunk["worker_metrics"]
        chunk_steps = max(int(worker.get("worker_profiled_decode_steps", 0)) for worker in workers)
        if chunk_steps == 0:
            continue
        draft_workers = [worker for worker in workers if worker["is_draft_rank"]]
        target_workers = [worker for worker in workers if not worker["is_draft_rank"]]
        draft_compute = max(worker["worker_profile_draft_compute_seconds"] for worker in draft_workers)
        draft_to_target = max(
            worker["worker_profile_draft_to_target_communication_seconds"] for worker in workers
        )
        target_compute = max(worker["worker_profile_target_compute_seconds"] for worker in target_workers)
        target_verdict = max(worker["worker_profile_target_verdict_seconds"] for worker in target_workers)
        target_to_draft = max(
            worker["worker_profile_target_to_draft_communication_seconds"] for worker in workers
        )
        wait_sync = max(worker["worker_profile_wait_sync_seconds"] for worker in workers)
        state_update = max(worker["worker_profile_state_update_seconds"] for worker in workers)
        totals["draft_compute"] += draft_compute
        totals["draft_to_target_communication"] += draft_to_target
        totals["target_verify"] += target_compute + target_verdict
        totals["target_to_draft_communication"] += target_to_draft
        totals["wait_sync_state_update"] += wait_sync + state_update
        component_totals["target_compute"] += target_compute
        component_totals["target_verdict"] += target_verdict
        component_totals["wait_sync"] += wait_sync
        component_totals["state_update"] += state_update
        profiled_steps += chunk_steps
        profiled_chunks += 1
    if profiled_steps == 0:
        return None
    return {
        "profiled_full_batch_chunks": profiled_chunks,
        "profiled_decode_steps": profiled_steps,
        "phase_seconds": totals,
        "phase_milliseconds_per_decode_step": {
            phase: seconds * 1000 / profiled_steps for phase, seconds in totals.items()
        },
        "component_seconds": component_totals,
        "component_milliseconds_per_decode_step": {
            phase: seconds * 1000 / profiled_steps for phase, seconds in component_totals.items()
        },
    }


def _worker_aclgraph_deltas(before_workers, after_workers):
    counter_names = (
        "aclgraph_captures",
        "aclgraph_capture_attempts",
        "aclgraph_replays",
        "aclgraph_failed_captures",
        "aclgraph_capacity_fallbacks",
        "aclgraph_shape_fallbacks",
    )
    before_by_rank = {int(worker["rank"]): worker for worker in before_workers}
    return [
        {
            "rank": int(worker["rank"]),
            **{
                f"{name}_delta": int(worker[name])
                - int(before_by_rank.get(int(worker["rank"]), {}).get(name, 0))
                for name in counter_names
            },
        }
        for worker in after_workers
    ]


if __name__ == "__main__":
    main()

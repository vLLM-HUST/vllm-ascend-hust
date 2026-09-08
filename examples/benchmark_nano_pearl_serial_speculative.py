# SPDX-License-Identifier: Apache-2.0
"""Benchmark vLLM-Ascend's in-engine serial draft-model speculation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=1)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--output-json",
        help="Also write the final benchmark payload to this path.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--gsm8k", help="Path to a GSM8K parquet file.")
    return parser


def _load_prompts(prompt: str | None, gsm8k: str | None, count: int) -> list[str]:
    if prompt is not None:
        return [prompt] * count
    import pyarrow.parquet as pq

    rows = pq.read_table(gsm8k, columns=["question"]).to_pylist()
    if len(rows) < count:
        raise ValueError(f"GSM8K contains {len(rows)} rows, but {count} were requested.")
    return [str(row["question"]) for row in rows[:count]]


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.num_prompts <= 0 or args.max_num_seqs <= 0:
        raise ValueError("Prompt count and max_num_seqs must be positive.")
    if args.draft_tp_size != args.target_tp_size:
        raise ValueError("vLLM-HUST serial draft-model speculation currently requires draft_tp_size == target_tp_size.")

    from vllm import LLM, SamplingParams, TokensPrompt

    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.spec_decode.pearl.qwen_pair import build_speculative_config

    # The ops import must precede PEARL, whose package initialization reaches
    # device_op before the normal vLLM worker import path.
    prompts = _load_prompts(args.prompt, args.gsm8k, args.num_prompts)
    speculative_config = build_speculative_config(
        args.draft_model,
        num_speculative_tokens=args.gamma,
        tensor_parallel_size=args.draft_tp_size,
    )
    llm = LLM(
        model=args.target_model,
        tensor_parallel_size=args.target_tp_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        speculative_config=speculative_config,
    )
    tokenizer = llm.get_tokenizer()
    prompt_token_ids = [
        list(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        )
        for prompt in prompts
    ]
    prompt_inputs = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    llm.generate(prompt_inputs, sampling_params, use_tqdm=False)
    started = time.perf_counter()
    outputs = llm.generate(prompt_inputs, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    output_token_rows = [list(output.outputs[0].token_ids) for output in outputs]
    payload = {
        "backend": "vllm-ascend-serial-speculative",
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "draft_tensor_parallel_size": args.draft_tp_size,
        "target_tensor_parallel_size": args.target_tp_size,
        "gamma": args.gamma,
        "batch_size": args.max_num_seqs,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
        "output_throughput_tokens_per_second": output_tokens / elapsed,
        "output_token_ids_sha256": hashlib.sha256(
            json.dumps(output_token_rows, separators=(",", ":")).encode()
        ).hexdigest(),
        "first_prompt_token_ids": prompt_token_ids[0],
        "first_output_token_ids": list(outputs[0].outputs[0].token_ids),
    }
    output = json.dumps(payload, ensure_ascii=True)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{output}\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

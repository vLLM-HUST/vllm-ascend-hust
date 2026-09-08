# SPDX-License-Identifier: Apache-2.0
"""Collect a PEARL-2 teacher trace from a Transformers causal LM.

Example::

    python examples/collect_pearl_teacher_trace.py \
      --model /data/shared-models/Qwen2.5-14B-Instruct \
      --prompt-file prompts.txt --output teacher.jsonl --max-new-tokens 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm_ascend.spec_decode.pearl.distill import (
    collect_pearl_teacher_trace,
    write_pearl_teacher_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    args = parser.parse_args()

    prompts = [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError("prompt file is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    model.to(args.device)
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    records = collect_pearl_teacher_trace(
        model,
        encoded["input_ids"].to(args.device),
        attention_mask=encoded["attention_mask"].to(args.device),
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=tokenizer.eos_token_id,
    )
    print(f"wrote {write_pearl_teacher_trace(args.output, records)} records to {args.output}")


if __name__ == "__main__":
    main()

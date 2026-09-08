# SPDX-License-Identifier: Apache-2.0
"""Build a deterministic ShareGPT request manifest for nano-PEARL benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--verification-margin",
        type=int,
        default=8,
        help="Reserve this many tokens for the largest gamma under test.",
    )
    args = parser.parse_args()
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(rows)

    selected = []
    for source_index, row in enumerate(rows):
        conversations = row.get("conversations", [])
        if len(conversations) < 2:
            continue
        prompt = str(conversations[0].get("value", ""))
        completion = str(conversations[1].get("value", ""))
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer.encode(formatted_prompt))
        output_tokens = len(tokenizer.encode(completion))
        if prompt_tokens < 4 or output_tokens < 4:
            continue
        if prompt_tokens > args.max_prompt_tokens:
            continue
        if prompt_tokens + output_tokens + args.verification_margin > args.max_model_len:
            continue
        selected.append(
            {
                "source_index_after_shuffle": source_index,
                "source_id": row.get("id"),
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "max_tokens": output_tokens,
                "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
            }
        )
        if len(selected) == args.num_prompts:
            break

    if len(selected) != args.num_prompts:
        raise RuntimeError(
            f"Only {len(selected)} valid ShareGPT rows found; {args.num_prompts} requested."
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "num_prompts": len(selected),
                "prompt_tokens": sum(row["prompt_tokens"] for row in selected),
                "output_tokens": sum(row["max_tokens"] for row in selected),
                "min_output_tokens": min(row["max_tokens"] for row in selected),
                "max_output_tokens": max(row["max_tokens"] for row in selected),
            }
        )
    )


if __name__ == "__main__":
    main()

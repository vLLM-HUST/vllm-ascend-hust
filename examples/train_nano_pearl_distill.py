#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train a PEARL-2 draft from a JSONL target-worker teacher trace.

The trace format is documented by ``load_pearl_distillation_records``. This
small entrypoint intentionally leaves teacher rollout to the caller, so it can
be used with either Transformers models or a native PEARL target worker.
"""

from __future__ import annotations

import argparse

import torch

from vllm_ascend.spec_decode.pearl.distill import (
    PearlDistillationConfig,
    collate_pearl_distillation_records,
    load_pearl_distillation_records,
    save_pearl_distillation_checkpoint,
    train_pearl_distillation_step,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--trace", required=True, help="JSONL target teacher trace")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--acceptance-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.student_model, torch_dtype=torch.bfloat16)
    model.to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    config = PearlDistillationConfig(
        temperature=args.temperature,
        distill_weight=args.distill_weight,
        acceptance_weight=args.acceptance_weight,
    )
    records = load_pearl_distillation_records(args.trace)
    metrics = None
    for _ in range(args.epochs):
        # A trace is already tokenized and target logits are precomputed. One
        # collated batch keeps this example deterministic and easy to profile;
        # callers with larger traces can shard records before invoking it.
        batch = collate_pearl_distillation_records(records, device=args.device)
        metrics = train_pearl_distillation_step(model, optimizer, **batch, config=config)
        print(metrics)
    save_pearl_distillation_checkpoint(args.output, model, optimizer, step=args.epochs, config=config)
    assert metrics is not None


if __name__ == "__main__":
    main()

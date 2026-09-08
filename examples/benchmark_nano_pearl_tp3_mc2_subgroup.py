# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the TP3 MC2 custom op in PEARL's nonzero-rank subgroup."""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch_npu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comm-rank", choices=("global", "local"), required=True)
    parser.add_argument("--prewarm-all-reduce", action="store_true")
    parser.add_argument("--queued-all-reduces", type=int, default=0)
    parser.add_argument("--pearl-groups", action="store_true")
    parser.add_argument("--token-count", type=int, default=512)
    args = parser.parse_args()

    local_device = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_device)
    dist.init_process_group(backend="hccl")
    global_rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise ValueError("The subgroup smoke test requires four processes.")

    target_ranks = (1, 2, 3)
    if args.pearl_groups:
        dist.new_group(ranks=[0], backend="hccl")
    target_group = dist.new_group(ranks=list(target_ranks), backend="hccl")
    if args.pearl_groups:
        dist.new_group(ranks=[0, 1, 2, 3], backend="hccl")
        dist.new_group(ranks=[0, 1], backend="hccl")
    if global_rank in target_ranks:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("Could not register the vLLM-Ascend custom ops.")
        local_rank = dist.get_rank(target_group)
        backend = target_group._get_backend(torch.device("npu"))
        comm_rank = global_rank if args.comm_rank == "global" else local_rank
        hcomm_info = backend.get_hccl_comm_name(comm_rank)

        source = torch.zeros(
            (args.token_count, 4608),
            dtype=torch.bfloat16,
            device="npu",
        )
        weight = torch_npu.npu_format_cast(
            torch.zeros((5120, 4608), dtype=torch.bfloat16, device="npu"),
            29,
        )
        residual = torch.zeros(
            (args.token_count, 5120),
            dtype=torch.bfloat16,
            device="npu",
        )
        gamma = torch.ones(5120, dtype=torch.bfloat16, device="npu")
        if args.prewarm_all_reduce:
            warmup = torch.zeros(1, dtype=torch.float32, device="npu")
            dist.all_reduce(warmup, group=target_group)
            torch.npu.synchronize()
        queued = torch.zeros(
            (args.token_count, 5120),
            dtype=torch.bfloat16,
            device="npu",
        )
        for _ in range(args.queued_all_reduces):
            dist.all_reduce(queued, group=target_group)

        dist.barrier(group=target_group)
        started = time.perf_counter()
        output, _ = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(
            source,
            weight,
            residual,
            gamma,
            hcomm_info,
            len(target_ranks),
            local_rank,
            1e-6,
            True,
            False,
        )
        output.cpu()
        elapsed = time.perf_counter() - started
        print(
            f"rank={global_rank} local_rank={local_rank} "
            f"comm_rank={comm_rank} elapsed={elapsed:.6f}s",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

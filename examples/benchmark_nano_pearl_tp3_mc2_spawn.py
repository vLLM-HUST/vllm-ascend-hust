# SPDX-License-Identifier: Apache-2.0
"""Reproduce the PEARL spawn launcher around a nonzero-rank TP3 MC2 group."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import socket
import time
from multiprocessing.connection import Connection
from multiprocessing.connection import wait


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _worker(
    rank: int,
    master_port: int,
    bind_before_init: bool,
    connection: Connection,
) -> None:
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(master_port),
        "RANK": str(rank),
        "WORLD_SIZE": "4",
        "LOCAL_RANK": str(rank),
    })
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")

    import torch
    import torch.distributed as dist
    import torch_npu

    try:
        connection.send(("stage", rank, "imported"))
        if bind_before_init:
            torch.npu.set_device(rank)
            connection.send(("stage", rank, "device-bound"))
        dist.init_process_group(backend="hccl")
        connection.send(("stage", rank, "world-initialized"))
        if not bind_before_init:
            torch.npu.set_device(rank)
            connection.send(("stage", rank, "device-bound"))

        dist.new_group(ranks=[0], backend="hccl")
        target_group = dist.new_group(ranks=[1, 2, 3], backend="hccl")
        dist.new_group(ranks=[0, 1, 2, 3], backend="hccl")
        dist.new_group(ranks=[0, 1], backend="hccl")
        connection.send(("stage", rank, "groups-created"))
        if rank in (1, 2, 3):
            from vllm_ascend.utils import enable_custom_op

            if not enable_custom_op():
                raise RuntimeError("Could not register vLLM-Ascend custom operators.")
            target_rank = dist.get_rank(target_group)
            backend = target_group._get_backend(torch.device("npu"))
            hcomm_info = backend.get_hccl_comm_name(rank)
            source = torch.zeros((128, 4608), dtype=torch.bfloat16, device="npu")
            weight = torch_npu.npu_format_cast(
                torch.zeros((5120, 4608), dtype=torch.bfloat16, device="npu"),
                29,
            )
            residual = torch.zeros((128, 5120), dtype=torch.bfloat16, device="npu")
            gamma = torch.ones(5120, dtype=torch.bfloat16, device="npu")
            output, _ = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(
                source,
                weight,
                residual,
                gamma,
                hcomm_info,
                3,
                target_rank,
                1e-6,
                True,
                False,
            )
            output.cpu()
            connection.send(("stage", rank, "mc2-complete"))
        connection.send(("ok", rank))
        if rank == 0:
            connection.recv()
    except Exception as error:
        connection.send(("error", rank, f"{type(error).__name__}: {error}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-before-init", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    context = mp.get_context("spawn")
    master_port = _reserve_local_port()
    processes: list[mp.Process] = []
    connections: list[Connection] = []
    for rank in range(4):
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_worker,
            args=(rank, master_port, args.bind_before_init, child),
        )
        process.start()
        child.close()
        processes.append(process)
        connections.append(parent)

    replies: list[tuple[object, ...]] = []
    pending = set(connections)
    deadline = time.monotonic() + 60
    while pending and time.monotonic() < deadline:
        ready = wait(pending, timeout=max(0.0, deadline - time.monotonic()))
        if not ready:
            break
        for connection in ready:
            while connection.poll():
                reply = connection.recv()
                print(reply, flush=True)
                if reply[0] in {"ok", "error"}:
                    replies.append(reply)
                    pending.remove(connection)
                    break
    connections[0].send("release")
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    if pending or len(replies) != 4 or any(reply[0] != "ok" for reply in replies):
        raise RuntimeError("At least one spawn worker failed.")


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HCCL binding for the vLLM-HUST communication-provider contract."""

from typing import Any

import torch

try:
    from vllm.distributed.communication_provider import (  # type: ignore[import-not-found]
        CommunicationProviderBase,
    )
except ModuleNotFoundError:
    # vLLM Ascend CI also checks against upstream vLLM, where the HUST host
    # contract is intentionally absent. Keep the adapter importable there; the
    # upstream coordinator never resolves this provider class.
    class CommunicationProviderBase:  # type: ignore[no-redef]
        def __init__(self, device_communicator: Any, group_name: str) -> None:
            self.device_communicator = device_communicator
            self.group_name = group_name


class HCCLCommunicationProvider(CommunicationProviderBase):
    """Expose the existing NPU communicator through the shared host contract.

    This adapter deliberately reuses vLLM's initialized HCCL process group; it
    does not create a second group or duplicate communicator lifecycle state.
    """

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return self.device_communicator.all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        return self.device_communicator.all_gather(input_, dim)

    def reduce_scatter(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        return self.device_communicator.reduce_scatter(input_, dim)

    def broadcast(self, input_: torch.Tensor, src: int) -> torch.Tensor:
        return self.device_communicator.broadcast(input_, src)

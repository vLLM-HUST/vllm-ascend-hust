# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch
from vllm_ascend.distributed.device_communicators.npu_communicator import (
    NPUCommunicator,
)


def test_all_reduce_returns_non_aliased_output() -> None:
    communicator = NPUCommunicator.__new__(NPUCommunicator)
    communicator.device_group = object()
    input_ = torch.tensor([1.0, 2.0])

    with patch(
        "vllm_ascend.distributed.device_communicators.npu_communicator.dist.all_reduce"
    ) as all_reduce:
        output = communicator.all_reduce(input_)

    assert output.data_ptr() != input_.data_ptr()
    torch.testing.assert_close(output, input_)
    all_reduce.assert_called_once_with(output, group=communicator.device_group)

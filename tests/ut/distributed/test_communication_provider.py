# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import torch

from vllm_ascend.distributed.communication_provider import HCCLCommunicationProvider


def test_hccl_provider_reuses_native_communicator() -> None:
    communicator = Mock()
    provider = HCCLCommunicationProvider(communicator, "tp")
    input_ = torch.tensor([1])
    communicator.all_reduce.return_value = input_
    communicator.all_gather.return_value = input_
    communicator.reduce_scatter.return_value = input_
    communicator.broadcast.return_value = input_

    assert provider.all_reduce(input_) is input_
    assert provider.all_gather(input_, 0) is input_
    assert provider.reduce_scatter(input_, 0) is input_
    assert provider.broadcast(input_, 0) is input_

    communicator.all_reduce.assert_called_once_with(input_)
    communicator.all_gather.assert_called_once_with(input_, 0)
    communicator.reduce_scatter.assert_called_once_with(input_, 0)
    communicator.broadcast.assert_called_once_with(input_, 0)

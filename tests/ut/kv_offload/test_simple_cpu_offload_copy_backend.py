"""验证 NPU 简单 CPU Offload 拷贝后端的事件依赖。

输入：模拟的 NPU 流、事件和拷贝参数。
输出：确认拷贝任务携带事件，并在拷贝前等待该事件。
"""

import queue
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.simple_kv_offload.copy_backend import NPUDmaCopyBackend


def test_launch_copy_queues_wait_event() -> None:
    backend = NPUDmaCopyBackend.__new__(NPUDmaCopyBackend)
    backend._store_params = object()
    backend._queue = queue.SimpleQueue()
    wait_event = object()
    events_list: list[tuple[int, torch.npu.Event]] = []

    backend.launch_copy([1], [2], True, 3, events_list, wait_event)

    assert backend._queue.get() == (
        [1],
        [2],
        backend._store_params,
        True,
        3,
        events_list,
        wait_event,
    )


def test_store_waits_for_compute_event_before_copy() -> None:
    backend = NPUDmaCopyBackend.__new__(NPUDmaCopyBackend)
    backend._device = object()
    backend._load_stream = MagicMock()
    backend._store_stream = MagicMock()
    backend._queue = queue.SimpleQueue()
    events_list: list[tuple[int, torch.npu.Event]] = []
    backend._queue.put(([1], [2], object(), True, 3, events_list, wait_event := MagicMock()))
    backend._queue.put(None)

    order = []
    completion_event = MagicMock()
    backend._store_stream.wait_event.side_effect = lambda event: order.append(("wait", event))

    with (
        patch("vllm_ascend.simple_kv_offload.copy_backend.torch.npu.set_device"),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.copy_blocks",
            side_effect=lambda *_args: order.append(("copy", None)),
        ),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.torch.npu.Event",
            return_value=completion_event,
        ),
    ):
        backend._copy_loop()

    assert order == [("wait", wait_event), ("copy", None)]
    completion_event.record.assert_called_once_with(backend._store_stream)
    assert events_list == [(3, completion_event)]

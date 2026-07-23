from collections.abc import Iterator
from typing import Any

import torch
from typing_extensions import override
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec as _CPUOffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from vllm_ascend.kv_offload.cpu_npu import CpuNpuOffloadingHandlers

_TIERING_IMPORT_ERROR: ModuleNotFoundError | None
try:
    from vllm.v1.kv_offload.tiering.spec import (
        TieringOffloadingSpec as _TieringOffloadingSpec,
    )
except ModuleNotFoundError as exc:
    _TieringOffloadingSpec = None
    _TIERING_IMPORT_ERROR = exc
else:
    _TIERING_IMPORT_ERROR = None


def _set_cpu_bytes_from_legacy_num_blocks(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    """Translate Ascend's legacy num_cpu_blocks knob to cpu_bytes_to_use."""
    extra_config: dict[str, Any] = vllm_config.kv_transfer_config.kv_connector_extra_config  # type: ignore[union-attr]
    if extra_config.get("cpu_bytes_to_use") is not None:
        return

    num_cpu_blocks = extra_config.get("num_cpu_blocks")
    if num_cpu_blocks is None:
        return
    if kv_cache_config.num_blocks <= 0:
        extra_config["cpu_bytes_to_use"] = 1
        return

    gpu_block_sizes = {kv_cache_group.kv_cache_spec.block_size for kv_cache_group in kv_cache_config.kv_cache_groups}
    block_size_factor = 1
    offloaded_block_size = extra_config.get("block_size")
    if offloaded_block_size is not None:
        assert len(gpu_block_sizes) == 1, (
            "If 'block_size' is specified in kv_connector_extra_config, "
            "all KV cache groups must have the same block size."
        )
        gpu_block_size = next(iter(gpu_block_sizes))
        offloaded_block_size = int(offloaded_block_size)
        assert offloaded_block_size % gpu_block_size == 0
        block_size_factor = offloaded_block_size // gpu_block_size

    world_size = vllm_config.parallel_config.world_size
    total_gpu_kv_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
    kv_bytes_per_block = (total_gpu_kv_bytes // kv_cache_config.num_blocks) * world_size
    kv_bytes_per_offloaded_block = kv_bytes_per_block * block_size_factor
    extra_config["cpu_bytes_to_use"] = int(num_cpu_blocks) * kv_bytes_per_offloaded_block


class _NPUHandlersMixin:
    _handlers: CpuNpuOffloadingHandlers | None

    def create_handlers(self, kv_caches: CanonicalKVCaches) -> CpuNpuOffloadingHandlers:
        raise NotImplementedError

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            self._handlers = self.create_handlers(kv_caches)

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.npu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_npu_handler


class NPUOffloadingSpec(_NPUHandlersMixin, _CPUOffloadingSpec):
    """Ascend NPU implementation of vLLM's CPU KV offloading spec."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        _set_cpu_bytes_from_legacy_num_blocks(vllm_config, kv_cache_config)
        super().__init__(vllm_config, kv_cache_config)

    @override
    def create_handlers(self, kv_caches: CanonicalKVCaches) -> CpuNpuOffloadingHandlers:
        return CpuNpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
        )


if _TieringOffloadingSpec is not None:

    class NPUTieringOffloadingSpec(_NPUHandlersMixin, _TieringOffloadingSpec):
        """Ascend NPU implementation of vLLM's multi-tier KV offloading spec."""

        def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
            _set_cpu_bytes_from_legacy_num_blocks(vllm_config, kv_cache_config)
            super().__init__(vllm_config, kv_cache_config)

        @override
        def create_handlers(self, kv_caches: CanonicalKVCaches) -> CpuNpuOffloadingHandlers:
            rank = torch.npu.current_device()
            worker_mmap = SharedOffloadRegion(
                instance_id=self.vllm_config.instance_id,
                total_size_bytes=self.total_size_bytes,
                num_blocks=self.num_blocks,
                rank=rank,
                num_workers=self.vllm_config.parallel_config.world_size,
                cpu_page_size=self.cpu_page_size_per_worker,
            )
            return CpuNpuOffloadingHandlers(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
                mmap_region=worker_mmap,
            )
else:

    class NPUTieringOffloadingSpec:  # type: ignore[no-redef]
        """Placeholder for vLLM builds that do not yet expose tiering APIs."""

        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("NPUTieringOffloadingSpec requires vllm.v1.kv_offload.tiering") from _TIERING_IMPORT_ERROR


CPUOffloadingSpec = NPUOffloadingSpec
TieringOffloadingSpec = NPUTieringOffloadingSpec

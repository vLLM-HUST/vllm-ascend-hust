from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.patch.worker.patch_mamba_utils import _get_mamba_groups


def _mamba_spec(block_size: int) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((1, 8),),
        dtypes=(torch.bfloat16,),
    )


def test_get_mamba_groups_preserves_current_core_mapping_contract() -> None:
    first = _mamba_spec(16)
    second = _mamba_spec(32)
    wrapped = UniformTypeKVCacheSpecs.from_specs({"layer.1": first, "layer.2": second})
    assert wrapped is not None
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer.0"], first),
            KVCacheGroupSpec(["layer.1", "layer.2"], wrapped),
            SimpleNamespace(kv_cache_spec=object()),
        ],
    )

    groups = _get_mamba_groups(config)

    assert groups == {first: [0, 1], second: [1]}
    assert all(isinstance(spec, MambaSpec) for spec in groups)

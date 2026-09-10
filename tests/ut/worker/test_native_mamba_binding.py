# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "native_binding", Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/patch_bind_kv_cache.py"
)
assert spec is not None, "unable to create native Mamba binding module spec"
assert spec.loader is not None, "native Mamba binding module spec has no loader"
binding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(binding)


def layer():
    return SimpleNamespace(get_state_shape=lambda: ((4,), (8,)), get_state_dtype=lambda: (torch.float16, torch.float32))


def test_native_state_views_bind_without_reinterpretation():
    target = layer()
    states = (torch.zeros(3, 4, dtype=torch.float16), torch.zeros(3, 8, dtype=torch.float32))
    binding.bind_mamba_cache(target, states)
    assert target.kv_cache is states


@pytest.mark.parametrize("invalid", ["shape", "dtype", "blocks", "count"])
def test_invalid_native_states_are_rejected(invalid):
    states = [torch.zeros(3, 4, dtype=torch.float16), torch.zeros(3, 8, dtype=torch.float32)]
    if invalid == "shape":
        states[0] = torch.zeros(3, 5, dtype=torch.float16)
    elif invalid == "dtype":
        states[0] = states[0].float()
    elif invalid == "blocks":
        states[0] = states[0][:2]
    else:
        states.pop()
    with pytest.raises(ValueError):
        binding.bind_mamba_cache(layer(), tuple(states))


def test_core_raw_page_binding_is_preserved():
    target = layer()
    pages = torch.zeros(3, 1, 1, 40, dtype=torch.int8)
    binding.bind_mamba_cache(target, pages)
    assert target.kv_cache[0].shape == (3, 4)
    assert target.kv_cache[1].shape == (3, 8)

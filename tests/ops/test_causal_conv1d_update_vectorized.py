"""CPU equivalence gate for the vectorized causal_conv1d_update decode fast path.

Loads the module by file path with the single external symbol stubbed, so the
suite runs in the CPU gate environment without torch_npu or package __init__.
"""

import importlib.util
import inspect
import sys
import types

import pytest
import torch
import torch.nn.functional as F

_stub = types.ModuleType("vllm.v1.attention.backends.utils")
_stub.PAD_SLOT_ID = -1
for _name in ("vllm", "vllm.v1", "vllm.v1.attention"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["vllm.v1.attention.backends.utils"] = _stub

_spec = importlib.util.spec_from_file_location(
    "causal_conv1d_cpu",
    __file__.replace("test_causal_conv1d_update_vectorized.py", "") + "../../vllm_ascend/ops/causal_conv1d.py",
)
cc1d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc1d)
causal_conv1d_update = cc1d.causal_conv1d_update

PAD = -1


def _make(batch, slots, dim, width, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, dim, generator=g, dtype=dtype)
    conv_state = torch.randn(slots, dim, width - 1, generator=g, dtype=dtype)
    weight = torch.randn(dim, width, generator=g, dtype=dtype)
    bias = torch.randn(dim, generator=g, dtype=dtype)
    return x, conv_state, weight, bias


@pytest.mark.parametrize("activation", ["silu", "swish", None])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_tensor_layout_matches_legacy(activation, use_bias, seed):
    x, conv_state, weight, bias = _make(8, 32, 64, 4, seed)
    bias = bias if use_bias else None

    ref_state = conv_state.clone()
    ref_out = causal_conv1d_update(x.clone(), ref_state, weight, bias, activation=activation)
    fast_state = conv_state.clone()
    fast_out = causal_conv1d_update(x.clone(), fast_state, weight, bias, activation=activation)

    torch.testing.assert_close(ref_out, fast_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ref_state, fast_state, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("pad_ratio", [0.0, 0.5])
@pytest.mark.parametrize("seed", [2, 3])
def test_index_layout_matches_legacy(pad_ratio, seed):
    batch, slots, dim, width = 6, 16, 48, 4
    x, conv_state, weight, bias = _make(batch, slots, dim, width, seed)
    num_pad = int(batch * pad_ratio)
    indices = torch.tensor([PAD] * num_pad + list(range(batch - num_pad)), dtype=torch.int32)

    ref_state = conv_state.clone()
    ref_out = causal_conv1d_update(
        x.clone(),
        ref_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=indices,
        pad_slot_id=PAD,
    )
    fast_state = conv_state.clone()
    fast_out = causal_conv1d_update(
        x.clone(),
        fast_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=indices,
        pad_slot_id=PAD,
    )

    torch.testing.assert_close(ref_out, fast_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ref_state, fast_state, rtol=1e-5, atol=1e-5)


def test_query_start_loc_layout_matches_legacy():
    batch, slots, dim, width = 4, 8, 32, 4
    x, conv_state, weight, bias = _make(batch, slots, dim, width, 4)
    qsl = torch.arange(batch + 1, dtype=torch.int32)
    csi = torch.arange(batch, dtype=torch.int32)

    ref_state = conv_state.clone()
    ref_out = causal_conv1d_update(
        x.clone(),
        ref_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=csi,
        query_start_loc=qsl,
    )
    fast_state = conv_state.clone()
    fast_out = causal_conv1d_update(
        x.clone(),
        fast_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=csi,
        query_start_loc=qsl,
    )

    torch.testing.assert_close(ref_out, fast_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ref_state, fast_state, rtol=1e-5, atol=1e-5)


def test_window_math_matches_grouped_conv_ground_truth():
    batch, dim, width = 4, 16, 4
    x, conv_state, weight, bias = _make(batch, batch, dim, width, 5)

    fast_state = conv_state.clone()
    fast_out = causal_conv1d_update(x.clone(), fast_state, weight, bias, activation="silu")

    cat_x = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)
    ref_out = F.silu(F.conv1d(cat_x, weight.unsqueeze(1), bias, padding=0, groups=dim).squeeze(-1))
    torch.testing.assert_close(ref_out, fast_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cat_x[..., 1:], fast_state, rtol=1e-5, atol=1e-5)


def test_fast_path_source_has_no_host_sync():
    source = inspect.getsource(causal_conv1d_update)
    fast_block = source[source.index("is_single_token_decode") : source.index("def _select_state")]
    for banned in (".item()", "torch.cuda.synchronize", "cpu()"):
        assert banned not in fast_block, f"host-sync pattern in fast path: {banned}"

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.attention.continuing_prefill import (
    C8ContinuingPrefillProviderConfig,
    C8ContinuingPrefillRequest,
    C8ContinuingPrefillResult,
    load_c8_continuing_prefill_provider,
)


class RecordingProvider:
    def __init__(self, result: object = C8ContinuingPrefillResult(), eligible: object = True):
        self.result = result
        self.eligible = eligible
        self.requests: list[C8ContinuingPrefillRequest] = []
        self.eligibility_requests: list[C8ContinuingPrefillRequest] = []

    def is_eligible(self, request: C8ContinuingPrefillRequest):
        self.eligibility_requests.append(request)
        return self.eligible

    def forward(self, request: C8ContinuingPrefillRequest):
        self.requests.append(request)
        if isinstance(self.result, C8ContinuingPrefillResult):
            request.output.fill_(7)
        return self.result


def make_provider_config() -> C8ContinuingPrefillProviderConfig:
    return C8ContinuingPrefillProviderConfig(
        layer_name="model.layers.0.self_attn.attn",
        model="Qwen/Qwen3.5-35B-A3B",
        model_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        tensor_parallel_rank=0,
        tensor_parallel_size=2,
        num_heads=2,
        num_kv_heads=1,
        head_size=32,
        scale=0.125,
        kv_cache_dtype=torch.int8,
        provider_config_json='{"profile_sha256":"abc"}',
    )


def make_impl(provider=None):
    from vllm_ascend.attention.attention_v1 import AscendC8AttentionBackendImpl

    impl = object.__new__(AscendC8AttentionBackendImpl)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 32
    impl.scale = 0.125
    impl.key_cache = torch.zeros((2, 32, 1, 32), dtype=torch.int8)
    impl.value_cache = torch.zeros((2, 32, 1, 32), dtype=torch.int8)
    impl._c8_continuing_prefill_provider = provider
    impl._c8_continuing_prefill_eager_workspace = ()
    impl._c8_continuing_prefill_graph_workspaces = []
    return impl


def make_layer():
    return SimpleNamespace(
        _c8_k_aq_scale_nz_bnsd=torch.ones((1, 1, 32)),
        _c8_v_aq_scale_nz_bnsd=torch.ones((1, 1, 32)),
        _c8_k_offset=torch.full((1, 1, 32), 2.0),
        _c8_v_offset=torch.full((1, 1, 32), -3.0),
    )


def make_metadata():
    return SimpleNamespace(
        num_decode_tokens=0,
        num_decodes=0,
        num_prefills=1,
        actual_seq_lengths_q=[2],
        seq_lens_list=[34],
        block_tables=torch.tensor([[0, 1]]),
        attn_mask=None,
        causal=True,
    )


def build_request(impl, output=None):
    output = torch.zeros((2, 2, 32)) if output is None else output
    return impl._build_c8_continuing_prefill_request(torch.zeros((2, 2, 32)), make_metadata(), output, make_layer())


def test_provider_config_is_default_off_and_accepts_factory_path():
    default = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    configured = AscendConfig(
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
        c8_continuing_prefill_provider="example.provider:create",
        c8_continuing_prefill_provider_config={"profile_sha256": "abc"},
    )

    assert default.c8_continuing_prefill_provider is None
    assert default.c8_continuing_prefill_provider_config == {}
    assert configured.c8_continuing_prefill_provider == "example.provider:create"
    assert configured.c8_continuing_prefill_provider_config == {"profile_sha256": "abc"}


def test_provider_config_rejects_unbound_or_non_json_values():
    with pytest.raises(ValueError, match="requires c8_continuing_prefill_provider"):
        AscendConfig(
            sparse_kv_offload_config=SimpleNamespace(enabled=False),
            c8_continuing_prefill_provider_config={"profile_sha256": "abc"},
        )

    with pytest.raises(ValueError, match="finite JSON values"):
        AscendConfig(
            sparse_kv_offload_config=SimpleNamespace(enabled=False),
            c8_continuing_prefill_provider="example.provider:create",
            c8_continuing_prefill_provider_config={"threshold": float("nan")},
        )


def test_configure_provider_passes_runtime_identity_and_canonical_config():
    impl = make_impl()
    impl.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="Qwen/Qwen3.5-35B-A3B",
            revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        )
    )
    provider = RecordingProvider()
    config = SimpleNamespace(
        c8_continuing_prefill_provider="example.provider:create",
        c8_continuing_prefill_provider_config={
            "zero_offsets_attested": True,
            "profile_sha256": "abc",
        },
    )

    with (
        patch("vllm_ascend.attention.attention_v1.get_ascend_config", return_value=config),
        patch(
            "vllm_ascend.attention.attention_v1.get_tensor_model_parallel_rank",
            return_value=1,
        ),
        patch(
            "vllm_ascend.attention.attention_v1.get_tensor_model_parallel_world_size",
            return_value=2,
        ),
        patch(
            "vllm_ascend.attention.attention_v1.load_c8_continuing_prefill_provider",
            return_value=provider,
        ) as load_provider,
    ):
        impl.configure_c8_continuing_prefill_provider("model.layers.3.self_attn.attn")

    factory_path, factory_config = load_provider.call_args.args
    assert factory_path == "example.provider:create"
    assert factory_config.layer_name == "model.layers.3.self_attn.attn"
    assert factory_config.model == "Qwen/Qwen3.5-35B-A3B"
    assert factory_config.model_revision == "59d61f3ce65a6d9863b86d2e96597125219dc754"
    assert factory_config.tensor_parallel_rank == 1
    assert factory_config.tensor_parallel_size == 2
    assert factory_config.provider_config_json == ('{"profile_sha256":"abc","zero_offsets_attested":true}')
    assert impl._c8_continuing_prefill_provider is provider


def test_load_provider_from_explicit_factory(monkeypatch):
    provider = RecordingProvider()
    factory = MagicMock(return_value=provider)
    monkeypatch.setitem(sys.modules, "test_c8_provider", SimpleNamespace(create=factory))

    loaded = load_c8_continuing_prefill_provider("test_c8_provider:create", make_provider_config())

    assert loaded is provider
    factory.assert_called_once_with(make_provider_config())


@pytest.mark.parametrize("factory_path", ["", "module", ":factory", "module:", "module:factory:extra"])
def test_load_provider_rejects_invalid_factory_path(factory_path):
    with pytest.raises(ValueError, match="module:factory"):
        load_c8_continuing_prefill_provider(factory_path, make_provider_config())


def test_load_provider_rejects_object_without_contract(monkeypatch):
    monkeypatch.setitem(sys.modules, "test_invalid_c8_provider", SimpleNamespace(create=lambda config: object()))

    with pytest.raises(TypeError, match="does not implement"):
        load_c8_continuing_prefill_provider("test_invalid_c8_provider:create", make_provider_config())


@pytest.mark.parametrize(
    ("num_decodes", "num_prefills", "actual_q", "seq_lens", "causal", "expected"),
    [
        (1, 0, [1], [32], True, False),
        (0, 1, [4], [4], True, False),
        (0, 1, [1], [33], True, False),
        (0, 1, [4], [36], False, False),
        (0, 1, [4], [36], True, True),
        (1, 1, [1, 5], [32, 36], True, True),
    ],
)
def test_host_classifies_only_cached_multi_token_prefill(
    num_decodes, num_prefills, actual_q, seq_lens, causal, expected
):
    impl = make_impl()
    metadata = SimpleNamespace(
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        actual_seq_lengths_q=actual_q,
        seq_lens_list=seq_lens,
        causal=causal,
    )

    assert impl._has_cached_multi_token_prefill(metadata) is expected


def test_provider_receives_paged_int8_contract_and_owns_output():
    workspace = (torch.ones(1),)
    provider = RecordingProvider(C8ContinuingPrefillResult(workspace=workspace))
    impl = make_impl(provider)
    output = torch.zeros((2, 2, 32))

    with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(capturing=False)):
        handled = impl._try_c8_continuing_prefill_provider(build_request(impl, output))

    assert handled is True
    assert torch.count_nonzero(output != 7) == 0
    request = provider.requests[0]
    assert request.key_cache.dtype == torch.int8
    assert request.key_cache.shape == (2, 1, 1, 32, 32)
    assert request.block_table.tolist() == [[0, 1]]
    assert request.actual_seq_lengths_q == (2,)
    assert request.actual_seq_lengths_kv == (34,)
    assert torch.all(request.key_antiquant_offset == 2)
    assert torch.all(request.value_antiquant_offset == -3)
    assert request.output.data_ptr() == output.data_ptr()
    assert impl._c8_continuing_prefill_eager_workspace == workspace


def test_mixed_batch_request_slices_decode_rows_from_all_prefill_metadata():
    impl = make_impl()
    impl.key_cache = torch.zeros((8, 32, 1, 32), dtype=torch.int8)
    impl.value_cache = torch.zeros((8, 32, 1, 32), dtype=torch.int8)
    metadata = SimpleNamespace(
        num_decode_tokens=2,
        num_decodes=2,
        num_prefills=2,
        actual_seq_lengths_q=[1, 2, 5, 9],
        seq_lens_list=[33, 65, 35, 68],
        block_tables=torch.tensor(
            [
                [0, 1, -1],
                [2, 3, -1],
                [4, 5, -1],
                [6, 7, 0],
            ]
        ),
        attn_mask=None,
        causal=True,
    )
    query = torch.arange(9 * 2 * 32, dtype=torch.float32).reshape(9, 2, 32)
    output = torch.zeros_like(query)

    request = impl._build_c8_continuing_prefill_request(query, metadata, output, make_layer())

    assert request.query.shape == (7, 2, 32)
    assert request.query.data_ptr() == query[2:].data_ptr()
    assert request.output.shape == (7, 2, 32)
    assert request.output.data_ptr() == output[2:].data_ptr()
    assert request.actual_seq_lengths_q == (3, 7)
    assert request.actual_seq_lengths_kv == (35, 68)
    assert request.block_table.tolist() == [[4, 5, -1], [6, 7, 0]]


def test_provider_ineligible_uses_host_fallback_without_execution():
    provider = RecordingProvider(eligible=False)
    impl = make_impl(provider)

    with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(capturing=False)):
        handled = impl._try_c8_continuing_prefill_provider(build_request(impl))

    assert handled is False
    assert len(provider.eligibility_requests) == 1
    assert provider.requests == []


def test_provider_invalid_eligibility_fails_closed():
    impl = make_impl(RecordingProvider(eligible=1))

    with (
        patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(capturing=False)),
        pytest.raises(TypeError, match="must return bool"),
    ):
        impl._try_c8_continuing_prefill_provider(build_request(impl))


def test_provider_invalid_result_fails_closed():
    impl = make_impl(RecordingProvider(object()))

    with (
        patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(capturing=False)),
        pytest.raises(TypeError, match="must return C8ContinuingPrefillResult"),
    ):
        impl._try_c8_continuing_prefill_provider(build_request(impl))


def test_capture_workspace_is_retained_for_graph_lifetime():
    workspace = (torch.ones(4),)
    impl = make_impl(RecordingProvider(C8ContinuingPrefillResult(workspace=workspace)))

    with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(capturing=True)):
        impl._try_c8_continuing_prefill_provider(build_request(impl))

    assert impl._c8_continuing_prefill_graph_workspaces == [workspace]


def test_cached_prefill_provider_bypasses_dense_materialization():
    provider = RecordingProvider()
    impl = make_impl(provider)
    metadata = make_metadata()
    impl._dequant_paged_kv_to_dense = MagicMock(side_effect=AssertionError("dense fallback must not run"))
    output = torch.zeros((2, 2, 32))

    with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(capturing=False)):
        result = impl._forward_c8_chunked_prefill(
            query=torch.zeros((2, 2, 32)),
            float_key=None,
            float_value=None,
            attn_metadata=metadata,
            output=output,
            layer=make_layer(),
        )

    assert result is output
    assert len(provider.requests) == 1
    impl._dequant_paged_kv_to_dense.assert_not_called()


@pytest.mark.parametrize(
    ("provider_enabled", "query_length", "kv_length"),
    [
        (False, 2, 34),
        (True, 2, 2),
        (True, 1, 33),
    ],
)
def test_chunked_fallback_does_not_offer_ineligible_work_to_provider(provider_enabled, query_length, kv_length):
    provider = RecordingProvider() if provider_enabled else None
    impl = make_impl(provider)
    impl._build_c8_continuing_prefill_request = MagicMock(side_effect=AssertionError("unexpected provider request"))
    impl._dequant_paged_kv_to_dense = MagicMock(
        return_value=(torch.zeros((kv_length, 1, 32)), torch.zeros((kv_length, 1, 32)))
    )
    metadata = make_metadata()
    metadata.actual_seq_lengths_q = [query_length]
    metadata.seq_lens_list = [kv_length]
    query = torch.zeros((query_length, 2, 32))
    output = torch.zeros_like(query)

    with (
        patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(capturing=False)),
        patch(
            "vllm_ascend.attention.attention_v1.torch_npu.npu_fused_infer_attention_score",
            return_value=(torch.ones_like(query), None),
        ),
    ):
        result = impl._forward_c8_chunked_prefill(query, None, None, metadata, output, make_layer())

    assert result is output
    assert torch.count_nonzero(output != 1) == 0
    impl._build_c8_continuing_prefill_request.assert_not_called()
    impl._dequant_paged_kv_to_dense.assert_called_once()
    if provider is not None:
        assert provider.eligibility_requests == []
        assert provider.requests == []


def test_prefill_cache_hit_provider_bypasses_fused_dense_fallback():
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    provider = RecordingProvider()
    impl = make_impl(provider)
    impl._get_fia_params = MagicMock(side_effect=AssertionError("dense fallback must not run"))
    metadata = make_metadata()
    metadata.attn_state = AscendAttentionState.PrefillCacheHit
    output = torch.zeros((2, 2, 32))

    with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(capturing=False)):
        result = impl._forward_c8_fused_infer_attention(
            query=torch.zeros((2, 2, 32)),
            key=torch.zeros((2, 1, 32), dtype=torch.int8),
            value=torch.zeros((2, 1, 32), dtype=torch.int8),
            attn_metadata=metadata,
            output=output,
            layer=make_layer(),
        )

    assert result is output
    assert len(provider.requests) == 1
    impl._get_fia_params.assert_not_called()


@pytest.mark.parametrize("provider_enabled", [False, True])
def test_capture_dispatch_preserves_default_and_routes_enabled_provider(provider_enabled):
    impl = make_impl(RecordingProvider() if provider_enabled else None)
    impl.vllm_config = SimpleNamespace(kv_transfer_config=None)
    impl._layer_name = None
    impl._prepare_c8_scales = MagicMock()
    impl._quantize_kv_to_int8 = MagicMock(side_effect=lambda key, value, layer, count: (key, value))
    impl._reshape_and_cache = MagicMock(
        side_effect=lambda query, key, value, cache, metadata, output: (query, key, value, output)
    )
    impl._has_cached_multi_token_prefill = MagicMock(return_value=True)
    impl._build_c8_continuing_prefill_request = MagicMock(return_value=MagicMock())
    impl._try_c8_continuing_prefill_provider = MagicMock(return_value=provider_enabled)
    impl._forward_c8_chunked_prefill = MagicMock(return_value=torch.full((2, 2, 32), 3.0))
    impl.full_graph_fia = MagicMock(return_value=(torch.full((2, 2, 32), 5.0), 2))
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        model_runner_type="generate",
        attn_state=None,
    )
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn")
    query = torch.zeros((2, 2, 32))
    key = torch.zeros((2, 1, 32))
    value = torch.zeros((2, 1, 32))
    output = torch.zeros_like(query)

    with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(capturing=True)):
        result = impl.forward(layer, query, key, value, (impl.key_cache, impl.value_cache), metadata, output)

    if provider_enabled:
        assert torch.all(result == 3)
        impl._forward_c8_chunked_prefill.assert_called_once()
        impl.full_graph_fia.assert_not_called()
    else:
        assert torch.all(result == 5)
        impl._forward_c8_chunked_prefill.assert_not_called()
        impl.full_graph_fia.assert_called_once()


def test_capture_ineligible_provider_preserves_existing_graph_path():
    impl = make_impl(RecordingProvider(eligible=False))
    impl.vllm_config = SimpleNamespace(kv_transfer_config=None)
    impl._layer_name = None
    impl._prepare_c8_scales = MagicMock()
    impl._quantize_kv_to_int8 = MagicMock(side_effect=lambda key, value, layer, count: (key, value))
    impl._reshape_and_cache = MagicMock(
        side_effect=lambda query, key, value, cache, metadata, output: (query, key, value, output)
    )
    impl._has_cached_multi_token_prefill = MagicMock(return_value=True)
    impl._build_c8_continuing_prefill_request = MagicMock(return_value=MagicMock())
    impl._try_c8_continuing_prefill_provider = MagicMock(return_value=False)
    impl._forward_c8_chunked_prefill = MagicMock()
    impl.full_graph_fia = MagicMock(return_value=(torch.full((2, 2, 32), 5.0), 2))
    metadata = SimpleNamespace(num_actual_tokens=2, model_runner_type="generate", attn_state=None)
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn")
    query = torch.zeros((2, 2, 32))
    key = torch.zeros((2, 1, 32))
    value = torch.zeros((2, 1, 32))
    output = torch.zeros_like(query)

    with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(capturing=True)):
        result = impl.forward(layer, query, key, value, (impl.key_cache, impl.value_cache), metadata, output)

    assert torch.all(result == 5)
    impl._try_c8_continuing_prefill_provider.assert_called_once()
    impl._forward_c8_chunked_prefill.assert_not_called()
    impl.full_graph_fia.assert_called_once()

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_ascend.kv_cache_compression import registry
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker import worker as worker_module
from vllm_ascend.worker.worker import NPUWorker


class FakeEntryPoint:
    def __init__(self, factory) -> None:
        self.factory = factory

    def load(self):
        return self.factory


class FakeEntryPoints(list):
    def select(self, *, group: str, name: str):
        assert group == registry.PROVIDER_ENTRY_POINT_GROUP
        return self


def _config(provider: str = "pyramidkv_ascend") -> SimpleNamespace:
    return SimpleNamespace(schema_version=1, provider=provider, provider_config={})


def _install_contract_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("vllm.v1.kv_cache_compression")
    module.KVCacheCompressionCompatibility = SimpleNamespace
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_platform_declares_only_the_generic_registry() -> None:
    assert NPUPlatform.get_kv_cache_compression_provider_factory() == (
        "vllm_ascend.kv_cache_compression.registry:get_kv_cache_compression_provider"
    )


@pytest.mark.parametrize("count", [0, 2])
def test_registry_requires_exactly_one_installed_provider(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    loaded = []
    entry_points = FakeEntryPoints(FakeEntryPoint(lambda config: loaded.append(config)) for _ in range(count))
    monkeypatch.setattr(registry, "entry_points", lambda: entry_points)

    with pytest.raises(RuntimeError, match=f"found {count}"):
        registry.get_kv_cache_compression_provider(_config())

    assert loaded == []


def test_registry_loads_selected_provider_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = []

    def factory(config):
        loaded.append(config)
        return "provider"

    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda: FakeEntryPoints([FakeEntryPoint(factory)]),
    )
    config = _config()

    assert registry.get_kv_cache_compression_provider(config) == "provider"
    assert loaded == [config]


def test_worker_fails_closed_when_provider_cannot_be_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_contract_stub(monkeypatch)
    config = _config()
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_cache_compression_config=config),
        current_platform=SimpleNamespace(
            device_type="npu",
            get_kv_cache_compression_provider_factory=lambda: "missing.provider:get_provider",
        ),
    )

    def missing_module(_name: str):
        raise ModuleNotFoundError("No module named 'missing.provider'")

    monkeypatch.setattr(worker_module, "import_module", missing_module)
    report = NPUWorker.validate_kv_cache_compression(worker)

    assert not report.supported
    assert report.runtime_spec is None
    assert "provider initialization failed" in report.reasons[0]
    assert "missing.provider" in report.reasons[0]


def test_worker_retains_provider_only_after_successful_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_contract_stub(monkeypatch)
    config = _config()
    provider = SimpleNamespace(validate_worker=lambda config, worker, factory: SimpleNamespace(supported=True))
    module = SimpleNamespace(get_provider=lambda config: provider)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_cache_compression_config=config),
        current_platform=SimpleNamespace(
            device_type="npu",
            get_kv_cache_compression_provider_factory=lambda: "provider.module:get_provider",
        ),
        kv_cache_compression_provider=None,
    )
    monkeypatch.setattr(worker_module, "import_module", lambda name: module)

    report = NPUWorker.validate_kv_cache_compression(worker)

    assert report.supported
    assert worker.kv_cache_compression_provider is provider

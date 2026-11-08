# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "native_cache", Path(__file__).resolve().parents[3] / "csrc/scripts/run_host_compiler_cache.py"
)
assert spec is not None and spec.loader is not None
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def test_config_permissions_ca_and_symlink(tmp_path):
    path = tmp_path / "config.json"
    (tmp_path / "ca.crt").write_text("test certificate")
    path.write_text(json.dumps({"ca_cert": "ca.crt"}))
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        cache.read_config(path)
    path.chmod(0o600)
    assert cache.read_config(path)["ca_cert"] == str(tmp_path / "ca.crt")
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="0600"):
        cache.read_config(link)


@pytest.mark.parametrize("available", [True, False])
def test_private_scoped_daemon_and_fallback(tmp_path, monkeypatch, available):
    monkeypatch.setenv("SCCACHE_REDIS_PASSWORD", "unrelated-secret")
    monkeypatch.setenv("SCCACHE_SERVER_PORT", "49999")
    monkeypatch.delenv("VLLM_ASCEND_HOST_COMPILER_LAUNCHER", raising=False)
    monkeypatch.delenv("CMAKE_CXX_COMPILER_LAUNCHER", raising=False)
    calls = []
    secret_file = None

    def subprocess_run(args, *, env, **kwargs):
        nonlocal secret_file
        calls.append(args)
        assert env["SCCACHE_IDLE_TIMEOUT"] == "600"
        assert "SCCACHE_REDIS_PASSWORD" not in env and "SCCACHE_SERVER_PORT" not in env
        secret_file = Path(env["SCCACHE_CONF"])
        assert secret_file.stat().st_mode & 0o777 == 0o600
        assert "test-secret" in secret_file.read_text()
        assert "test-secret" not in repr(env) and "test-secret" not in repr(args)
        assert env["SSL_CERT_FILE"] == "/trusted/ca.crt"
        if "--start-server" in args:
            return SimpleNamespace(returncode=0 if available else 1)
        if "--show-stats" in args:
            return SimpleNamespace(
                returncode=0, stdout=json.dumps({"cache_location": "private endpoint", "stats": {"compilations": 0}})
            )
        if args == ["build"]:
            assert env["VLLM_ASCEND_HOST_COMPILER_LAUNCHER"] == ("sccache" if available else "")
            return SimpleNamespace(returncode=7)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cache.subprocess, "run", subprocess_run)
    config = dict(
        host="cache.invalid",
        port=6380,
        username="native",
        password="test-secret",
        ca_cert="/trusted/ca.crt",
        prefix="native-sccache/test/",
    )
    receipt = tmp_path / "stats.json"
    assert cache.run(config, "sccache", ["build"], receipt) == 7
    assert calls[-1] == ["sccache", "--stop-server"]
    assert secret_file is not None
    assert not secret_file.exists()
    assert receipt.exists() == available
    if available:
        assert json.loads(receipt.read_text()) == {"compilations": 0}


def test_does_not_replace_existing_launcher(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_HOST_COMPILER_LAUNCHER", "canonical-depfiles")
    with pytest.raises(ValueError, match="existing compiler launcher"):
        cache.run({}, "sccache", ["build"])


def test_start_timeout_still_builds_locally(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_HOST_COMPILER_LAUNCHER", raising=False)
    monkeypatch.delenv("CMAKE_CXX_COMPILER_LAUNCHER", raising=False)
    calls = []

    def execute(args, *, env, **kwargs):
        calls.append(args)
        if "--start-server" in args:
            raise cache.subprocess.TimeoutExpired(args, kwargs["timeout"])
        if args == ["build"]:
            assert env["VLLM_ASCEND_HOST_COMPILER_LAUNCHER"] == ""
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cache.subprocess, "run", execute)
    config = dict(
        host="cache.invalid",
        port=6380,
        username="project",
        password="test-secret",
        ca_cert="/trusted/ca.crt",
        prefix="native/project/",
    )
    assert cache.run(config, "sccache", ["build"]) == 0
    assert calls[-1] == ["sccache", "--stop-server"]


def test_standalone_cli_imports_without_package_dependencies():
    result = subprocess.run([sys.executable, str(cache.__file__), "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout

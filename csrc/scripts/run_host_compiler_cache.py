#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a host-extension build with an isolated sccache Redis client (Linux).

Credentials stay in a private temporary TOML, not argv or compiler environment.
The server stores objects only; all compilation and linking remain on this client.
"""

import argparse
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

START_TIMEOUT_SECONDS = 20
CONTROL_TIMEOUT_SECONDS = 10
IDLE_TIMEOUT_SECONDS = 600


def configured_host_launcher():
    # Bootstrap tooling runs before the package (and vLLM) is installed, as does
    # setup.py: load the dependency-free central env module without __init__.
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/envs.py"
    spec = importlib.util.spec_from_file_location("build_envs", path)
    envs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(envs)
    return envs.VLLM_ASCEND_HOST_COMPILER_LAUNCHER


def read_config(path):
    path = Path(path).absolute()
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Cache config must be an owned regular file with mode 0600")
    config = json.loads(path.read_text())
    ca = Path(config["ca_cert"])
    config["ca_cert"] = str((path.parent / ca).resolve())
    if not Path(config["ca_cert"]).is_file():
        raise ValueError("Cache CA file is missing")
    return config


def run(config, sccache, command, stats_path=None):
    if configured_host_launcher() or os.environ.get("CMAKE_CXX_COMPILER_LAUNCHER"):
        raise ValueError("Do not overwrite an existing compiler launcher")
    # Never attach to another build's daemon or inherit another backend/credentials.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SCCACHE_")}
    with tempfile.TemporaryDirectory(prefix="native-cache-") as directory:
        root = Path(directory)
        conf = root / "cache.toml"
        values = dict(
            endpoint=f"rediss://{config['host']}:{int(config['port'])}",
            username=config["username"],
            password=config["password"],
            key_prefix=config["prefix"],
        )
        with open(conf, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as f:
            f.write("[cache.redis]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in values.items()))
        # Match sccache's default: a 60s idle budget can lose counters during long links.
        env.update(
            SCCACHE_CONF=str(conf),
            SCCACHE_SERVER_UDS=str(root / "socket"),
            SSL_CERT_FILE=config["ca_cert"],
            SCCACHE_IDLE_TIMEOUT=str(IDLE_TIMEOUT_SECONDS),
            SCCACHE_IGNORE_SERVER_IO_ERROR="1",
        )
        started = False
        try:
            try:
                result = subprocess.run(
                    [sccache, "--start-server"], env=env, capture_output=True, timeout=START_TIMEOUT_SECONDS
                )
                started = result.returncode == 0
            except subprocess.TimeoutExpired:
                pass
            if not started:
                print("Native cache unavailable; building locally without cache.", file=sys.stderr)
            # setup.py forwards an empty value too, clearing a previous selection.
            env["VLLM_ASCEND_HOST_COMPILER_LAUNCHER"] = sccache if started else ""
            result = subprocess.run(command, env=env)
            if started and stats_path:
                stats = subprocess.run(
                    [sccache, "--show-stats", "--stats-format", "json"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=CONTROL_TIMEOUT_SECONDS,
                )
                if stats.returncode == 0:
                    # Only counters: no endpoint, config or environment in public receipts.
                    data = json.loads(stats.stdout)
                    Path(stats_path).write_text(json.dumps(data["stats"], indent=2) + "\n")
            return result.returncode
        finally:
            try:
                subprocess.run(
                    [sccache, "--stop-server"], env=env, capture_output=True, timeout=CONTROL_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired:
                print("Native cache client shutdown timed out; idle timeout is 600s.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sccache", default="sccache")
    parser.add_argument("--stats", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a build command after --")
    binary = shutil.which(args.sccache)
    if not binary:
        parser.error("Install the qualified sccache binary first")
    return run(read_config(args.config), binary, command, args.stats)


if __name__ == "__main__":
    sys.exit(main())

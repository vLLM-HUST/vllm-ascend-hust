<!-- markdownlint-disable MD013 MD033 MD041 -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vLLM Ascend logo" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width="420">
  </picture>
</p>

<h3 align="center">
HUST-maintained Ascend/NPU plugin paired with vLLM-HUST
</h3>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a> |
<a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>Upstream Ascend Docs</b></a> |
<a href="https://github.com/vllm-project/vllm-ascend"><b>Upstream vLLM Ascend</b></a> |
<a href="https://github.com/vLLM-HUST/vllm-hust"><b>HUST Core Fork</b></a> |
<a href="README.zh.md"><b>中文</b></a> |
</p>

# vLLM-Ascend-SpecSLO

本仓库是基于 vLLM-Ascend-HUST 的 Ascend/NPU SpecSLO 实现，包含
nano-PEARL 原生运行时、树状投机解码基础设施、SLO 调度与 Ascend
图/通信适配。完整的中文迁移记录见
[`docs/specslo_work_record_zh.md`](docs/specslo_work_record_zh.md)。
逐项对照 atc26v0 与论文的实现审计见
[`docs/atc26v0_feature_audit_zh.md`](docs/atc26v0_feature_audit_zh.md)。

原 vLLM-Ascend-HUST 说明保留如下，便于安装、开发和追踪上游差异。

## vLLM-Ascend-HUST

`vLLM-Ascend-HUST` is the HUST-maintained fork of the
[`vllm-project/vllm-ascend`](https://github.com/vllm-project/vllm-ascend)
backend plugin. It is paired with
[`vllm-hust`](https://github.com/vLLM-HUST/vllm-hust), the HUST core vLLM fork.

The repository keeps the upstream vLLM Ascend plugin identity while carrying the
local patches, scripts, and runtime integration needed for HUST Ascend/NPU
experiments.

## What Comes From Upstream vLLM Ascend

The upstream `vllm-ascend` project provides the Ascend backend plugin for vLLM:

- NPU platform registration and device integration
- model execution support on Ascend hardware
- custom ops and kernel integration paths
- worker, attention, sampling, and distributed-runtime adaptations
- documentation for running vLLM on Ascend devices
- community support through the vLLM Ascend SIG

For general Ascend plugin usage, start with the upstream documentation:
<https://docs.vllm.ai/projects/ascend/en/latest/>.

## What HUST Adds

HUST-specific changes focus on making the plugin practical for local research
and managed NPU service workflows:

- Pairing and compatibility with `vllm-hust`.
- HUST patches for model loading, tool-call parsing, sampling, scheduling, and
  worker behavior on Ascend.
- Local custom-kernel build switches and kernel-development helpers.
- Single-device Ascend environment scripts.
- Managed-service compatibility for dev-hub `manage.sh`.
- Upstream merge/version metadata used to keep fork drift visible.

The plugin should stay thin where possible. Core vLLM behavior belongs in
`vllm-hust`; Ascend-only runtime behavior belongs here.

## Repository Map

| Path | Purpose |
| --- | --- |
| `vllm_ascend/` | Python plugin package and Ascend runtime patches. |
| `csrc/` | Optional C/C++/kernel sources and third-party kernel dependencies. |
| `scripts/` | Local install and Ascend environment helpers. |
| `tests/` | Plugin-side tests and smoke coverage. |
| `upstream_version.json` | Current upstream anchor and HUST release base. |
| `AGENTS.md` | Required workflow rules for AI-assisted changes. |

## Paired HUST Repositories

| Repository | Role |
| --- | --- |
| [`vllm-hust`](https://github.com/vLLM-HUST/vllm-hust) | Core vLLM fork. |
| [`vllm-ascend-hust`](https://github.com/vLLM-HUST/vllm-ascend-hust) | Ascend/NPU plugin fork. |
| [`vllm-hust-dev-hub`](https://github.com/vLLM-HUST/vllm-hust-dev-hub) | Multi-repo workspace, managed service scripts, and NPU smoke-test entrypoint. |
| [`vllm-hust-benchmark`](https://github.com/vLLM-HUST/vllm-hust-benchmark) | Benchmark orchestration and result export. |
| [`vllm-hust-perf-analyzer`](https://github.com/vLLM-HUST/vllm-hust-perf-analyzer) | Offline profiler timeline analysis. |

Mixing this plugin with an unrelated vLLM checkout can hide ABI, API, or runtime
contract mismatches. Keep both HUST repositories on their paired `main` heads
unless a branch explicitly says otherwise.

## Versioning

This fork follows the same upstream-anchored version rule as `vllm-hust`:

```text
<upstream release>.post1.dev<HUST-only commit count>+g<short sha>
```

`upstream_version.json` records the anchor:

- `upstream_commit`: exact upstream commit included in this fork graph.
- `upstream_version`: upstream-compatible version string, including rc suffix
  when upstream is on an rc.
- `release_version`: the same version line without the rc suffix.

After an upstream sync lands, the fork should be zero commits behind upstream:

```bash
git fetch upstream main
git rev-list --left-right --count origin/main...upstream/main
# <HUST-only commits>  0
```

The left side is HUST-only delta; the right side should be `0`.

## Install For Development

Use `uv` and the paired `vllm-hust` virtual environment. Do not install with
system `python3` or bare `pip`.

```bash
cd /path/to/vllm-hust
uv venv --python 3.12
source .venv/bin/activate
uv pip install \
  -r requirements/common.txt \
  -r /path/to/vllm-ascend-hust/requirements.txt \
  -r requirements/build/empty.txt \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --index-strategy unsafe-best-match
VLLM_TARGET_DEVICE=empty uv pip install -e . \
  --no-build-isolation --no-deps
```

The first command installs the core common runtime, the Ascend-selected
Torch/NPU runtime, and the Torch-free build tools for the `empty` target into
one environment. The core editable install can then reuse those packages
without selecting a CUDA wheel or invoking the generic PEP 517 Torch 2.11
requirement.

Then install this plugin:

```bash
cd /path/to/vllm-ascend-hust
COMPILE_CUSTOM_KERNELS=0 uv pip install -e . --no-deps
```

The plugin keeps its normal isolated build. `--no-deps` is safe here because
the target runtime requirements were installed and resolved together above.

On HUST local hosts, prefer the repository install helper:

```bash
cd /path/to/vllm-ascend-hust
bash scripts/install_local_ascend_plugin.sh /path/to/vllm-ascend-hust
```

If custom kernels are part of the change under test, set the build flags
required by the local Ascend/CANN environment before installing.

## Ascend Environment

For single-device local tests, source the helper script:

```bash
cd /path/to/vllm-ascend-hust
source scripts/use_single_ascend_env.sh /usr/local/Ascend/ascend-toolkit/latest
```

Then verify that the plugin imports with the paired vLLM checkout:

```bash
VLLM_PLUGINS=ascend \
VLLM_TARGET_DEVICE=npu \
.venv/bin/python -c "import vllm, vllm_ascend; print(vllm.__version__)"
```

On shared machines, use only the allocated device. The current HUST smoke-test
workflow is constrained to NPU 1 unless the operator explicitly assigns another
device.

## Managed NPU Smoke Tests

Production-like serving tests should be launched through dev-hub:

```bash
cd /path/to/vllm-hust-dev-hub
./manage.sh status
./manage.sh restart
./manage.sh health --json
```

This keeps environment setup, device selection, ports, logs, and cleanup aligned
with HUST experiment services.

## Validation Checklist

SpecSLO/Ascend 算子能力（不加载模型）可直接检查：

```bash
python examples/check_specslo_capabilities.py --device npu --tp-size 3
```

README-only changes:

```bash
git diff --check -- README.md README.zh.md
```

Plugin Python changes:

```bash
.venv/bin/python -m py_compile path/to/file.py
pre-commit run --files path/to/file.py
```

Upstream merges:

```bash
git diff --name-only --diff-filter=U
git rev-list --left-right --count origin/main...upstream/main
git submodule status --recursive
```

NPU runtime changes should also be tested through `manage.sh` from
`vllm-hust-dev-hub`, using NPU 1 unless another device is assigned.

## Upstream Sync Workflow

1. Fetch upstream and create a staging branch from `origin/main`.
2. Merge `upstream/main` into that branch with a real merge commit when
   possible.
3. Resolve conflicts by keeping Ascend-only deltas here and moving core behavior
   into `vllm-hust`.
4. Update `upstream_version.json` and the derived package version.
5. Validate imports, syntax, submodules, and NPU smoke behavior.
6. Merge the staging PR, pull `main`, and confirm the fork is zero commits
   behind upstream.

Avoid routine cherry-pick/backport stacks when a real upstream merge can keep
the fork graph honest.

## Documentation And Community

- Upstream vLLM Ascend docs: <https://docs.vllm.ai/projects/ascend/en/latest/>
- Upstream vLLM Ascend repository: <https://github.com/vllm-project/vllm-ascend>
- Upstream vLLM repository: <https://github.com/vllm-project/vllm>
- HUST core fork: <https://github.com/vLLM-HUST/vllm-hust>
- HUST organization: <https://github.com/vLLM-HUST>
- HUST agent workflow: [`AGENTS.md`](AGENTS.md)

Chinese documentation is available in [`README.zh.md`](README.zh.md).

## License

This repository follows the upstream vLLM Ascend license. See
[`LICENSE`](LICENSE) and upstream notices for details.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import-not-found]
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found]


def test_build_requirements_only_contain_setup_dependencies():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    build_requirements = tomllib.loads(pyproject.read_text())["build-system"]["requires"]

    package_names = {
        requirement.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].lower() for requirement in build_requirements
    }
    runtime_only_packages = {
        "fastapi",
        "torch",
        "torch-npu",
        "transformers",
        "triton-ascend",
    }

    assert package_names.isdisjoint(runtime_only_packages)


def test_paired_editable_workflow_uses_empty_target_dependency_sets():
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/pr_test.yaml").read_text()

    job = workflow.index("validate-hust-dual-editable:")
    checkout = workflow.index("repository: vLLM-HUST/vllm-hust")
    runtime_install = workflow.index("-r ./vllm-hust/requirements/common.txt")
    build_tools = workflow.index("-r ./vllm-hust/requirements/build/empty.txt")
    core_install = workflow.index("VLLM_TARGET_DEVICE=empty uv pip install")
    plugin_install = workflow.index("COMPILE_CUSTOM_KERNELS=0")

    assert "runs-on: ubuntu-24.04-arm" in workflow[job:checkout]
    assert "runs-on: linux-aarch64-a2b3-1" not in workflow[job:checkout]
    assert "container:" not in workflow[job:checkout]
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD: 0" in workflow[job:checkout]
    assert "ref: main" in workflow[checkout:runtime_install]
    assert "-r ./requirements.txt" in workflow[runtime_install:core_install]
    assert runtime_install < build_tools < core_install < plugin_install
    assert "--no-build-isolation" in workflow[core_install:plugin_install]
    assert "--no-deps" in workflow[core_install:plugin_install]
    assert "--no-build-isolation" not in workflow[plugin_install:]
    assert "from importlib.metadata import distribution" in workflow[plugin_install:]
    assert 'Path(get_path("purelib")).glob(' in workflow[plugin_install:]
    assert 'f"__editable__.{normalized_name}-*.pth"' in workflow[plugin_install:]
    assert "all(path.is_file() for path in editable_paths)" in workflow[plugin_install:]
    assert 'for package in ("vllm", "vllm-ascend-hust")' in workflow[plugin_install:]
    assert "import vllm; import vllm_ascend" not in workflow[plugin_install:]


def test_dual_editable_documentation_uses_target_specific_flow():
    root = Path(__file__).resolve().parents[2]
    required = (
        "requirements/common.txt",
        "/path/to/vllm-ascend-hust/requirements.txt",
        "requirements/build/empty.txt",
        "VLLM_TARGET_DEVICE=empty uv pip install -e .",
        "--no-build-isolation --no-deps",
        "COMPILE_CUSTOM_KERNELS=0 uv pip install -e . --no-deps",
    )

    for readme in ("README.md", "README.zh.md"):
        contents = (root / readme).read_text()
        for command_fragment in required:
            assert command_fragment in contents

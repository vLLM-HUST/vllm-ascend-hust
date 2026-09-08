# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
CONFIG_SCRIPT = REPO_ROOT / "csrc/cmake/scripts/util/ascendc_ops_config.py"


def test_generated_kernel_path_does_not_include_build_bin_directory(tmp_path: Path) -> None:
    kernel_root = tmp_path / "ascend910b" / "bin"
    op_dir = kernel_root / "matmul_allreduce_add_rmsnorm"
    op_dir.mkdir(parents=True)
    kernel_json = op_dir / "kernel.json"
    kernel_json.write_text("{}", encoding="utf-8")

    subprocess.run(
        [sys.executable, str(CONFIG_SCRIPT), "-p", str(kernel_root), "-s", "ascend910b"],
        check=True,
    )

    generated = json.loads(kernel_json.read_text(encoding="utf-8"))
    assert generated["filePath"] == "ascend910b/matmul_allreduce_add_rmsnorm/kernel.json"

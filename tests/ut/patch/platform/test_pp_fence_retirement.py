# SPDX-License-Identifier: Apache-2.0
"""Exercise exact patch functions without importing NPU runtime dependencies."""

import ast
import copy
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_older_prefill_output_cannot_release_later_decode_fence(monkeypatch):
    class Scheduler:
        def _update_after_schedule(self, output):
            pass

        def update_from_output(self, output, model_output):
            return {}

    fake_module = ModuleType("vllm.v1.core.sched.scheduler")
    fake_module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    path = Path(__file__).parents[4] / "vllm_ascend/patch/platform/patch_pp_mtp.py"
    tree = ast.parse(path.read_text())
    names = {"_patch_scheduler_update_after_schedule", "_patch_scheduler_update_from_output"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(wraps=wraps, copy=copy, _PP_IN_FLIGHT_STEP=1 << 60, _use_pp_ipc_runtime_patch=lambda *args: True)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "<pp-fence-regression>", "exec"
        ),
        namespace,
    )
    namespace["_patch_scheduler_update_after_schedule"]()
    namespace["_patch_scheduler_update_from_output"]()
    scheduler = Scheduler()
    request = SimpleNamespace(is_prefill_chunk=True, next_decode_eligible_step=0)
    scheduler.requests = {"a": request}
    scheduler.use_pp = True
    scheduler.vllm_config = SimpleNamespace(speculative_config=None)
    intermediate = SimpleNamespace(num_scheduled_tokens={"a": 4096})
    final = SimpleNamespace(num_scheduled_tokens={"a": 2048})
    scheduler._update_after_schedule(intermediate)
    request.is_prefill_chunk = False
    scheduler._update_after_schedule(final)
    assert request.next_decode_eligible_step == 1 << 60
    scheduler.update_from_output(intermediate, None)
    assert request.next_decode_eligible_step == 1 << 60
    scheduler.update_from_output(final, None)
    assert request.next_decode_eligible_step == 0

"""Exercise the runner's sampler construction without allocating device buffers."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("mode", ["raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits"])
def test_runner_forwards_configured_logprobs_mode(mode):
    source = Path(__file__).parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text())
    runner = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    init = next(
        node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "sampler" for target in node.targets)
    ]
    assert len(assignments) == 1
    constructor = Mock()
    instance = SimpleNamespace(model_config=SimpleNamespace(logprobs_mode=mode))
    code = compile(ast.Module(body=assignments, type_ignores=[]), str(source), "exec")
    exec(code, {"self": instance, "AscendSampler": constructor})
    constructor.assert_called_once_with(logprobs_mode=mode)
    assert instance.sampler is constructor.return_value

# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for PP rejection feedback transport, independent of NPU imports."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "pp_hybrid_feedback",
    Path(__file__).parents[3] / "vllm_ascend/worker/pp_hybrid_feedback.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "ids,expected",
    [
        ([[12], [17]], [[12, -1, -1], [17, -1, -1]]),
        ([[12, 13, 14], [17, -1, -1]], [[12, 13, 14], [17, -1, -1]]),
        ([[-1, -1, -1]], [[-1, -1, -1]]),
    ],
)
def test_feedback_preserves_rejection_counts(ids, expected):
    pp = SimpleNamespace(is_last_rank=True, last_rank=2, device_group=object())
    with patch.object(torch.distributed, "broadcast") as broadcast:
        output = module.sync_sampled_tokens(pp, torch.tensor(ids), len(ids), 2, "cpu")
    assert output.tolist() == expected
    broadcast.assert_called_once_with(output, src=2, group=pp.device_group)


def test_nonfinal_rank_receives_actual_final_stage_rows():
    pp = SimpleNamespace(is_last_rank=False, last_rank=2, device_group=object())

    def receive(tensor, **kwargs):
        tensor.copy_(torch.tensor([[12, 13, -1], [15, -1, -1]]))

    with patch.object(torch.distributed, "broadcast", side_effect=receive):
        output = module.sync_sampled_tokens(pp, None, 2, 2, "cpu")
    assert (output != -1).sum(dim=1).tolist() == [2, 1]


@pytest.mark.parametrize("shape", [(2,), (3, 1), (2, 4), (2, 0)])
def test_bad_shape_fails_before_collective(shape):
    pp = SimpleNamespace(is_last_rank=True, last_rank=2, device_group=object())
    with patch.object(torch.distributed, "broadcast") as broadcast:
        with pytest.raises(ValueError, match="shape"):
            module.sync_sampled_tokens(pp, torch.zeros(shape), 2, 2, "cpu")
        broadcast.assert_not_called()


def test_tail_and_counts_preserve_partial_and_empty_rows():
    ids = torch.tensor([[12, 13, 14], [17, -1, -1], [-1, -1, -1]])
    tail, counts = module.sampled_tail_and_counts(ids)
    assert tail.tolist() == [14, 17, -1]
    assert counts.tolist() == [3, 1, 0]


def test_feedback_survives_empty_and_disjoint_batches():
    retained = module.retain_request_feedback({}, ["a", "b"], [3, 1], [])
    retained = module.retain_request_feedback(retained, [], [], [])
    retained = module.retain_request_feedback(retained, ["c"], [2], [])
    assert [retained[r] for r in ["b", "a", "c"]] == [1, 3, 2]


def test_feedback_resets_only_explicit_lifecycle_ids():
    retained = module.retain_request_feedback({"old": 3}, ["done", "kept"], [2, 3], ["old", "done"])
    assert retained == {"kept": 3}


def test_runner_retains_offsets_before_empty_batch_removes_rows():
    import ast
    from collections.abc import Callable
    from unittest.mock import Mock

    source = Path(__file__).parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "_update_states")

    class Base:
        def _update_states(self, output):
            self.input_batch.req_ids = []
            return "base-updated"

    namespace = dict(Base=Base, Callable=Callable, retain_request_feedback=module.retain_request_feedback)
    wrapper = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "<runner-regression>", "exec"),
        namespace,
    )
    obj = namespace["Runner"]()
    obj.cache_config = SimpleNamespace(mamba_cache_mode="align")
    obj.num_accepted_tokens_event = Mock()
    obj.input_batch = SimpleNamespace(req_ids=["a"])
    obj.accepted_tokens_feedback_cpu = torch.tensor([3])
    obj._hybrid_accepted_by_req = {}
    obj.use_async_scheduling = False
    obj._apply_pp_sampled_tokens_from_scheduler_output = Mock()
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )
    assert obj._update_states(output) == "base-updated"
    assert obj.input_batch.req_ids == []
    assert obj._hybrid_accepted_by_req == {"a": 3}
    obj._update_states(output)
    assert obj._hybrid_accepted_by_req == {"a": 3}
    output.finished_req_ids = {"a"}
    obj._update_states(output)
    assert obj._hybrid_accepted_by_req == {}


def test_drafts_map_after_each_requests_sampled_token():
    indices, tokens = module.scheduled_draft_inputs(
        ["prefill", "decode-b", "decode-a"],
        {"decode-a": [90], "decode-b": [70, 71]},
        [8, 11, 13],
    )
    assert indices == [9, 10, 12]
    assert tokens == [70, 71, 90]


@pytest.mark.parametrize("drafts", [[-1], [1, 2, 3]])
def test_placeholder_or_oversized_drafts_fail_closed(drafts):
    with pytest.raises(ValueError, match="actual scheduled"):
        module.scheduled_draft_inputs(["a"], {"a": drafts}, [2])


@pytest.mark.parametrize("last_rank", [False, True])
def test_runner_replaces_stale_nonfinal_drafts_on_all_common_rows(last_rank):
    import ast

    source = Path(__file__).parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text())
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_input_ids")

    class Base:
        def _prepare_input_ids(self, *args):
            self.input_ids.gpu[0] = 82601  # base scatters only the confirmed token

    namespace = dict(
        Base=Base,
        torch=torch,
        scheduled_draft_inputs=module.scheduled_draft_inputs,
        get_pp_group=lambda: SimpleNamespace(world_size=2, is_last_rank=last_rank),
    )
    wrapper = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "<pp-draft-regression>", "exec"
        ),
        namespace,
    )
    obj = namespace["Runner"]()
    obj.use_async_scheduling = True
    obj.device = "cpu"
    obj.input_ids = SimpleNamespace(gpu=torch.tensor([0, 13, 198], dtype=torch.int32))
    obj.input_batch = SimpleNamespace(req_ids=["a"])
    output = SimpleNamespace(scheduled_spec_decode_tokens={"a": [3045, 75351]})
    obj._prepare_input_ids(output, 1, 3, [3])
    assert obj.input_ids.gpu.tolist() == ([82601, 13, 198] if last_rank else [82601, 3045, 75351])


@pytest.mark.parametrize("old_output_count, confirmed_output_count", [(3, 2), (1, 3)])
def test_fenced_counts_do_not_apply_optimistic_rejection_twice(old_output_count, confirmed_output_count):
    import ast
    from collections.abc import Callable

    import numpy as np

    source = Path(__file__).parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text())
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "_update_states")

    class Base:
        def _update_states(self, output):
            state = self.requests["a"]
            assert state.prev_num_draft_len == 0
            assert len(state.output_token_ids) == confirmed_output_count
            assert self.input_batch.num_tokens_no_spec[0] == 1024 + confirmed_output_count
            return None

    namespace = dict(Base=Base, Callable=Callable, PLACEHOLDER_TOKEN_ID=-1)
    wrapper = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
            "<pp-confirmed-count-regression>",
            "exec",
        ),
        namespace,
    )
    obj = namespace["Runner"]()
    obj.cache_config = SimpleNamespace(mamba_cache_mode="align")
    obj.num_accepted_tokens_event = None
    obj.use_async_scheduling = True
    obj.requests = {
        "a": SimpleNamespace(prev_num_draft_len=2, num_computed_tokens=1027, output_token_ids=[12] * old_output_count)
    }
    obj.input_batch = SimpleNamespace(
        req_id_to_index={"a": 0},
        num_tokens_no_spec=[1024 + old_output_count],
        num_prompt_tokens=[1024],
        is_token_ids=np.zeros((1, 1040), dtype=bool),
    )
    obj._apply_pp_sampled_tokens_from_scheduler_output = lambda output: None
    output = SimpleNamespace(
        _ascend_pp_confirmed_counts=True,
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["a"], num_computed_tokens=[1029], num_output_tokens=[confirmed_output_count]
        ),
    )
    obj._update_states(output)

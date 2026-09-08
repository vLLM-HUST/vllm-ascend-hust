# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ACLGraph capture and paged-attention graph-task updates for native PEARL."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu

DEFAULT_MAX_ACLGRAPH_ENTRIES = 16
MAX_GREEDY_REPLAY_EAGER_DIVERGENCE = 0.05


@dataclass
class NativePagedAttentionGraphTask:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    num_kv_heads: int
    num_heads: int
    scale: float
    block_table: torch.Tensor
    context_lens: torch.Tensor
    output: torch.Tensor
    workspace: torch.Tensor
    handle: Any
    event: torch.npu.ExternalEvent


@dataclass
class NativeFusedInferAttentionGraphTask:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    num_kv_heads: int
    num_heads: int
    scale: float
    block_table: torch.Tensor
    attention_mask: torch.Tensor
    output: torch.Tensor
    softmax_lse: torch.Tensor
    block_size: int
    workspace: torch.Tensor
    handle: Any
    event: torch.npu.ExternalEvent


@dataclass
class NativeACLGraphEntry:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    request_block_tables: torch.Tensor | None
    actual_seq_lengths_q: tuple[int, ...]
    sequence_lens: tuple[int, ...]
    graph: torch.npu.NPUGraph
    output: torch.Tensor
    tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask]
    runtime_validated: bool = False


@dataclass
class NativeDraftACLGraphEntry:
    input_ids: torch.Tensor
    positions: tuple[torch.Tensor, ...]
    slot_mappings: tuple[torch.Tensor, ...]
    context_lens: tuple[torch.Tensor, ...]
    block_tables: tuple[torch.Tensor, ...]
    request_block_tables: tuple[torch.Tensor | None, ...]
    actual_seq_lengths_q: tuple[tuple[int, ...], ...]
    sequence_lens: tuple[tuple[int, ...], ...]
    graph: torch.npu.NPUGraph
    output: torch.Tensor
    tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask]
    tasks_per_step: int


_CAPTURED_TASKS: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask] | None = None
_CAPTURED_PA_WORKSPACES: dict[tuple[Any, ...], torch.Tensor] | None = None
_CAPTURED_FIA_WORKSPACES: dict[tuple[Any, ...], torch.Tensor] | None = None

# Match vLLM-Ascend's ordinary causal FIA path.  The explicit attention mask
# carries the causal/tree structure; these bounds must stay unbounded so FIA
# does not apply a second relative window to TND queries.
_FIA_INT_MAX = 2147483647


@contextmanager
def _collect_graph_tasks():
    global _CAPTURED_FIA_WORKSPACES, _CAPTURED_PA_WORKSPACES, _CAPTURED_TASKS
    if _CAPTURED_TASKS is not None:
        raise RuntimeError("Nested native PEARL ACLGraph capture is not supported.")
    tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask] = []
    _CAPTURED_TASKS = tasks
    _CAPTURED_PA_WORKSPACES = {}
    _CAPTURED_FIA_WORKSPACES = {}
    try:
        yield tasks
    finally:
        _CAPTURED_TASKS = None
        _CAPTURED_PA_WORKSPACES = None
        _CAPTURED_FIA_WORKSPACES = None


def run_native_paged_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    scale: float,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Run eager paged attention or record its replay-time host parameters."""
    if _CAPTURED_TASKS is None:
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            block_table=block_table,
            context_lens=context_lens,
            out=output,
        )
        return

    assert _CAPTURED_PA_WORKSPACES is not None
    workspace_key = (
        tuple(query.shape),
        tuple(key_cache.shape),
        query.dtype,
        num_kv_heads,
        num_heads,
    )
    workspace = _CAPTURED_PA_WORKSPACES.get(workspace_key)
    if workspace is None:
        workspace = torch_npu._npu_paged_attention_get_workspace(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            block_table=block_table,
            context_lens=context_lens,
            out=output,
        )
        _CAPTURED_PA_WORKSPACES[workspace_key] = workspace
    stream = torch.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
    torch.npu.graph_task_group_begin(stream)
    torch_npu._npu_paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale_value=scale,
        block_table=block_table,
        context_lens=context_lens,
        out=output,
        workspace=workspace,
    )
    handle = torch.npu.graph_task_group_end(stream)
    _CAPTURED_TASKS.append(
        NativePagedAttentionGraphTask(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            block_table=block_table,
            context_lens=context_lens,
            output=output,
            workspace=workspace,
            handle=handle,
            event=event,
        )
    )


def run_native_fused_infer_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    scale: float,
    block_table: torch.Tensor,
    attention_mask: torch.Tensor,
    actual_seq_lengths_q: list[int],
    actual_seq_lengths_kv: list[int],
    block_size: int,
    output: torch.Tensor,
) -> None:
    """Run or capture the FIA task used by speculative verification."""
    softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
    args = {
        "query": query,
        "key": key_cache,
        "value": value_cache,
        "atten_mask": attention_mask,
        "block_table": block_table,
        "input_layout": "TND",
        "block_size": block_size,
        "actual_seq_lengths": actual_seq_lengths_q,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,
        "num_key_value_heads": num_kv_heads,
        "num_heads": num_heads,
        "scale": scale,
        "sparse_mode": 3,
        "pre_tokens": _FIA_INT_MAX,
        "next_tokens": _FIA_INT_MAX,
    }
    if _CAPTURED_TASKS is None:
        torch_npu.npu_fused_infer_attention_score.out(
            **args,
            out=[output, softmax_lse],
        )
        return

    assert _CAPTURED_FIA_WORKSPACES is not None
    workspace_key = (
        tuple(query.shape),
        tuple(key_cache.shape),
        query.dtype,
        num_kv_heads,
        num_heads,
        block_size,
    )
    workspace = _CAPTURED_FIA_WORKSPACES.get(workspace_key)
    if workspace is None:
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**args)
        _CAPTURED_FIA_WORKSPACES[workspace_key] = workspace
    stream = torch.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
    torch.npu.graph_task_group_begin(stream)
    torch_npu.npu_fused_infer_attention_score.out(
        **args,
        workspace=workspace,
        out=[output, softmax_lse],
    )
    handle = torch.npu.graph_task_group_end(stream)
    _CAPTURED_TASKS.append(
        NativeFusedInferAttentionGraphTask(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            block_table=block_table,
            attention_mask=attention_mask,
            output=output,
            softmax_lse=softmax_lse,
            block_size=block_size,
            workspace=workspace,
            handle=handle,
            event=event,
        )
    )


class NativeACLGraphRunner:
    """Capture native Qwen model decode graphs keyed by packed token count."""

    def __init__(
        self,
        model,
        enabled: bool = True,
        max_graph_tokens: int = 512,
        max_graph_entries: int = DEFAULT_MAX_ACLGRAPH_ENTRIES,
    ) -> None:
        if max_graph_entries <= 0:
            raise ValueError("Native PEARL max_graph_entries must be positive.")
        self.model = model
        self.enabled = enabled
        self.max_graph_tokens = max_graph_tokens
        self.max_graph_entries = max_graph_entries
        self.entries: dict[tuple[str, int], NativeACLGraphEntry] = {}
        self.draft_entries: dict[tuple[str, int], NativeDraftACLGraphEntry] = {}
        self.update_stream = torch.npu.Stream() if enabled else None
        self.capture_attempt_count = 0
        self.capture_count = 0
        self.replay_count = 0
        self.failed_capture_count = 0
        self.capacity_fallback_count = 0
        self.shape_fallback_count = 0
        self.disabled_entry_keys: set[tuple[str, int]] = set()
        self.expected_fia_batch_size: int | None = None

    def set_expected_fia_batch_size(self, batch_size: int) -> None:
        """Restrict FIA capture to the stable, full request batch."""
        if batch_size <= 0:
            raise ValueError("Native PEARL FIA batch size must be positive.")
        self.expected_fia_batch_size = batch_size

    def __call__(self, input_ids: torch.Tensor, positions: torch.Tensor, attention_metadata) -> torch.Tensor:
        return self._run(
            input_ids,
            positions,
            attention_metadata,
            output_kind="hidden",
            output_transform=None,
        )

    def run_greedy(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        vocabulary_size: int,
    ) -> torch.Tensor:
        return self._run(
            input_ids,
            positions,
            attention_metadata,
            output_kind=f"greedy:{vocabulary_size}",
            output_transform=lambda hidden_states: self.model.compute_greedy_tokens(
                hidden_states,
                vocabulary_size,
            ),
        )

    def run_draft_greedy(
        self,
        input_ids: torch.Tensor,
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
        vocabulary_size: int,
    ) -> torch.Tensor:
        """Replay all draft proposal steps inside one ACLGraph."""
        if not positions or len(positions) != len(attention_metadatas):
            raise ValueError("A draft ACLGraph requires matching non-empty step metadata.")
        if not self.enabled:
            return self._execute_draft(input_ids, positions, attention_metadatas, vocabulary_size)
        attention_modes = [metadata.use_fused_infer_attention for metadata in attention_metadatas]
        if any(attention_modes) != all(attention_modes):
            raise ValueError("Every step in a draft ACLGraph must use the same attention backend.")
        if self.expected_fia_batch_size is not None and input_ids.shape[0] != self.expected_fia_batch_size:
            self.shape_fallback_count += 1
            return self._execute_draft(input_ids, positions, attention_metadatas, vocabulary_size)
        if all(attention_modes):
            query_shape = ",".join(str(length) for length in attention_metadatas[0].actual_seq_lengths_q)
            attention_key = f"fia:{query_shape}"
        else:
            attention_key = "paged"
        entry_key = (
            f"draft-greedy:{vocabulary_size}|steps:{len(positions)}|{attention_key}",
            input_ids.shape[0],
        )
        if entry_key in self.disabled_entry_keys:
            return self._execute_draft(input_ids, positions, attention_metadatas, vocabulary_size)
        entry = self.draft_entries.get(entry_key)
        if entry is None:
            if len(self.entries) + len(self.draft_entries) >= self.max_graph_entries:
                self.capacity_fallback_count += 1
                return self._execute_draft(input_ids, positions, attention_metadatas, vocabulary_size)
            self.capture_attempt_count += 1
            return self._capture_draft(
                entry_key,
                input_ids,
                positions,
                attention_metadatas,
                vocabulary_size,
            )
        self._copy_draft_inputs(entry, input_ids, positions, attention_metadatas)
        assert self.update_stream is not None
        self.update_stream.wait_stream(torch.npu.current_stream())
        self._update_draft_attention_tasks(entry)
        entry.graph.replay()
        self.replay_count += 1
        return entry.output

    def _execute_draft(
        self,
        input_ids: torch.Tensor,
        positions: list[torch.Tensor] | tuple[torch.Tensor, ...],
        attention_metadatas: list[Any],
        vocabulary_size: int,
    ) -> torch.Tensor:
        outputs = []
        step_input = input_ids
        for step_positions, metadata in zip(positions, attention_metadatas):
            hidden_states = self.model(step_input, step_positions, metadata)
            step_input = self.model.compute_greedy_tokens(hidden_states, vocabulary_size)
            outputs.append(step_input)
        return torch.stack(outputs, dim=1)

    def _capture_draft(
        self,
        entry_key: tuple[str, int],
        input_ids: torch.Tensor,
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
        vocabulary_size: int,
    ) -> torch.Tensor:
        reference_output = self._execute_draft(
            input_ids,
            positions,
            attention_metadatas,
            vocabulary_size,
        ).clone()
        torch.npu.synchronize()
        captured_input_ids = input_ids.clone()
        captured_positions = tuple(value.clone() for value in positions)
        captured_metadatas = []
        for metadata in attention_metadatas:
            captured_request_block_tables = (
                metadata.request_block_tables.clone()
                if metadata.use_fused_infer_attention and metadata.request_block_tables is not None
                else None
            )
            captured_metadatas.append(
                type(metadata)(
                    slot_mapping=metadata.slot_mapping.clone(),
                    context_lens=metadata.context_lens.clone(),
                    block_tables=metadata.block_tables.clone(),
                    actual_seq_lengths_q=metadata.actual_seq_lengths_q,
                    sequence_lens=metadata.sequence_lens,
                    request_block_tables=captured_request_block_tables,
                    attention_mask=metadata.attention_mask,
                    use_fused_infer_attention=metadata.use_fused_infer_attention,
                )
            )
        graph = torch.npu.NPUGraph()
        with _collect_graph_tasks() as tasks, torch.npu.graph(graph):
            output = self._execute_draft(
                captured_input_ids,
                list(captured_positions),
                captured_metadatas,
                vocabulary_size,
            )
        if len(tasks) % len(captured_metadatas):
            raise RuntimeError("Draft ACLGraph attention tasks do not divide evenly across proposal steps.")
        tasks_per_step = len(tasks) // len(captured_metadatas)
        entry = NativeDraftACLGraphEntry(
            input_ids=captured_input_ids,
            positions=captured_positions,
            slot_mappings=tuple(metadata.slot_mapping for metadata in captured_metadatas),
            context_lens=tuple(metadata.context_lens for metadata in captured_metadatas),
            block_tables=tuple(metadata.block_tables for metadata in captured_metadatas),
            request_block_tables=tuple(metadata.request_block_tables for metadata in captured_metadatas),
            actual_seq_lengths_q=tuple(metadata.actual_seq_lengths_q for metadata in captured_metadatas),
            sequence_lens=tuple(metadata.sequence_lens for metadata in captured_metadatas),
            graph=graph,
            output=output,
            tasks=tasks,
            tasks_per_step=tasks_per_step,
        )
        self.draft_entries[entry_key] = entry
        torch.npu.current_stream().synchronize()
        self._update_draft_attention_tasks(entry)
        entry.graph.replay()
        torch.npu.synchronize()
        if not self._outputs_match(output, reference_output):
            self.draft_entries.pop(entry_key)
            self.disabled_entry_keys.add(entry_key)
            self.failed_capture_count += 1
            warnings.warn(
                "Native PEARL disabled a gamma-step draft ACLGraph whose replay did not match eager execution: "
                f"{entry_key!r}; {self._mismatch_summary(output, reference_output)}",
                RuntimeWarning,
                stacklevel=2,
            )
            return reference_output
        self.capture_count += 1
        self.replay_count += 1
        return output

    @staticmethod
    def _copy_draft_inputs(
        entry: NativeDraftACLGraphEntry,
        input_ids: torch.Tensor,
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
    ) -> None:
        entry.input_ids.copy_(input_ids)
        for step, (step_positions, metadata) in enumerate(zip(positions, attention_metadatas)):
            entry.positions[step].copy_(step_positions)
            entry.slot_mappings[step].copy_(metadata.slot_mapping)
            if entry.request_block_tables[step] is not None:
                assert metadata.request_block_tables is not None
                entry.request_block_tables[step].copy_(metadata.request_block_tables)
            else:
                entry.context_lens[step].copy_(metadata.context_lens)
                entry.block_tables[step].copy_(metadata.block_tables)
        entry.actual_seq_lengths_q = tuple(metadata.actual_seq_lengths_q for metadata in attention_metadatas)
        entry.sequence_lens = tuple(metadata.sequence_lens for metadata in attention_metadatas)

    def _run(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        *,
        output_kind: str,
        output_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> torch.Tensor:
        num_tokens = input_ids.shape[0]
        if not self.enabled or num_tokens > self.max_graph_tokens:
            return self._execute(input_ids, positions, attention_metadata, output_transform)
        if attention_metadata.use_fused_infer_attention:
            if not self._is_reusable_fia_shape(attention_metadata.actual_seq_lengths_q):
                self.shape_fallback_count += 1
                return self._execute(input_ids, positions, attention_metadata, output_transform)
            query_shape = ",".join(str(length) for length in attention_metadata.actual_seq_lengths_q)
            output_kind = f"{output_kind}|fia:{query_shape}"
            capture_size = num_tokens
        else:
            capture_size = min(
                (size for kind, size in self.entries if kind == output_kind and size >= num_tokens),
                default=num_tokens,
            )
        padded_inputs = self._pad_inputs(input_ids, positions, attention_metadata, capture_size)
        padded_input_ids, padded_positions, padded_metadata = padded_inputs
        entry_key = (output_kind, capture_size)
        if entry_key in self.disabled_entry_keys:
            return self._execute(input_ids, positions, attention_metadata, output_transform)
        entry = self.entries.get(entry_key)
        if entry is None:
            if len(self.entries) >= self.max_graph_entries or self.capture_attempt_count >= self.max_graph_entries:
                self.capacity_fallback_count += 1
                return self._execute(input_ids, positions, attention_metadata, output_transform)
            self.capture_attempt_count += 1
            output = self._capture(
                entry_key,
                padded_input_ids,
                padded_positions,
                padded_metadata,
                output_transform,
            )
            return output[:num_tokens]
        self._copy_inputs(entry, padded_input_ids, padded_positions, padded_metadata)
        # Order task updates after the previous replay and the current input
        # copies without blocking the host. Each captured attention task waits
        # on the ExternalEvent recorded by its update before it executes.
        assert self.update_stream is not None
        self.update_stream.wait_stream(torch.npu.current_stream())
        self._update_attention_tasks(entry)
        entry.graph.replay()
        self.replay_count += 1
        if not entry.runtime_validated:
            torch.npu.synchronize()
            graph_output = entry.output[:num_tokens].clone()
            reference_output = self._execute(
                input_ids,
                positions,
                attention_metadata,
                output_transform,
            )
            torch.npu.synchronize()
            if not self._outputs_match(graph_output, reference_output):
                self.entries.pop(entry_key)
                self.disabled_entry_keys.add(entry_key)
                self.failed_capture_count += 1
                del entry
                warnings.warn(
                    "Native PEARL disabled an ACLGraph entry whose runtime replay did not match eager execution: "
                    f"{entry_key!r}; {self._mismatch_summary(graph_output, reference_output)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return reference_output
            entry.runtime_validated = True
        return entry.output[:num_tokens]

    def _is_reusable_fia_shape(self, actual_seq_lengths_q: tuple[int, ...]) -> bool:
        if not actual_seq_lengths_q:
            return False
        if self.expected_fia_batch_size is not None and len(actual_seq_lengths_q) != self.expected_fia_batch_size:
            return False
        segment_lengths = (actual_seq_lengths_q[0],) + tuple(
            current - previous for previous, current in zip(actual_seq_lengths_q, actual_seq_lengths_q[1:])
        )
        # The full request set has stable KV ownership even when target
        # verification mixes one-token pre-verify rows with gamma-token rows.
        # Its exact cumulative shape is part of the graph key. Dynamic tails
        # remain eager because their request count fails the guard above.
        return all(length > 0 for length in segment_lengths)

    def _execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        output_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> torch.Tensor:
        output = self.model(input_ids, positions, attention_metadata)
        return output if output_transform is None else output_transform(output)

    @staticmethod
    def _pad_inputs(input_ids, positions, attention_metadata, capture_size: int):
        pad_size = capture_size - input_ids.shape[0]
        if pad_size == 0:
            return input_ids, positions, attention_metadata
        padded_input_ids = torch.zeros(capture_size, dtype=input_ids.dtype, device=input_ids.device)
        padded_positions = torch.zeros(capture_size, dtype=positions.dtype, device=positions.device)
        padded_slot_mapping = torch.full(
            (capture_size,),
            -1,
            dtype=attention_metadata.slot_mapping.dtype,
            device=attention_metadata.slot_mapping.device,
        )
        padded_context_lens = torch.zeros(capture_size, dtype=attention_metadata.context_lens.dtype)
        padded_block_tables = torch.zeros(
            (capture_size, attention_metadata.block_tables.shape[1]),
            dtype=attention_metadata.block_tables.dtype,
            device=attention_metadata.block_tables.device,
        )
        num_tokens = input_ids.shape[0]
        padded_input_ids[:num_tokens].copy_(input_ids)
        padded_positions[:num_tokens].copy_(positions)
        padded_slot_mapping[:num_tokens].copy_(attention_metadata.slot_mapping)
        padded_context_lens[:num_tokens].copy_(attention_metadata.context_lens)
        padded_block_tables[:num_tokens].copy_(attention_metadata.block_tables)
        padded_metadata = type(attention_metadata)(
            slot_mapping=padded_slot_mapping,
            context_lens=padded_context_lens,
            block_tables=padded_block_tables,
        )
        return padded_input_ids, padded_positions, padded_metadata

    def _capture(
        self,
        entry_key: tuple[str, int],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        output_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> torch.Tensor:
        # Warm up allocations and collectives before entering stream capture.
        reference_output = self._execute(
            input_ids,
            positions,
            attention_metadata,
            output_transform,
        ).clone()
        torch.npu.synchronize()
        captured_input_ids = input_ids.clone()
        captured_positions = positions.clone()
        captured_slot_mapping = attention_metadata.slot_mapping.clone()
        captured_context_lens = attention_metadata.context_lens.clone()
        captured_block_tables = attention_metadata.block_tables.clone()
        captured_request_block_tables = (
            attention_metadata.request_block_tables.clone()
            if attention_metadata.use_fused_infer_attention and attention_metadata.request_block_tables is not None
            else None
        )
        captured_metadata = type(attention_metadata)(
            slot_mapping=captured_slot_mapping,
            context_lens=captured_context_lens,
            block_tables=captured_block_tables,
            actual_seq_lengths_q=attention_metadata.actual_seq_lengths_q,
            sequence_lens=attention_metadata.sequence_lens,
            request_block_tables=captured_request_block_tables,
            attention_mask=attention_metadata.attention_mask,
            use_fused_infer_attention=attention_metadata.use_fused_infer_attention,
        )
        graph = torch.npu.NPUGraph()
        with _collect_graph_tasks() as tasks, torch.npu.graph(graph):
            output = self._execute(
                captured_input_ids,
                captured_positions,
                captured_metadata,
                output_transform,
            )
        entry = NativeACLGraphEntry(
            input_ids=captured_input_ids,
            positions=captured_positions,
            slot_mapping=captured_slot_mapping,
            context_lens=captured_context_lens,
            block_tables=captured_block_tables,
            request_block_tables=captured_request_block_tables,
            actual_seq_lengths_q=attention_metadata.actual_seq_lengths_q,
            sequence_lens=attention_metadata.sequence_lens,
            graph=graph,
            output=output,
            tasks=tasks,
        )
        self.entries[entry_key] = entry
        # Capture execution only records the graph; CANN does not guarantee
        # that its output buffers contain a valid inference result. Rebind the
        # host metadata and replay once before serving the triggering request.
        torch.npu.current_stream().synchronize()
        self._update_attention_tasks(entry)
        entry.graph.replay()
        torch.npu.synchronize()
        graph_matches_eager = self._outputs_match(output, reference_output)
        if not graph_matches_eager:
            self.entries.pop(entry_key)
            self.disabled_entry_keys.add(entry_key)
            self.failed_capture_count += 1
            warnings.warn(
                "Native PEARL disabled an ACLGraph entry whose first replay did not match eager execution: "
                f"{entry_key!r}; {self._mismatch_summary(output, reference_output)}",
                RuntimeWarning,
                stacklevel=2,
            )
            return reference_output
        self.capture_count += 1
        self.replay_count += 1
        return output

    @staticmethod
    def _outputs_match(graph_output: torch.Tensor, reference_output: torch.Tensor) -> bool:
        if graph_output.dtype.is_floating_point:
            return torch.allclose(graph_output, reference_output, rtol=1e-3, atol=1e-3)
        mismatch_count = torch.count_nonzero(graph_output != reference_output).item()
        allowed_mismatches = max(
            1,
            math.ceil(reference_output.numel() * MAX_GREEDY_REPLAY_EAGER_DIVERGENCE),
        )
        return mismatch_count <= allowed_mismatches

    @staticmethod
    def _mismatch_summary(graph_output: torch.Tensor, reference_output: torch.Tensor) -> str:
        if graph_output.dtype.is_floating_point:
            max_error = (graph_output.float() - reference_output.float()).abs().max().item()
            return f"shape={tuple(graph_output.shape)}, max_abs_error={max_error:g}"
        mismatches = torch.count_nonzero(graph_output != reference_output).item()
        return (
            f"mismatches={mismatches}/{reference_output.numel()}, "
            f"eager={reference_output.flatten()[:16].cpu().tolist()}, "
            f"graph={graph_output.flatten()[:16].cpu().tolist()}"
        )

    @staticmethod
    def _copy_inputs(entry: NativeACLGraphEntry, input_ids, positions, attention_metadata) -> None:
        entry.input_ids.copy_(input_ids)
        entry.positions.copy_(positions)
        entry.slot_mapping.copy_(attention_metadata.slot_mapping)
        if entry.request_block_tables is not None:
            assert attention_metadata.request_block_tables is not None
            entry.request_block_tables.copy_(attention_metadata.request_block_tables)
        else:
            entry.context_lens.copy_(attention_metadata.context_lens)
            entry.block_tables.copy_(attention_metadata.block_tables)
        entry.actual_seq_lengths_q = attention_metadata.actual_seq_lengths_q
        entry.sequence_lens = attention_metadata.sequence_lens

    def _update_attention_tasks(self, entry: NativeACLGraphEntry) -> None:
        task_lengths = [(entry.actual_seq_lengths_q, entry.sequence_lens)] * len(entry.tasks)
        self._update_attention_task_list(entry.tasks, task_lengths)

    def _update_draft_attention_tasks(self, entry: NativeDraftACLGraphEntry) -> None:
        task_lengths = [
            lengths
            for lengths in zip(entry.actual_seq_lengths_q, entry.sequence_lens)
            for _ in range(entry.tasks_per_step)
        ]
        self._update_attention_task_list(entry.tasks, task_lengths)

    def _update_attention_task_list(
        self,
        tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask],
        task_lengths: list[tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> None:
        assert self.update_stream is not None
        if len(tasks) != len(task_lengths):
            raise RuntimeError("ACLGraph attention tasks and replay metadata differ in length.")
        fia_length_lists: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[list[int], list[int]],
        ] = {}
        with torch.npu.stream(self.update_stream):
            for task, (actual_seq_lengths_q, sequence_lens) in zip(tasks, task_lengths):
                if isinstance(task, NativeFusedInferAttentionGraphTask):
                    length_key = (actual_seq_lengths_q, sequence_lens)
                    length_lists = fia_length_lists.get(length_key)
                    if length_lists is None:
                        length_lists = (
                            list(actual_seq_lengths_q),
                            list(sequence_lens),
                        )
                        fia_length_lists[length_key] = length_lists
                    self._update_fused_infer_attention_task(
                        task,
                        *length_lists,
                    )
                    continue
                torch.npu.graph_task_update_begin(self.update_stream, task.handle)
                torch_npu._npu_paged_attention(
                    query=task.query,
                    key_cache=task.key_cache,
                    value_cache=task.value_cache,
                    num_kv_heads=task.num_kv_heads,
                    num_heads=task.num_heads,
                    scale_value=task.scale,
                    block_table=task.block_table,
                    context_lens=task.context_lens,
                    out=task.output,
                    workspace=task.workspace,
                )
                torch.npu.graph_task_update_end(self.update_stream)
                task.event.record(self.update_stream)

    def _update_fused_infer_attention_task(
        self,
        task: NativeFusedInferAttentionGraphTask,
        actual_seq_lengths_q: list[int],
        sequence_lens: list[int],
    ) -> None:
        args = {
            "query": task.query,
            "key": task.key_cache,
            "value": task.value_cache,
            "atten_mask": task.attention_mask,
            "block_table": task.block_table,
            "input_layout": "TND",
            "block_size": task.block_size,
            "actual_seq_lengths": actual_seq_lengths_q,
            "actual_seq_lengths_kv": sequence_lens,
            "num_key_value_heads": task.num_kv_heads,
            "num_heads": task.num_heads,
            "scale": task.scale,
            "sparse_mode": 3,
            "next_tokens": 0,
        }
        torch.npu.graph_task_update_begin(self.update_stream, task.handle)
        torch_npu.npu_fused_infer_attention_score.out(
            **args,
            workspace=task.workspace,
            out=[task.output, task.softmax_lse],
        )
        torch.npu.graph_task_update_end(self.update_stream)
        task.event.record(self.update_stream)

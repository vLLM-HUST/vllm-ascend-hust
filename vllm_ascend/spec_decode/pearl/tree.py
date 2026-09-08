# SPDX-License-Identifier: Apache-2.0
"""Tree-aware SpecRhythm planning and Ascend attention-mask construction.

The regular vLLM tree verifier owns the final token comparison.  This module
owns the parts that are specific to PEARL scheduling: translating a scalar
roofline allocation into a bounded tree and selecting candidates without
breaking parent-prefix dependencies.  Masks use the vLLM convention where
``True`` means blocked attention and ``False`` means visible attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmBudgetPlan


def make_spine_first_parents(
    width: int, depth: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return contiguous spine-first parent indices for a uniform tree."""

    if width < 1 or depth < 1:
        raise ValueError("tree width and depth must be positive")
    parents = [-1] + list(range(depth - 1))
    for level in range(depth):
        parent = -1 if level == 0 else level - 1
        parents.extend([parent] * (width - 1))
    return torch.tensor(parents, dtype=torch.int32, device=device)


def _node_levels(width: int, depth: int) -> list[int]:
    return list(range(1, depth + 1)) + [level for level in range(1, depth + 1) for _ in range(width - 1)]


def build_tree_attention_mask(
    width: int,
    depth: int,
    prefix_len: int,
    max_model_len: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build one root-plus-tree causal mask for CANN attention.

    The returned shape is ``[1 + width * depth, max_model_len]``.  Prefix
    tokens are visible to every query, the virtual root sees itself, and each
    draft node sees only its prefix, root and ancestor chain.  Invalid columns
    remain blocked so the tensor can be reused in a fixed-shape graph bucket.
    """

    if prefix_len < 0 or max_model_len <= 0:
        raise ValueError("prefix_len must be non-negative and max_model_len positive")
    if prefix_len + depth >= max_model_len:
        raise ValueError("tree query positions exceed max_model_len")
    parents = make_spine_first_parents(width, depth).tolist()
    levels = _node_levels(width, depth)
    query_len = 1 + width * depth
    mask = torch.ones(
        (query_len, max_model_len), dtype=torch.bool, device=device
    )
    mask[:, :prefix_len] = False
    mask[0, prefix_len] = False
    for node, parent in enumerate(parents):
        row = node + 1
        mask[row, prefix_len] = False
        current = node
        while current >= 0:
            mask[row, prefix_len + current + 1] = False
            current = parents[current]
    # ``levels`` is intentionally computed above as a validation of the
    # spine-first layout; positions are exposed by TreeSpeculationPlan.
    assert len(levels) == len(parents)
    return mask


@dataclass(frozen=True)
class TreeSpeculationPlan:
    """Fixed-shape tree metadata consumed by an Ascend target worker."""

    width: int
    depth: int
    candidate_budget: int
    prefix_len: int
    max_model_len: int
    parent_indices: torch.Tensor
    positions: torch.Tensor
    attention_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.candidate_budget < 1 or self.candidate_budget > self.width * self.depth:
            raise ValueError("candidate_budget must fit inside the configured tree")
        if self.parent_indices.numel() != self.width * self.depth:
            raise ValueError("parent_indices do not match tree shape")
        if self.positions.numel() != self.width * self.depth + 1:
            raise ValueError("positions must include root and every tree node")
        if tuple(self.attention_mask.shape) != (self.width * self.depth + 1, self.max_model_len):
            raise ValueError("attention_mask does not match tree shape")


def build_tree_speculation_plan(
    width: int,
    depth: int,
    prefix_len: int,
    max_model_len: int,
    *,
    candidate_budget: int | None = None,
    device: torch.device | str | None = None,
) -> TreeSpeculationPlan:
    """Construct parent, position and mask tensors for one request."""

    if width < 1 or depth < 1:
        raise ValueError("tree width and depth must be positive")
    budget = width * depth if candidate_budget is None else int(candidate_budget)
    if not 1 <= budget <= width * depth:
        raise ValueError("candidate_budget must be in [1, width * depth]")
    parents = make_spine_first_parents(width, depth, device=device)
    levels = torch.tensor(
        [0] + _node_levels(width, depth), dtype=torch.int32, device=device
    )
    positions = levels + int(prefix_len)
    mask = build_tree_attention_mask(
        width, depth, prefix_len, max_model_len, device=device
    )
    return TreeSpeculationPlan(
        width=width,
        depth=depth,
        candidate_budget=budget,
        prefix_len=int(prefix_len),
        max_model_len=int(max_model_len),
        parent_indices=parents,
        positions=positions,
        attention_mask=mask,
    )


@dataclass(frozen=True)
class TreeCandidateSelection:
    """Top-scoring tree candidates with all required ancestors included."""

    indices: torch.Tensor
    token_ids: torch.Tensor


def select_tree_candidates(
    candidate_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    scores: torch.Tensor,
    budget: int,
) -> TreeCandidateSelection:
    """Select a dependency-closed candidate subset deterministically.

    ``scores`` is one scalar per node.  A high-scoring descendant is admitted
    only when its complete ancestor chain fits in ``budget``; selected indices
    are returned in original topological order for direct target packing.
    """

    if candidate_token_ids.ndim != 1 or parent_indices.shape != candidate_token_ids.shape:
        raise ValueError("candidate_token_ids and parent_indices must be aligned 1-D tensors")
    if scores.ndim != 1 or scores.numel() != candidate_token_ids.numel():
        raise ValueError("scores must contain one value per candidate")
    if not 1 <= int(budget) <= candidate_token_ids.numel():
        raise ValueError("budget must be in [1, number of candidates]")
    parents = [int(value) for value in parent_indices.detach().cpu().tolist()]
    count = len(parents)
    for index, parent in enumerate(parents):
        if parent < -1 or parent >= index:
            raise ValueError("parent indices must reference an earlier node or the virtual root")
    score_values = scores.detach().float().cpu().tolist()
    order = sorted(range(count), key=lambda index: (-score_values[index], index))
    selected: set[int] = set()
    for index in order:
        chain: list[int] = []
        current = index
        while current >= 0 and current not in selected:
            chain.append(current)
            current = parents[current]
        if len(selected) + len(chain) <= int(budget):
            selected.update(chain)
    selected_indices = torch.tensor(
        sorted(selected), dtype=torch.long, device=candidate_token_ids.device
    )
    return TreeCandidateSelection(
        indices=selected_indices,
        token_ids=torch.index_select(candidate_token_ids, 0, selected_indices),
    )


def tree_budget_from_spec_rhythm(
    plan: "SpecRhythmBudgetPlan",
    request_index: int,
    width: int,
    max_depth: int,
) -> int:
    """Map one scalar SpecRhythm request allocation to tree node capacity."""

    if width < 1 or max_depth < 1:
        raise ValueError("tree width and max_depth must be positive")
    index = int(request_index)
    allocated = int(plan.normal_budgets.get(index, 0)) + int(plan.eager_budgets.get(index, 0))
    return max(0, min(allocated, width * max_depth))


class SpecRhythmTreeCoordinator:
    """Turn per-request SpecRhythm budgets into fixed-shape tree plans."""

    def __init__(self, *, width: int, max_depth: int) -> None:
        if width < 1 or max_depth < 1:
            raise ValueError("tree width and max_depth must be positive")
        self.width = int(width)
        self.max_depth = int(max_depth)

    def for_request(
        self,
        budget_plan: "SpecRhythmBudgetPlan",
        request_index: int,
        *,
        prefix_len: int,
        max_model_len: int,
        device: torch.device | str | None = None,
    ) -> TreeSpeculationPlan | None:
        """Build a request tree, returning ``None`` for an unallocated row."""

        budget = tree_budget_from_spec_rhythm(
            budget_plan, request_index, self.width, self.max_depth
        )
        if budget == 0:
            return None
        depth = min(self.max_depth, max(1, math.ceil(budget / self.width)))
        return build_tree_speculation_plan(
            self.width,
            depth,
            prefix_len,
            max_model_len,
            candidate_budget=min(budget, self.width * depth),
            device=device,
        )


__all__ = [
    "TreeCandidateSelection",
    "TreeSpeculationPlan",
    "SpecRhythmTreeCoordinator",
    "build_tree_attention_mask",
    "build_tree_speculation_plan",
    "make_spine_first_parents",
    "select_tree_candidates",
    "tree_budget_from_spec_rhythm",
]

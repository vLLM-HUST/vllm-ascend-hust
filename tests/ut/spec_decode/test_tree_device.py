# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.spec_decode.tree import (
    make_spine_first_tree_parents,
    verify_greedy_tree_batch,
)
from vllm_ascend.spec_decode.tree_kv import verify_greedy_tree_device


def test_verify_greedy_tree_device_matches_reference():
    width = 2
    depth = 3
    num_nodes = width * depth
    drafts = torch.tensor(
        [
            [10, 11, 12, 20, 21, 22],
            [30, 31, 32, 40, 41, 42],
            [50, 51, 52, 60, 61, 62],
        ]
    )
    parents = make_spine_first_tree_parents(width, depth).repeat(3, 1)
    targets = torch.tensor(
        [
            [10, 11, 12, 91, 92, 93],
            [40, 81, 82, 83, 84, 85],
            [99, 71, 72, 73, 74, 75],
        ]
    )
    bonuses = torch.tensor([100, 101, 102])

    expected = verify_greedy_tree_batch(
        drafts.flatten(),
        parents.flatten(),
        [num_nodes] * 3,
        targets.flatten(),
        bonuses,
        depth,
    )
    output, accepted = verify_greedy_tree_device(
        drafts.flatten(),
        parents.flatten(),
        targets.flatten(),
        bonuses,
        3,
        num_nodes,
        depth,
    )

    assert torch.equal(output, expected.token_ids)
    assert torch.equal(accepted, expected.accepted_node_indices)

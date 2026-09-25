# SPDX-License-Identifier: Apache-2.0
"""Final-stage sampling feedback for hybrid pipeline recurrent state."""

import torch


def sync_sampled_tokens(pp, sampled_token_ids, num_reqs, num_spec_tokens, device):
    """Broadcast actual sampled IDs padded to a topology-known fixed width.

    Padding is -1, the same invalid-token convention consumed by the core
    Mamba postprocess. This preserves rejection counts, including rows with
    no accepted tokens, without a device-to-host transfer or shape handshake.
    Both native and policy arms use this experimental common runtime glue.
    """
    width = num_spec_tokens + 1
    tokens = torch.full((num_reqs, width), -1, dtype=torch.int32, device=device)
    if pp.is_last_rank:
        if (
            sampled_token_ids is None
            or sampled_token_ids.ndim != 2
            or sampled_token_ids.shape[0] != num_reqs
            or not 1 <= sampled_token_ids.shape[1] <= width
        ):
            raise ValueError("Unexpected PP hybrid sampled-token shape")
        tokens[:, : sampled_token_ids.shape[1]].copy_(sampled_token_ids)
    elif sampled_token_ids is not None:
        raise ValueError("Only the final PP rank supplies sampled tokens")
    torch.distributed.broadcast(tokens, src=pp.last_rank, group=pp.device_group)
    return tokens


def sampled_tail_and_counts(sampled):
    """Return next-input token and valid counts without reading device scalars."""
    counts = (sampled != -1).sum(dim=1)
    tail = sampled.gather(1, (counts - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
    return tail, counts


def retain_request_feedback(previous, request_ids, accepted_counts, reset_ids):
    """Preserve accepted-state offsets when persistent batch rows are removed."""
    updated = dict(previous)
    for req_id, count in zip(request_ids, accepted_counts):
        updated[req_id] = int(count)
    for req_id in reset_ids:
        updated.pop(req_id, None)
    return updated


def scheduled_draft_inputs(request_ids, scheduled_drafts, cumulative_tokens):
    """Map authoritative scheduler drafts into this batch's flattened inputs."""
    if len(request_ids) != len(cumulative_tokens):
        raise ValueError("PP draft row dimensions mismatch")
    indices, tokens = [], []
    start = 0
    for req_id, end in zip(request_ids, cumulative_tokens):
        end = int(end)
        drafts = scheduled_drafts.get(req_id, ())
        if len(drafts) > end - start or any(token < 0 for token in drafts):
            raise ValueError("PP draft inputs require actual scheduled token IDs")
        indices.extend(range(end - len(drafts), end))
        tokens.extend(drafts)
        start = end
    return indices, tokens

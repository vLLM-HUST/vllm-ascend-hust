import torch

from vllm_ascend.sample.sampler import _apply_top_k_top_p_pytorch


def apply_top_k_top_p_npu(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    has_topp_only_rows: bool = True,
) -> torch.Tensor:
    # The Host uses has_topp_only_rows to select Triton specializations. The
    # Ascend sort-and-mask implementation handles mixed rows directly.
    del has_topp_only_rows
    # NOTE: During the warmup stage, if both k and p are None, the NPU op is skipped.
    # As a result, its workspace is not reserved during warmup, which may lead to an OOM
    # when the op is executed later.
    # TODO: Fix the workspace reservation issue in the future.
    if k is None and p is None:
        return logits
    # use pytorch ops
    return _apply_top_k_top_p_pytorch(logits, k, p)

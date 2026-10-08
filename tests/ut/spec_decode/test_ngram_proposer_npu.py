"""Unit tests for AscendNgramProposerNPU."""

import inspect

from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU

from vllm_ascend.spec_decode.ngram_proposer_npu import AscendNgramProposerNPU


def test_propose_signature_matches_parent():
    """Verify that propose() has the same signature as the parent class."""
    child_sig = inspect.signature(AscendNgramProposerNPU.propose)
    parent_sig = inspect.signature(NgramProposerGPU.propose)

    child_params = list(child_sig.parameters.keys())
    parent_params = list(parent_sig.parameters.keys())

    assert child_params == parent_params, (
        f"Signature mismatch:\n"
        f"  AscendNgramProposerNPU.propose params: {child_params}\n"
        f"  NgramProposerGPU.propose params:       {parent_params}\n"
        f"\n"
        f"The NPU proposer must accept the same parameters as the parent.\n"
        f"Missing: {set(parent_params) - set(child_params)}"
    )


def test_propose_inherits_parent_implementation():
    """The NPU proposer must not shadow the working parent implementation."""
    assert "propose" not in AscendNgramProposerNPU.__dict__
    assert AscendNgramProposerNPU.propose is NgramProposerGPU.propose

import pytest

from vllm_ascend.spec_decode.medusa_proposer import (
    _MEDUSA_IMPORT_ERROR,
    AscendMedusaProposer,
)


def test_removed_host_medusa_fails_only_when_selected() -> None:
    if _MEDUSA_IMPORT_ERROR is None:
        pytest.skip("host vLLM still provides Medusa")

    with pytest.raises(RuntimeError, match="Medusa speculative decoding is unavailable"):
        AscendMedusaProposer(None, None)

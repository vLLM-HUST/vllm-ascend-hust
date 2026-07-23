#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Unit tests for internal Sim-LLM control-plane helpers."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vllm_ascend.simllm.utils import (
    cumsum_to_ranges,
    resolve_input_embedding_dim,
    resolve_input_embedding_layer,
    tensor_to_float_list,
    tensor_to_int_list,
    tensor_to_int_matrix,
)


class _Nested:
    hidden_size: int
    get_input_embeddings: Any
    model: _Nested
    embed_tokens: Any


class _CallableEmbeddingModel:
    def __init__(self, dim: int = 32):
        self.config = _Nested()
        self.config.hidden_size = dim

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones(input_ids.shape[0], self.config.hidden_size)


class _InputIdGetterModel:
    def __init__(self, dim: int = 32):
        self.config = _Nested()
        self.config.hidden_size = dim

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones(input_ids.shape[0], self.config.hidden_size)


def test_tensor_to_int_list_empty_tensor():
    assert tensor_to_int_list(torch.tensor([], dtype=torch.int64)) == []


def test_tensor_to_int_list_flattens_tensor():
    values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    assert tensor_to_int_list(values) == [1, 2, 3, 4]


def test_tensor_to_float_list_flattens_tensor():
    values = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
    assert tensor_to_float_list(values) == [0.25, 0.5]


def test_tensor_to_int_matrix_requires_rank_2():
    with pytest.raises(ValueError, match="rank-2"):
        tensor_to_int_matrix(torch.tensor([1, 2, 3], dtype=torch.int64))


def test_tensor_to_int_matrix_materializes_rows():
    values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    assert tensor_to_int_matrix(values) == [[1, 2], [3, 4]]


def test_cumsum_to_ranges():
    qsl = torch.tensor([0, 4, 9, 12], dtype=torch.int64)
    assert cumsum_to_ranges(qsl) == [(0, 4), (4, 9), (9, 12)]


def test_resolve_input_embedding_layer_from_getter():
    embedding = torch.nn.Embedding(16, 32)
    model = _Nested()
    model.get_input_embeddings = lambda: embedding
    assert resolve_input_embedding_layer(model) is embedding


def test_resolve_input_embedding_layer_from_vllm_qwen_path():
    embedding = torch.nn.Embedding(16, 32)
    model = _Nested()
    model.model = _Nested()
    model.model.embed_tokens = embedding
    assert resolve_input_embedding_layer(model) is embedding


def test_resolve_input_embedding_layer_from_vllm_callable():
    model = _CallableEmbeddingModel(48)
    layer = resolve_input_embedding_layer(model)
    output = layer(torch.arange(3))
    assert output.shape == (3, 48)


def test_resolve_input_embedding_layer_from_input_id_getter():
    model = _InputIdGetterModel(48)
    layer = resolve_input_embedding_layer(model)
    output = layer(torch.arange(3))
    assert output.shape == (3, 48)


def test_resolve_input_embedding_layer_from_deep_wrapper_path():
    embedding = torch.nn.Embedding(16, 32)
    model = _Nested()
    model.model = _Nested()
    model.model.model = _Nested()
    model.model.model.embed_tokens = embedding
    assert resolve_input_embedding_layer(model) is embedding


def test_resolve_input_embedding_layer_requires_real_weight():
    model = _Nested()
    model.model = _Nested()
    model.model.embed_tokens = _Nested()
    with pytest.raises(AttributeError, match="token embedding layer"):
        resolve_input_embedding_layer(model)


def test_resolve_input_embedding_dim_from_weight():
    embedding = torch.nn.Embedding(16, 32)
    model = _Nested()
    model.model = _Nested()
    model.model.embed_tokens = embedding
    assert resolve_input_embedding_dim(model) == 32


def test_resolve_input_embedding_dim_from_config():
    model = _CallableEmbeddingModel(48)
    assert resolve_input_embedding_dim(model) == 48

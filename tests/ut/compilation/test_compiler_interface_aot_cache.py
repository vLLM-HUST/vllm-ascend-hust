#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import torch

from vllm_ascend.compilation.compiler_interface import (
    _disable_pytorch_aot_cache_for_non_inductor_backend,
    fusion_pass_compile,
)


def test_non_inductor_aot_cache_settings_are_scoped():
    dynamo_config = torch._dynamo.config
    functorch_config = torch._functorch.config
    original = {
        "caching_precompile": dynamo_config.caching_precompile,
        "bundled_autograd_cache": functorch_config.bundled_autograd_cache,
        "force_autograd_cache": functorch_config.force_autograd_cache,
        "enable_autograd_cache": functorch_config.enable_autograd_cache,
    }

    with (
        dynamo_config.patch(caching_precompile=True),
        functorch_config.patch(
            bundled_autograd_cache=True,
            force_autograd_cache=True,
            enable_autograd_cache=True,
        ),
    ):
        with _disable_pytorch_aot_cache_for_non_inductor_backend():
            assert dynamo_config.caching_precompile is False
            assert functorch_config.bundled_autograd_cache is False
            assert functorch_config.force_autograd_cache is False
            assert functorch_config.enable_autograd_cache is False

        assert dynamo_config.caching_precompile is True
        assert functorch_config.bundled_autograd_cache is True
        assert functorch_config.force_autograd_cache is True
        assert functorch_config.enable_autograd_cache is True

    assert dynamo_config.caching_precompile == original["caching_precompile"]
    assert functorch_config.bundled_autograd_cache == original["bundled_autograd_cache"]
    assert functorch_config.force_autograd_cache == original["force_autograd_cache"]
    assert functorch_config.enable_autograd_cache == original["enable_autograd_cache"]


def test_fusion_pass_compiles_without_inductor_aot_cache():
    dynamo_config = torch._dynamo.config
    functorch_config = torch._functorch.config

    def check_compile_settings(**_kwargs):
        assert dynamo_config.caching_precompile is False
        assert functorch_config.bundled_autograd_cache is False
        assert functorch_config.force_autograd_cache is False
        assert functorch_config.enable_autograd_cache is False
        return object()

    with (
        dynamo_config.patch(caching_precompile=True),
        functorch_config.patch(
            bundled_autograd_cache=True,
            force_autograd_cache=True,
            enable_autograd_cache=True,
        ),
        patch("vllm_ascend.compilation.compiler_interface.compile_fx", side_effect=check_compile_settings),
    ):
        compiled, handle = fusion_pass_compile(object(), [], {}, None)
        assert compiled is not None
        assert handle is None
        assert dynamo_config.caching_precompile is True
        assert functorch_config.bundled_autograd_cache is True

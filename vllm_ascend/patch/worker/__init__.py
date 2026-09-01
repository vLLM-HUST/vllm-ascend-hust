#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

import importlib

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.utils import is_310p, vllm_version_is

# The v2 model runner is intentionally NOT made compatible with the v0.23.0
# release. vLLM v0.23.0 and the verified main commit are diverged, and the v2
# worker patches target main-only APIs; rather than maintain a separate v0.23.0
# compatibility path we keep v2 main-only. With v0.23.0 installed this flag is
# False, so none of the patch_v2.* / routed-experts-capture patches below are
# imported and the v2 worker stays dormant (the release uses the v1 runner).
if vllm_version_is("0.23.0"):
    _V2_MODEL_RUNNER_SUPPORTED = False
else:
    _V2_MODEL_RUNNER_SUPPORTED = True

logger = init_logger(__name__)


def _import_optional_patch(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in {module_name, "torchvision"}:
            raise
    except ImportError as exc:
        logger.warning(
            "Skipping optional worker patch %s because its target API is unavailable: %s",
            module_name,
            exc,
        )


if HAS_TRITON:
    import vllm_ascend.patch.worker.patch_triton

    if _V2_MODEL_RUNNER_SUPPORTED:
        import vllm_ascend.patch.worker.patch_v2.patch_triton  # noqa


import vllm_ascend.patch.worker.patch_weight_utils  # noqa
import vllm_ascend.patch.worker.patch_distributed  # noqa
import vllm_ascend.patch.worker.patch_minimax_m2  # noqa
import vllm_ascend.patch.worker.patch_minimax_m2_linear_attn  # noqa
import vllm_ascend.patch.worker.patch_mamba_utils  # noqa
import vllm_ascend.patch.worker.patch_qwen3_next_mtp  # noqa

if not is_310p():
    _import_optional_patch("vllm_ascend.patch.worker.patch_qwen3_5")
    _import_optional_patch("vllm_ascend.patch.worker.patch_gdn_attn")
    import vllm_ascend.patch.worker.patch_qwen3_dflash  # noqa

    _import_optional_patch("vllm_ascend.patch.worker.patch_qwen3vl")
else:
    import vllm_ascend.patch.worker.patch_idex_310  # noqa
import vllm_ascend.patch.worker.patch_rejection_sampler  # noqa

# torchair/npugraph_ex is only available on NPU; silently skip when missing
# so that CPU-only environments (e.g. UT runners without torch_npu) can still
# import this module without crashing.
try:  # noqa: SIM105
    import vllm_ascend.patch.worker.patch_npugraph_ex_triton  # noqa
except ImportError:
    pass
import vllm_ascend.patch.worker.patch_kimi_k25  # noqa
import vllm_ascend.patch.worker.patch_draft_quarot  # noqa
import vllm_ascend.patch.worker.patch_eagle3_init  # noqa
import vllm_ascend.patch.worker.patch_cudagraph  # noqa
import vllm_ascend.patch.worker.patch_deepseek_mtp  # noqa
import vllm_ascend.patch.worker.patch_deepseek_v2  # noqa
import vllm_ascend.patch.worker.patch_gqa_c8  # noqa

_import_optional_patch("vllm_ascend.patch.worker.patch_qwen3vl")

# Sim-LLM KV reuse — auto-loaded at worker init, gated behind
# VLLM_ASCEND_SIMLLM_ENABLED=1 (no-op when disabled).
_import_optional_patch("vllm_ascend.patch.worker.patch_simllm")

# vLLM's use_v2_model_runner may enable the v2 runner without the
# VLLM_USE_V2_MODEL_RUNNER env var (e.g. based on model architecture).
# We always patch it so that on Ascend the v2 runner is enabled only
# when the env var is explicitly set.
import vllm_ascend.patch.worker.patch_v2.patch_use_v2_model_runner  # noqa

if not vllm_version_is("0.23.0"):
    import vllm_ascend.patch.worker.patch_fused_moe  # noqa

if _V2_MODEL_RUNNER_SUPPORTED:
    import vllm_ascend.patch.worker.patch_v2.patch_uva  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_input_batch  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_model_state  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_block_table  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_attn_utils  # noqa

# only patch routed experts capture in main2main.
if _V2_MODEL_RUNNER_SUPPORTED:
    import vllm_ascend.patch.worker.patch_routed_experts_capture  # noqa

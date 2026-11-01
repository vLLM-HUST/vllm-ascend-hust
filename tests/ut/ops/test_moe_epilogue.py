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

import os
from unittest import mock

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.triton import moe_epilogue


class TestMoeEpilogueGate(TestBase):
    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(moe_epilogue.is_enabled())

    def test_enabled_only_on_exact_token(self):
        for value in ("0", "2", "true", "yes", ""):
            with mock.patch.dict(os.environ, {moe_epilogue.ENV_MOE_EPILOGUE: value}, clear=True):
                self.assertFalse(moe_epilogue.is_enabled(), value)
        with mock.patch.dict(os.environ, {moe_epilogue.ENV_MOE_EPILOGUE: "1"}, clear=True):
            self.assertTrue(moe_epilogue.is_enabled())

    def test_max_rows_default_and_override(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(moe_epilogue.max_rows(), moe_epilogue.DEFAULT_MAX_ROWS)
        with mock.patch.dict(os.environ, {moe_epilogue.ENV_MOE_EPILOGUE_MAX_ROWS: "16"}, clear=True):
            self.assertEqual(moe_epilogue.max_rows(), 16)
        with mock.patch.dict(os.environ, {moe_epilogue.ENV_MOE_EPILOGUE_MAX_ROWS: "abc"}, clear=True):
            self.assertEqual(moe_epilogue.max_rows(), moe_epilogue.DEFAULT_MAX_ROWS)


class TestMoeEpilogueFusable(TestBase):
    def _pair(self, rows=8, width=2048, dtype=torch.bfloat16):
        gate = torch.randn(rows, 1, dtype=dtype)
        out = torch.randn(rows, width, dtype=dtype)
        return gate, out

    def test_decode_shape_is_fusable(self):
        gate, out = self._pair()
        self.assertTrue(moe_epilogue._fusable(gate, out))

    def test_row_upper_bound(self):
        limit = moe_epilogue.DEFAULT_MAX_ROWS
        gate, out = self._pair(rows=limit)
        self.assertTrue(moe_epilogue._fusable(gate, out))
        gate, out = self._pair(rows=limit + 1)
        self.assertFalse(moe_epilogue._fusable(gate, out))

    def test_empty_rows_rejected(self):
        gate, out = self._pair(rows=0)
        self.assertFalse(moe_epilogue._fusable(gate, out))

    def test_non_bfloat16_rejected(self):
        gate, out = self._pair(dtype=torch.float16)
        self.assertFalse(moe_epilogue._fusable(gate, out))

    def test_shape_mismatch_rejected(self):
        gate = torch.randn(8, 2, dtype=torch.bfloat16)
        out = torch.randn(8, 2048, dtype=torch.bfloat16)
        self.assertFalse(moe_epilogue._fusable(gate, out))

        gate = torch.randn(4, 1, dtype=torch.bfloat16)
        self.assertFalse(moe_epilogue._fusable(gate, out))

    def test_non_contiguous_rejected(self):
        gate, out = self._pair()
        self.assertFalse(moe_epilogue._fusable(gate, out.t()))

    def test_1d_rejected(self):
        gate = torch.randn(8, dtype=torch.bfloat16)
        out = torch.randn(8, dtype=torch.bfloat16)
        self.assertFalse(moe_epilogue._fusable(gate, out))


class TestMoeEpilogueGateMul(TestBase):
    def _pair(self, rows=8, width=2048):
        gate = torch.randn(rows, 1, dtype=torch.bfloat16)
        out = torch.randn(rows, width, dtype=torch.bfloat16)
        return gate, out

    def test_disabled_matches_torch_mul(self):
        gate, out = self._pair()
        with mock.patch.dict(os.environ, {}, clear=True):
            result = moe_epilogue.gate_mul(gate, out)
        torch.testing.assert_close(result, gate * out, rtol=0, atol=0)

    def test_prefill_rows_fall_back(self):
        gate, out = self._pair(rows=moe_epilogue.DEFAULT_MAX_ROWS + 1)
        with mock.patch.dict(os.environ, {moe_epilogue.ENV_MOE_EPILOGUE: "1"}, clear=True):
            result = moe_epilogue.gate_mul(gate, out)
        torch.testing.assert_close(result, gate * out, rtol=0, atol=0)

    def test_decode_shape_dispatches_kernel(self):
        gate, out = self._pair()
        with mock.patch.dict(os.environ, {moe_epilogue.ENV_MOE_EPILOGUE: "1"}, clear=True):
            with mock.patch.object(moe_epilogue, "_gate_mul_kernel") as kernel:
                moe_epilogue.gate_mul(gate, out)
        self.assertEqual(kernel.__getitem__.call_count, 1)

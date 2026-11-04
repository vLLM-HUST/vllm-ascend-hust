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
from vllm_ascend.ops import gdn_epilogue


class TestGate(TestBase):
    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(gdn_epilogue.is_enabled())
            self.assertFalse(gdn_epilogue.is_output_norm_enabled())

    def test_enabled_only_on_exact_token(self):
        for value in ("0", "true", "", "2"):
            with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_UNPACK_FUSED: value}, clear=True):
                self.assertFalse(gdn_epilogue.is_enabled(), value)
            with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_OUTPUT_NORM_FUSED: value}, clear=True):
                self.assertFalse(gdn_epilogue.is_output_norm_enabled(), value)
        with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_UNPACK_FUSED: "1"}, clear=True):
            self.assertTrue(gdn_epilogue.is_enabled())
        with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_OUTPUT_NORM_FUSED: "1"}, clear=True):
            self.assertTrue(gdn_epilogue.is_output_norm_enabled())

    def test_max_rows_default_and_override(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gdn_epilogue._max_rows(), gdn_epilogue.DEFAULT_MAX_ROWS)
        with mock.patch.dict(os.environ, {"VLLM_HUST_GDN_UNPACK_MAXROWS": "64"}, clear=True):
            self.assertEqual(gdn_epilogue._max_rows(), 64)
        with mock.patch.dict(os.environ, {"VLLM_HUST_GDN_UNPACK_MAXROWS": "abc"}, clear=True):
            self.assertEqual(gdn_epilogue._max_rows(), gdn_epilogue.DEFAULT_MAX_ROWS)


class TestFusable(TestBase):
    def _pair(self, rows=8, dtype=torch.bfloat16):
        return (
            torch.randn(rows, 3072, dtype=dtype),
            torch.randn(rows, 256, dtype=dtype),
        )

    def test_enabled_decode_shape(self):
        qkvz, ba = self._pair()
        with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_UNPACK_FUSED: "1"}, clear=True):
            self.assertTrue(gdn_epilogue.unpack_fusable(qkvz, ba))
            self.assertTrue(gdn_epilogue.rearrange_fusable(qkvz))

    def test_disabled_rejects(self):
        qkvz, ba = self._pair()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(gdn_epilogue.unpack_fusable(qkvz, ba))
            self.assertFalse(gdn_epilogue.rearrange_fusable(qkvz))

    def test_single_row_rejected(self):
        # One row loses to the stock chain; measured at 1.05us vs 0.37us.
        qkvz, ba = self._pair(rows=1)
        with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_UNPACK_FUSED: "1"}, clear=True):
            self.assertFalse(gdn_epilogue.unpack_fusable(qkvz, ba))
            self.assertFalse(gdn_epilogue.rearrange_fusable(qkvz))

    def test_row_mismatch_rejected(self):
        qkvz, _ = self._pair(rows=8)
        _, ba = self._pair(rows=4)
        with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_UNPACK_FUSED: "1"}, clear=True):
            self.assertFalse(gdn_epilogue.unpack_fusable(qkvz, ba))

    def test_non_bfloat16_rejected(self):
        qkvz, ba = self._pair(dtype=torch.float16)
        with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_UNPACK_FUSED: "1"}, clear=True):
            self.assertFalse(gdn_epilogue.unpack_fusable(qkvz, ba))

    def test_non_contiguous_rejected(self):
        qkvz, ba = self._pair()
        with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_UNPACK_FUSED: "1"}, clear=True):
            self.assertFalse(gdn_epilogue.unpack_fusable(qkvz.t(), ba))
            self.assertFalse(gdn_epilogue.rearrange_fusable(qkvz.t()))

    def test_row_upper_bound(self):
        with mock.patch.dict(
            os.environ,
            {gdn_epilogue.ENV_GDN_UNPACK_FUSED: "1", "VLLM_HUST_GDN_UNPACK_MAXROWS": "4"},
            clear=True,
        ):
            qkvz, ba = self._pair(rows=4)
            self.assertTrue(gdn_epilogue.unpack_fusable(qkvz, ba))
            qkvz, ba = self._pair(rows=5)
            self.assertFalse(gdn_epilogue.unpack_fusable(qkvz, ba))

    def test_output_norm_fusable_checks(self):
        c = torch.randn(2, 4, 128, dtype=torch.bfloat16)
        z = torch.randn(2, 4, 128, dtype=torch.bfloat16)
        w = torch.randn(128, dtype=torch.bfloat16)
        norm_mod = torch.nn.Module()
        norm_mod.weight = torch.nn.Parameter(w)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(gdn_epilogue.output_norm_fusable(c, z, norm_mod))

        with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_OUTPUT_NORM_FUSED: "1"}, clear=True):
            self.assertTrue(gdn_epilogue.output_norm_fusable(c, z, norm_mod))

            # Dimension mismatch (must be 3D)
            self.assertFalse(gdn_epilogue.output_norm_fusable(c.view(8, 128), z.view(8, 128), norm_mod))

            # Dtype mismatch
            self.assertFalse(gdn_epilogue.output_norm_fusable(c.float(), z.float(), norm_mod))

            # Shape mismatch between c and z
            self.assertFalse(gdn_epilogue.output_norm_fusable(c, z[:1], norm_mod))

            # Weight dim mismatch
            bad_norm = torch.nn.Module()
            bad_norm.weight = torch.nn.Parameter(torch.randn(64, dtype=torch.bfloat16))
            self.assertFalse(gdn_epilogue.output_norm_fusable(c, z, bad_norm))


class TestStockFallback(TestBase):
    QKV = 2048
    HALF = 128

    def _pair(self, rows=8):
        return (
            torch.randn(rows, self.QKV + 1024, dtype=torch.bfloat16),
            torch.randn(rows, 2 * self.HALF, dtype=torch.bfloat16),
        )

    def test_unpack_disabled_matches_stock(self):
        qkvz, ba = self._pair()
        with mock.patch.dict(os.environ, {}, clear=True):
            mixed, b, a = gdn_epilogue.gdn_unpack(qkvz, ba, self.QKV, self.HALF)
        torch.testing.assert_close(mixed, qkvz[:, : self.QKV].contiguous(), rtol=0, atol=0)
        expect_b, expect_a = ba.chunk(2, dim=-1)
        torch.testing.assert_close(b, expect_b.contiguous(), rtol=0, atol=0)
        torch.testing.assert_close(a, expect_a.contiguous(), rtol=0, atol=0)

    def test_unpack_bad_width_falls_back(self):
        qkvz, ba = self._pair()
        with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_UNPACK_FUSED: "1"}, clear=True):
            mixed, _, _ = gdn_epilogue.gdn_unpack(qkvz, ba, self.QKV, self.HALF + 1)
        torch.testing.assert_close(mixed, qkvz[:, : self.QKV].contiguous(), rtol=0, atol=0)

    def test_rearrange_disabled_matches_stock(self):
        q = k = 1024
        v = 2048
        mixed = torch.randn(8, q + k + v, dtype=torch.bfloat16)
        with mock.patch.dict(os.environ, {}, clear=True):
            out_q, out_k, out_v = gdn_epilogue.gdn_rearrange(mixed, q, k, v)
        ref_q, ref_k, ref_v = gdn_epilogue._stock_rearrange(mixed, q, k, v)
        for got, ref in ((out_q, ref_q), (out_k, ref_k), (out_v, ref_v)):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
            self.assertTrue(got.is_contiguous())

    def test_rearrange_non_contiguous_falls_back(self):
        q = k = 1024
        v = 2048
        base = torch.randn(8, 2 * (q + k + v), dtype=torch.bfloat16)
        mixed = base[:, : q + k + v]
        self.assertFalse(mixed.is_contiguous())
        with mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_UNPACK_FUSED: "1"}, clear=True):
            out_q, out_k, out_v = gdn_epilogue.gdn_rearrange(mixed, q, k, v)
        ref_q, ref_k, ref_v = gdn_epilogue._stock_rearrange(mixed, q, k, v)
        for got, ref in ((out_q, ref_q), (out_k, ref_k), (out_v, ref_v)):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_output_norm_stock_fallback(self):
        c = torch.randn(2, 4, 128, dtype=torch.bfloat16)
        z = torch.randn(2, 4, 128, dtype=torch.bfloat16)
        w = torch.randn(128, dtype=torch.bfloat16)

        with mock.patch.dict(os.environ, {}, clear=True):
            out = gdn_epilogue.gdn_output_norm_rearrange(c, z, w)
            ref = gdn_epilogue._stock_output_norm_rearrange(c, z, w)
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
            self.assertEqual(out.shape, (2, 512))


class TestKernelDispatch(TestBase):
    def test_unpack_dispatches_kernel(self):
        qkvz = torch.randn(8, 3072, dtype=torch.bfloat16)
        ba = torch.randn(8, 256, dtype=torch.bfloat16)
        with (
            mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_UNPACK_FUSED: "1"}, clear=True),
            mock.patch.object(gdn_epilogue, "_unpack_kernel") as kernel,
        ):
            gdn_epilogue.gdn_unpack(qkvz, ba, 2048, 128)
        self.assertEqual(kernel.__getitem__.call_count, 1)

    def test_rearrange_dispatches_kernel(self):
        mixed = torch.randn(8, 4096, dtype=torch.bfloat16)
        with (
            mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_UNPACK_FUSED: "1"}, clear=True),
            mock.patch.object(gdn_epilogue, "_rearrange_kernel") as kernel,
        ):
            gdn_epilogue.gdn_rearrange(mixed, 1024, 1024, 2048)
        self.assertEqual(kernel.__getitem__.call_count, 1)

    def test_output_norm_dispatches_kernel(self):
        c = torch.randn(2, 4, 128, dtype=torch.bfloat16)
        z = torch.randn(2, 4, 128, dtype=torch.bfloat16)
        w = torch.randn(128, dtype=torch.bfloat16)
        with (
            mock.patch.dict(os.environ, {gdn_epilogue.ENV_GDN_OUTPUT_NORM_FUSED: "1"}, clear=True),
            mock.patch.object(gdn_epilogue, "_output_norm_kernel") as kernel,
        ):
            gdn_epilogue.gdn_output_norm_rearrange(c, z, w)
        self.assertEqual(kernel.__getitem__.call_count, 1)

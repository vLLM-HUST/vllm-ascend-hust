/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "types.h"

namespace {

__aicore__ inline float AbsFloat(float value)
{
    return value < 0.0F ? -value : value;
}

template <typename T>
__aicore__ inline float SparseScalarToFloat(const T& value)
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        union BFloat16Bits {
            __aicore__ BFloat16Bits() {}
            bfloat16_t value;
            uint16_t bits;
        } source;
        union FloatBits {
            __aicore__ FloatBits() {}
            uint32_t bits;
            float value;
        } result;
        source.value = value;
        result.bits = static_cast<uint32_t>(source.bits) << 16;
        return result.value;
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
__aicore__ inline T SparseFloatToScalar(float value)
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        union FloatBits {
            __aicore__ FloatBits() {}
            float value;
            uint32_t bits;
        } source;
        union BFloat16Bits {
            __aicore__ BFloat16Bits() {}
            uint16_t bits;
            bfloat16_t value;
        } result;
        source.value = value;
        result.bits = static_cast<uint16_t>(source.bits >> 16);
        return result.value;
    } else {
        return static_cast<T>(value);
    }
}

template <typename T>
__aicore__ inline float SparseMagnitudeBitsToFloat(uint16_t bits)
{
    union ScalarBits {
        __aicore__ ScalarBits() {}
        T value;
        uint16_t bits;
    } source;
    source.bits = bits;
    return SparseScalarToFloat(source.value);
}

template <typename scalar_t>
class ActivationSparseTopkThreshold {
public:
    using X_T = scalar_t;
    static constexpr uint32_t INPUT_ALIGN_ELEMENTS = 32;
    static constexpr uint32_t OUTPUT_ALIGN_ELEMENTS = 16;

    __aicore__ inline explicit ActivationSparseTopkThreshold(AscendC::TPipe* pipe)
        : pipe_(pipe)
    {}

    __aicore__ inline void Init(__gm__ void* x, __gm__ void* threshold,
                                uint32_t batch_size, uint32_t input_dim,
                                uint32_t keep, uint32_t block_dim)
    {
        batchSize_ = batch_size;
        inputDim_ = input_dim;
        keep_ = keep;
        blockDim_ = block_dim;
        inputAligned_ = AlignElements(input_dim, INPUT_ALIGN_ELEMENTS);
        keepAligned_ = AlignElements(keep, OUTPUT_ALIGN_ELEMENTS);
        xGm_.SetGlobalBuffer((__gm__ X_T*)x);
        thresholdGm_.SetGlobalBuffer((__gm__ float*)threshold);
        pipe_->InitBuffer(inputQueue_, 1, inputAligned_ * sizeof(X_T));
        pipe_->InitBuffer(valuesQueue_, 1, keepAligned_ * sizeof(half));
        pipe_->InitBuffer(indicesQueue_, 1,
                          keepAligned_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t row = block_idx; row < batchSize_; row += blockDim_) {
            uint64_t row_offset = static_cast<uint64_t>(row) * inputDim_;
            CopyMagnitudeToUb(row_offset);
            AscendC::LocalTensor<X_T> x_local =
                inputQueue_.DeQue<X_T>();
            AscendC::LocalTensor<half> magnitude =
                x_local.template ReinterpretCast<half>();
            AscendC::Abs(magnitude, magnitude, inputDim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> values =
                valuesQueue_.AllocTensor<half>();
            AscendC::LocalTensor<int32_t> indices =
                indicesQueue_.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> source_indices;
            AscendC::LocalTensor<bool> finish;
            AscendC::TopKInfo info;
            info.outter = 1;
            info.inner = inputAligned_;
            info.n = inputDim_;
            AscendC::tiling::TopkTiling tiling = BuildTopkTiling();
            AscendC::TopK<half, false, false, false,
                          AscendC::TopKMode::TOPK_NORMAL>(
                values, indices, magnitude, source_indices, finish,
                static_cast<int32_t>(keep_), tiling, info, true);
            event_t event_id = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
            AscendC::SetFlag<AscendC::HardEvent::V_S>(event_id);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(event_id);
            uint16_t threshold_bits =
                values.template ReinterpretCast<uint16_t>().GetValue(keep_ - 1);
            thresholdGm_.SetValue(
                row, SparseMagnitudeBitsToFloat<X_T>(threshold_bits));
            valuesQueue_.FreeTensor(values);
            indicesQueue_.FreeTensor(indices);
            inputQueue_.FreeTensor(x_local);
        }
    }

private:
    __aicore__ inline uint32_t AlignElements(uint32_t value,
                                             uint32_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    __aicore__ inline void CopyMagnitudeToUb(uint64_t row_offset)
    {
        AscendC::LocalTensor<X_T> x_local =
            inputQueue_.AllocTensor<X_T>();
        AscendC::LocalTensor<half> magnitude =
            x_local.template ReinterpretCast<half>();
        AscendC::Duplicate(
            magnitude, static_cast<half>(-65504), inputAligned_);
        event_t event_id = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
        AscendC::DataCopyExtParams copy_params{
            1, static_cast<uint32_t>(inputDim_ * sizeof(X_T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<X_T> pad_params{false, 0, 0, 0};
        AscendC::DataCopyPad(x_local, xGm_[row_offset], copy_params,
                             pad_params);
        inputQueue_.EnQue(x_local);
    }

    __aicore__ inline AscendC::tiling::TopkTiling BuildTopkTiling()
    {
        AscendC::tiling::TopkTiling tiling;
        tiling.tmpLocalSize = 10 * inputAligned_;
        tiling.allDataSize = inputAligned_;
        tiling.innerDataSize = 4 * inputAligned_;
        tiling.sortRepeat = inputAligned_ / 32;
        tiling.mrgSortRepeat = inputAligned_ / 4;
        tiling.kAlignFourBytes = (keep_ + 7) & ~7U;
        tiling.kAlignTwoBytes = keepAligned_;
        tiling.maskOffset = keepAligned_;
        tiling.maskVreducev2FourBytes = 2 * keep_;
        tiling.maskVreducev2TwoBytes = 4 * keep_;
        tiling.mrgSortSrc1offset = 4;
        tiling.mrgSortSrc2offset = 8;
        tiling.mrgSortSrc3offset = 12;
        tiling.mrgSortTwoQueueSrc1Offset = 4;
        tiling.mrgFourQueueTailPara1 = 2 * inputAligned_;
        tiling.mrgFourQueueTailPara2 = 2;
        tiling.srcIndexOffset = 8 * inputAligned_;
        tiling.copyUbToUbBlockCount = inputAligned_ / 4;
        return tiling;
    }

    AscendC::TPipe* pipe_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<float> thresholdGm_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> valuesQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> indicesQueue_;
    uint32_t batchSize_;
    uint32_t inputDim_;
    uint32_t keep_;
    uint32_t blockDim_;
    uint32_t inputAligned_;
    uint32_t keepAligned_;
};

template <typename scalar_t>
class ActivationSparsePack {
public:
    using X_T = scalar_t;

    __aicore__ inline ActivationSparsePack() {}

    __aicore__ inline void Init(__gm__ void* x, __gm__ void* threshold,
                                __gm__ void* values, __gm__ void* indices,
                                __gm__ void* counts, uint32_t batch_size,
                                uint32_t input_dim, uint32_t block_dim,
                                bool threshold_per_row, bool inclusive)
    {
        batchSize_ = batch_size;
        inputDim_ = input_dim;
        blockDim_ = block_dim;
        thresholdPerRow_ = threshold_per_row;
        inclusive_ = inclusive;

        xGm_.SetGlobalBuffer((__gm__ X_T*)x);
        thresholdGm_.SetGlobalBuffer((__gm__ float*)threshold);
        valuesGm_.SetGlobalBuffer((__gm__ X_T*)values);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
        countsGm_.SetGlobalBuffer((__gm__ int32_t*)counts);
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t row = block_idx; row < batchSize_; row += blockDim_) {
            float threshold = thresholdPerRow_ ? thresholdGm_.GetValue(row)
                                               : thresholdGm_.GetValue(0);
            uint64_t row_offset = static_cast<uint64_t>(row) * inputDim_;
            int32_t count = 0;
            for (uint32_t in_col = 0; in_col < inputDim_; ++in_col) {
                X_T raw_value = xGm_.GetValue(row_offset + in_col);
                float x_value = SparseScalarToFloat(raw_value);
                float magnitude = AbsFloat(x_value);
                bool active = inclusive_ ? (magnitude >= threshold)
                                         : (magnitude > threshold);
                if (active) {
                    uint64_t out_offset =
                        row_offset + static_cast<uint32_t>(count);
                    valuesGm_.SetValue(out_offset, raw_value);
                    indicesGm_.SetValue(out_offset, static_cast<int32_t>(in_col));
                    ++count;
                }
            }
            countsGm_.SetValue(row, count);
        }
    }

private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<float> thresholdGm_;
    AscendC::GlobalTensor<X_T> valuesGm_;
    AscendC::GlobalTensor<int32_t> indicesGm_;
    AscendC::GlobalTensor<int32_t> countsGm_;
    uint32_t batchSize_;
    uint32_t inputDim_;
    uint32_t blockDim_;
    bool thresholdPerRow_;
    bool inclusive_;
};

template <typename scalar_t>
class ActivationSparseLinearPacked {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    __aicore__ inline ActivationSparseLinearPacked() {}

    __aicore__ inline void Init(__gm__ void* values, __gm__ void* indices,
                                __gm__ void* counts, __gm__ void* weight,
                                __gm__ void* y, uint32_t input_dim,
                                uint32_t output_dim, uint32_t work_items,
                                uint32_t block_dim)
    {
        inputDim_ = input_dim;
        outputDim_ = output_dim;
        workItems_ = work_items;
        blockDim_ = block_dim;

        valuesGm_.SetGlobalBuffer((__gm__ X_T*)values);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
        countsGm_.SetGlobalBuffer((__gm__ int32_t*)counts);
        weightGm_.SetGlobalBuffer((__gm__ W_T*)weight);
        yGm_.SetGlobalBuffer((__gm__ Y_T*)y);
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t linear_idx = block_idx; linear_idx < workItems_;
             linear_idx += blockDim_) {
            uint32_t row = linear_idx / outputDim_;
            uint32_t out_col = linear_idx - row * outputDim_;
            int32_t nnz = countsGm_.GetValue(row);
            float acc = 0.0F;

            uint64_t packed_offset = static_cast<uint64_t>(row) * inputDim_;
            uint64_t w_offset = static_cast<uint64_t>(out_col) * inputDim_;
            for (int32_t nz_pos = 0; nz_pos < nnz; ++nz_pos) {
                uint64_t value_offset = packed_offset + static_cast<uint32_t>(nz_pos);
                int32_t in_col = indicesGm_.GetValue(value_offset);
                float x_value =
                    SparseScalarToFloat(valuesGm_.GetValue(value_offset));
                float w_value = SparseScalarToFloat(
                    weightGm_.GetValue(w_offset + static_cast<uint32_t>(in_col)));
                acc += x_value * w_value;
            }

            yGm_.SetValue(linear_idx, SparseFloatToScalar<Y_T>(acc));
        }
    }

private:
    AscendC::GlobalTensor<X_T> valuesGm_;
    AscendC::GlobalTensor<int32_t> indicesGm_;
    AscendC::GlobalTensor<int32_t> countsGm_;
    AscendC::GlobalTensor<W_T> weightGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    uint32_t inputDim_;
    uint32_t outputDim_;
    uint32_t workItems_;
    uint32_t blockDim_;
};

template <typename scalar_t>
class ActivationSparseLinearPackedTransposed {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    static constexpr int32_t BUFFER_NUM = 1;
    static constexpr uint32_t OUTPUT_TILE = 1024;

    __aicore__ inline ActivationSparseLinearPackedTransposed(AscendC::TPipe* pipe)
        : pipe_(pipe)
    {}

    __aicore__ inline void Init(__gm__ void* values, __gm__ void* indices,
                                __gm__ void* counts, __gm__ void* weight_t,
                                __gm__ void* y, uint32_t input_dim,
                                uint32_t output_dim, uint32_t tile_count,
                                uint32_t work_items, uint32_t block_dim,
                                uint32_t output_tile)
    {
        inputDim_ = input_dim;
        outputDim_ = output_dim;
        tileCount_ = tile_count;
        workItems_ = work_items;
        blockDim_ = block_dim;
        outputTile_ = output_tile;

        valuesGm_.SetGlobalBuffer((__gm__ X_T*)values);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
        countsGm_.SetGlobalBuffer((__gm__ int32_t*)counts);
        weightTGm_.SetGlobalBuffer((__gm__ W_T*)weight_t);
        yGm_.SetGlobalBuffer((__gm__ Y_T*)y);

        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, OUTPUT_TILE * sizeof(W_T));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, OUTPUT_TILE * sizeof(Y_T));
        pipe_->InitBuffer(tmpBufferW_, OUTPUT_TILE * sizeof(float));
        pipe_->InitBuffer(accBufferY_, OUTPUT_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t work_idx = block_idx; work_idx < workItems_;
             work_idx += blockDim_) {
            uint32_t row = work_idx / tileCount_;
            uint32_t tile_idx = work_idx - row * tileCount_;
            uint32_t out_start = tile_idx * outputTile_;
            uint32_t tile_len = outputDim_ - out_start;
            if (tile_len > outputTile_) {
                tile_len = outputTile_;
            }
            ComputeTile(row, out_start, tile_len);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t row, uint32_t out_start,
                                       uint32_t tile_len)
    {
        AscendC::LocalTensor<float> acc = accBufferY_.Get<float>();
        Duplicate(acc, 0.0F, tile_len);

        uint64_t packed_offset = static_cast<uint64_t>(row) * inputDim_;
        int32_t nnz = countsGm_.GetValue(row);
        for (int32_t nz_pos = 0; nz_pos < nnz; ++nz_pos) {
            uint64_t value_offset = packed_offset + static_cast<uint32_t>(nz_pos);
            int32_t in_col = indicesGm_.GetValue(value_offset);
            float x_value =
                SparseScalarToFloat(valuesGm_.GetValue(value_offset));
            CopyInW(static_cast<uint32_t>(in_col), out_start, tile_len);
            Compute(x_value, tile_len);
        }
        CopyOut(row, out_start, tile_len);
    }

    __aicore__ inline void CopyInW(uint32_t in_col, uint32_t out_start,
                                   uint32_t tile_len)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        uint64_t weight_offset =
            static_cast<uint64_t>(in_col) * outputDim_ + out_start;
        AscendC::DataCopyPadExtParams<W_T> pad_params{false, 0, 0, 0};
        AscendC::DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(tile_len * sizeof(W_T)),
            0,
            0,
            0,
        };
        DataCopyPad(wLocal, weightTGm_[weight_offset], copy_params, pad_params);
        inQueueW_.EnQue(wLocal);
    }

    __aicore__ inline void Compute(float x_value, uint32_t tile_len)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.DeQue<W_T>();
        AscendC::LocalTensor<float> wTmp = tmpBufferW_.Get<float>();
        AscendC::LocalTensor<float> acc = accBufferY_.Get<float>();

        Cast(wTmp, wLocal, AscendC::RoundMode::CAST_NONE, tile_len);
        inQueueW_.FreeTensor(wLocal);

        Muls(wTmp, wTmp, x_value, tile_len);
        Add(acc, acc, wTmp, tile_len);
    }

    __aicore__ inline void CopyOut(uint32_t row, uint32_t out_start,
                                   uint32_t tile_len)
    {
        AscendC::LocalTensor<float> acc = accBufferY_.Get<float>();
        AscendC::LocalTensor<Y_T> yLocal = outQueueY_.AllocTensor<Y_T>();
        Cast(yLocal, acc, AscendC::RoundMode::CAST_RINT, tile_len);
        outQueueY_.EnQue<Y_T>(yLocal);

        yLocal = outQueueY_.DeQue<Y_T>();
        uint64_t y_offset = static_cast<uint64_t>(row) * outputDim_ + out_start;
        AscendC::DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(tile_len * sizeof(Y_T)),
            0,
            0,
            0,
        };
        DataCopyPad(yGm_[y_offset], yLocal, copy_params);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueW_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferW_, accBufferY_;
    AscendC::GlobalTensor<X_T> valuesGm_;
    AscendC::GlobalTensor<int32_t> indicesGm_;
    AscendC::GlobalTensor<int32_t> countsGm_;
    AscendC::GlobalTensor<W_T> weightTGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    uint32_t inputDim_;
    uint32_t outputDim_;
    uint32_t tileCount_;
    uint32_t workItems_;
    uint32_t blockDim_;
    uint32_t outputTile_;
};

template <typename scalar_t>
class ActivationSparseSiluAndMulPackedTransposed {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    static constexpr int32_t BUFFER_NUM = 1;
    static constexpr uint32_t OUTPUT_TILE = 1024;

    __aicore__ inline ActivationSparseSiluAndMulPackedTransposed(
        AscendC::TPipe* pipe)
        : pipe_(pipe)
    {}

    __aicore__ inline void Init(__gm__ void* values, __gm__ void* indices,
                                __gm__ void* counts, __gm__ void* weight_t,
                                __gm__ void* y, uint32_t input_dim,
                                uint32_t intermediate_dim,
                                uint32_t tile_count, uint32_t work_items,
                                uint32_t block_dim, uint32_t output_tile)
    {
        inputDim_ = input_dim;
        intermediateDim_ = intermediate_dim;
        gateUpDim_ = intermediate_dim * 2;
        tileCount_ = tile_count;
        workItems_ = work_items;
        blockDim_ = block_dim;
        outputTile_ = output_tile;

        valuesGm_.SetGlobalBuffer((__gm__ X_T*)values);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
        countsGm_.SetGlobalBuffer((__gm__ int32_t*)counts);
        weightTGm_.SetGlobalBuffer((__gm__ W_T*)weight_t);
        yGm_.SetGlobalBuffer((__gm__ Y_T*)y);

        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, 2 * OUTPUT_TILE * sizeof(W_T));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, OUTPUT_TILE * sizeof(Y_T));
        pipe_->InitBuffer(tmpBufferW_, 2 * OUTPUT_TILE * sizeof(float));
        pipe_->InitBuffer(accGateBuffer_, OUTPUT_TILE * sizeof(float));
        pipe_->InitBuffer(accUpBuffer_, OUTPUT_TILE * sizeof(float));
        pipe_->InitBuffer(siluBuffer_, OUTPUT_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t work_idx = block_idx; work_idx < workItems_;
             work_idx += blockDim_) {
            uint32_t row = work_idx / tileCount_;
            uint32_t tile_idx = work_idx - row * tileCount_;
            uint32_t out_start = tile_idx * outputTile_;
            uint32_t tile_len = intermediateDim_ - out_start;
            if (tile_len > outputTile_) {
                tile_len = outputTile_;
            }
            ComputeTile(row, out_start, tile_len);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t row, uint32_t out_start,
                                       uint32_t tile_len)
    {
        AscendC::LocalTensor<float> accGate = accGateBuffer_.Get<float>();
        AscendC::LocalTensor<float> accUp = accUpBuffer_.Get<float>();
        Duplicate(accGate, 0.0F, tile_len);
        Duplicate(accUp, 0.0F, tile_len);

        uint64_t packed_offset = static_cast<uint64_t>(row) * inputDim_;
        int32_t nnz = countsGm_.GetValue(row);
        for (int32_t nz_pos = 0; nz_pos < nnz; ++nz_pos) {
            uint64_t value_offset = packed_offset + static_cast<uint32_t>(nz_pos);
            int32_t in_col = indicesGm_.GetValue(value_offset);
            float x_value =
                SparseScalarToFloat(valuesGm_.GetValue(value_offset));
            CopyInGateUp(static_cast<uint32_t>(in_col), out_start, tile_len);
            ComputeGateUp(accGate, accUp, x_value, tile_len);
        }
        ApplySiluAndCopyOut(row, out_start, tile_len);
    }

    __aicore__ inline void CopyInGateUp(uint32_t in_col, uint32_t out_start,
                                        uint32_t tile_len)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        uint64_t gate_offset =
            static_cast<uint64_t>(in_col) * gateUpDim_ + out_start;
        uint64_t up_offset = static_cast<uint64_t>(in_col) * gateUpDim_ +
                             intermediateDim_ + out_start;
        AscendC::DataCopyPadExtParams<W_T> pad_params{false, 0, 0, 0};
        AscendC::DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(tile_len * sizeof(W_T)),
            0,
            0,
            0,
        };
        DataCopyPad(wLocal, weightTGm_[gate_offset], copy_params, pad_params);
        DataCopyPad(wLocal[OUTPUT_TILE], weightTGm_[up_offset], copy_params,
                    pad_params);
        inQueueW_.EnQue(wLocal);
    }

    __aicore__ inline void ComputeGateUp(AscendC::LocalTensor<float> accGate,
                                         AscendC::LocalTensor<float> accUp,
                                         float x_value, uint32_t tile_len)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.DeQue<W_T>();
        AscendC::LocalTensor<float> wTmp = tmpBufferW_.Get<float>();

        Cast(wTmp, wLocal, AscendC::RoundMode::CAST_NONE, tile_len);
        Cast(wTmp[OUTPUT_TILE], wLocal[OUTPUT_TILE],
             AscendC::RoundMode::CAST_NONE, tile_len);
        inQueueW_.FreeTensor(wLocal);

        Muls(wTmp, wTmp, x_value, tile_len);
        Add(accGate, accGate, wTmp, tile_len);
        Muls(wTmp[OUTPUT_TILE], wTmp[OUTPUT_TILE], x_value, tile_len);
        Add(accUp, accUp, wTmp[OUTPUT_TILE], tile_len);
    }

    __aicore__ inline void ApplySiluAndCopyOut(uint32_t row,
                                               uint32_t out_start,
                                               uint32_t tile_len)
    {
        AscendC::LocalTensor<float> accGate = accGateBuffer_.Get<float>();
        AscendC::LocalTensor<float> accUp = accUpBuffer_.Get<float>();
        AscendC::LocalTensor<float> silu = siluBuffer_.Get<float>();

        Muls(silu, accGate, -1.0F, tile_len);
        Exp(silu, silu, tile_len);
        Adds(silu, silu, 1.0F, tile_len);
        Div(silu, accGate, silu, tile_len);
        Mul(silu, silu, accUp, tile_len);

        AscendC::LocalTensor<Y_T> yLocal = outQueueY_.AllocTensor<Y_T>();
        Cast(yLocal, silu, AscendC::RoundMode::CAST_RINT, tile_len);
        outQueueY_.EnQue<Y_T>(yLocal);

        yLocal = outQueueY_.DeQue<Y_T>();
        uint64_t y_offset =
            static_cast<uint64_t>(row) * intermediateDim_ + out_start;
        AscendC::DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(tile_len * sizeof(Y_T)),
            0,
            0,
            0,
        };
        DataCopyPad(yGm_[y_offset], yLocal, copy_params);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueW_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferW_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accGateBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accUpBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> siluBuffer_;
    AscendC::GlobalTensor<X_T> valuesGm_;
    AscendC::GlobalTensor<int32_t> indicesGm_;
    AscendC::GlobalTensor<int32_t> countsGm_;
    AscendC::GlobalTensor<W_T> weightTGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    uint32_t inputDim_;
    uint32_t intermediateDim_;
    uint32_t gateUpDim_;
    uint32_t tileCount_;
    uint32_t workItems_;
    uint32_t blockDim_;
    uint32_t outputTile_;
};

template <typename scalar_t>
class ActivationSparseLinearDirectTransposed {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    static constexpr int32_t BUFFER_NUM = 1;
    static constexpr uint32_t OUTPUT_TILE = 1024;

    __aicore__ inline ActivationSparseLinearDirectTransposed(AscendC::TPipe* pipe)
        : pipe_(pipe)
    {}

    __aicore__ inline void Init(__gm__ void* x, __gm__ void* weight_t,
                                __gm__ void* threshold, __gm__ void* y,
                                uint32_t input_dim, uint32_t output_dim,
                                uint32_t tile_count, uint32_t work_items,
                                uint32_t block_dim, uint32_t output_tile,
                                bool threshold_per_row, bool inclusive)
    {
        inputDim_ = input_dim;
        outputDim_ = output_dim;
        tileCount_ = tile_count;
        workItems_ = work_items;
        blockDim_ = block_dim;
        outputTile_ = output_tile;
        thresholdPerRow_ = threshold_per_row;
        inclusive_ = inclusive;

        xGm_.SetGlobalBuffer((__gm__ X_T*)x);
        weightTGm_.SetGlobalBuffer((__gm__ W_T*)weight_t);
        thresholdGm_.SetGlobalBuffer((__gm__ float*)threshold);
        yGm_.SetGlobalBuffer((__gm__ Y_T*)y);

        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, OUTPUT_TILE * sizeof(W_T));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, OUTPUT_TILE * sizeof(Y_T));
        pipe_->InitBuffer(tmpBufferW_, OUTPUT_TILE * sizeof(float));
        pipe_->InitBuffer(accBufferY_, OUTPUT_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t work_idx = block_idx; work_idx < workItems_;
             work_idx += blockDim_) {
            uint32_t row = work_idx / tileCount_;
            uint32_t tile_idx = work_idx - row * tileCount_;
            uint32_t out_start = tile_idx * outputTile_;
            uint32_t tile_len = outputDim_ - out_start;
            if (tile_len > outputTile_) {
                tile_len = outputTile_;
            }
            ComputeTile(row, out_start, tile_len);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t row, uint32_t out_start,
                                       uint32_t tile_len)
    {
        AscendC::LocalTensor<float> acc = accBufferY_.Get<float>();
        Duplicate(acc, 0.0F, tile_len);

        float threshold = thresholdPerRow_ ? thresholdGm_.GetValue(row)
                                           : thresholdGm_.GetValue(0);
        uint64_t x_offset = static_cast<uint64_t>(row) * inputDim_;
        for (uint32_t in_col = 0; in_col < inputDim_; ++in_col) {
            float x_value =
                SparseScalarToFloat(xGm_.GetValue(x_offset + in_col));
            float magnitude = AbsFloat(x_value);
            bool active = inclusive_ ? (magnitude >= threshold)
                                     : (magnitude > threshold);
            if (active) {
                CopyInW(in_col, out_start, tile_len);
                Compute(x_value, tile_len);
            }
        }
        CopyOut(row, out_start, tile_len);
    }

    __aicore__ inline void CopyInW(uint32_t in_col, uint32_t out_start,
                                   uint32_t tile_len)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        uint64_t weight_offset =
            static_cast<uint64_t>(in_col) * outputDim_ + out_start;
        AscendC::DataCopyPadExtParams<W_T> pad_params{false, 0, 0, 0};
        AscendC::DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(tile_len * sizeof(W_T)),
            0,
            0,
            0,
        };
        DataCopyPad(wLocal, weightTGm_[weight_offset], copy_params, pad_params);
        inQueueW_.EnQue(wLocal);
    }

    __aicore__ inline void Compute(float x_value, uint32_t tile_len)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.DeQue<W_T>();
        AscendC::LocalTensor<float> wTmp = tmpBufferW_.Get<float>();
        AscendC::LocalTensor<float> acc = accBufferY_.Get<float>();

        Cast(wTmp, wLocal, AscendC::RoundMode::CAST_NONE, tile_len);
        inQueueW_.FreeTensor(wLocal);

        Muls(wTmp, wTmp, x_value, tile_len);
        Add(acc, acc, wTmp, tile_len);
    }

    __aicore__ inline void CopyOut(uint32_t row, uint32_t out_start,
                                   uint32_t tile_len)
    {
        AscendC::LocalTensor<float> acc = accBufferY_.Get<float>();
        AscendC::LocalTensor<Y_T> yLocal = outQueueY_.AllocTensor<Y_T>();
        Cast(yLocal, acc, AscendC::RoundMode::CAST_RINT, tile_len);
        outQueueY_.EnQue<Y_T>(yLocal);

        yLocal = outQueueY_.DeQue<Y_T>();
        uint64_t y_offset = static_cast<uint64_t>(row) * outputDim_ + out_start;
        AscendC::DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(tile_len * sizeof(Y_T)),
            0,
            0,
            0,
        };
        DataCopyPad(yGm_[y_offset], yLocal, copy_params);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueW_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferW_, accBufferY_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> weightTGm_;
    AscendC::GlobalTensor<float> thresholdGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    uint32_t inputDim_;
    uint32_t outputDim_;
    uint32_t tileCount_;
    uint32_t workItems_;
    uint32_t blockDim_;
    uint32_t outputTile_;
    bool thresholdPerRow_;
    bool inclusive_;
};

template <typename scalar_t>
class ActivationSparseSiluAndMulDirectTransposed {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    static constexpr int32_t BUFFER_NUM = 1;
    static constexpr uint32_t OUTPUT_TILE = 1024;

    __aicore__ inline ActivationSparseSiluAndMulDirectTransposed(
        AscendC::TPipe* pipe)
        : pipe_(pipe)
    {}

    __aicore__ inline void Init(__gm__ void* x, __gm__ void* weight_t,
                                __gm__ void* threshold, __gm__ void* y,
                                uint32_t input_dim, uint32_t intermediate_dim,
                                uint32_t tile_count, uint32_t work_items,
                                uint32_t block_dim, uint32_t output_tile,
                                bool threshold_per_row, bool inclusive)
    {
        inputDim_ = input_dim;
        intermediateDim_ = intermediate_dim;
        gateUpDim_ = intermediate_dim * 2;
        tileCount_ = tile_count;
        workItems_ = work_items;
        blockDim_ = block_dim;
        outputTile_ = output_tile;
        thresholdPerRow_ = threshold_per_row;
        inclusive_ = inclusive;

        xGm_.SetGlobalBuffer((__gm__ X_T*)x);
        weightTGm_.SetGlobalBuffer((__gm__ W_T*)weight_t);
        thresholdGm_.SetGlobalBuffer((__gm__ float*)threshold);
        yGm_.SetGlobalBuffer((__gm__ Y_T*)y);

        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, 2 * OUTPUT_TILE * sizeof(W_T));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, OUTPUT_TILE * sizeof(Y_T));
        pipe_->InitBuffer(tmpBufferW_, 2 * OUTPUT_TILE * sizeof(float));
        pipe_->InitBuffer(accGateBuffer_, OUTPUT_TILE * sizeof(float));
        pipe_->InitBuffer(accUpBuffer_, OUTPUT_TILE * sizeof(float));
        pipe_->InitBuffer(siluBuffer_, OUTPUT_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t work_idx = block_idx; work_idx < workItems_;
             work_idx += blockDim_) {
            uint32_t row = work_idx / tileCount_;
            uint32_t tile_idx = work_idx - row * tileCount_;
            uint32_t out_start = tile_idx * outputTile_;
            uint32_t tile_len = intermediateDim_ - out_start;
            if (tile_len > outputTile_) {
                tile_len = outputTile_;
            }
            ComputeTile(row, out_start, tile_len);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t row, uint32_t out_start,
                                       uint32_t tile_len)
    {
        AscendC::LocalTensor<float> accGate = accGateBuffer_.Get<float>();
        AscendC::LocalTensor<float> accUp = accUpBuffer_.Get<float>();
        Duplicate(accGate, 0.0F, tile_len);
        Duplicate(accUp, 0.0F, tile_len);

        float threshold = thresholdPerRow_ ? thresholdGm_.GetValue(row)
                                           : thresholdGm_.GetValue(0);
        uint64_t x_offset = static_cast<uint64_t>(row) * inputDim_;
        for (uint32_t in_col = 0; in_col < inputDim_; ++in_col) {
            float x_value =
                SparseScalarToFloat(xGm_.GetValue(x_offset + in_col));
            float magnitude = AbsFloat(x_value);
            bool active = inclusive_ ? (magnitude >= threshold)
                                     : (magnitude > threshold);
            if (active) {
                CopyInGateUp(in_col, out_start, tile_len);
                ComputeGateUp(accGate, accUp, x_value, tile_len);
            }
        }
        ApplySiluAndCopyOut(row, out_start, tile_len);
    }

    __aicore__ inline void CopyInGateUp(uint32_t in_col, uint32_t out_start,
                                        uint32_t tile_len)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        uint64_t gate_offset =
            static_cast<uint64_t>(in_col) * gateUpDim_ + out_start;
        uint64_t up_offset = static_cast<uint64_t>(in_col) * gateUpDim_ +
                             intermediateDim_ + out_start;
        AscendC::DataCopyPadExtParams<W_T> pad_params{false, 0, 0, 0};
        AscendC::DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(tile_len * sizeof(W_T)),
            0,
            0,
            0,
        };
        DataCopyPad(wLocal, weightTGm_[gate_offset], copy_params, pad_params);
        DataCopyPad(wLocal[OUTPUT_TILE], weightTGm_[up_offset], copy_params,
                    pad_params);
        inQueueW_.EnQue(wLocal);
    }

    __aicore__ inline void ComputeGateUp(AscendC::LocalTensor<float> accGate,
                                         AscendC::LocalTensor<float> accUp,
                                         float x_value, uint32_t tile_len)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.DeQue<W_T>();
        AscendC::LocalTensor<float> wTmp = tmpBufferW_.Get<float>();

        Cast(wTmp, wLocal, AscendC::RoundMode::CAST_NONE, tile_len);
        Cast(wTmp[OUTPUT_TILE], wLocal[OUTPUT_TILE],
             AscendC::RoundMode::CAST_NONE, tile_len);
        inQueueW_.FreeTensor(wLocal);

        Muls(wTmp, wTmp, x_value, tile_len);
        Add(accGate, accGate, wTmp, tile_len);
        Muls(wTmp[OUTPUT_TILE], wTmp[OUTPUT_TILE], x_value, tile_len);
        Add(accUp, accUp, wTmp[OUTPUT_TILE], tile_len);
    }

    __aicore__ inline void ApplySiluAndCopyOut(uint32_t row,
                                               uint32_t out_start,
                                               uint32_t tile_len)
    {
        AscendC::LocalTensor<float> accGate = accGateBuffer_.Get<float>();
        AscendC::LocalTensor<float> accUp = accUpBuffer_.Get<float>();
        AscendC::LocalTensor<float> silu = siluBuffer_.Get<float>();

        Muls(silu, accGate, -1.0F, tile_len);
        Exp(silu, silu, tile_len);
        Adds(silu, silu, 1.0F, tile_len);
        Div(silu, accGate, silu, tile_len);
        Mul(silu, silu, accUp, tile_len);

        AscendC::LocalTensor<Y_T> yLocal = outQueueY_.AllocTensor<Y_T>();
        Cast(yLocal, silu, AscendC::RoundMode::CAST_RINT, tile_len);
        outQueueY_.EnQue<Y_T>(yLocal);

        yLocal = outQueueY_.DeQue<Y_T>();
        uint64_t y_offset =
            static_cast<uint64_t>(row) * intermediateDim_ + out_start;
        AscendC::DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(tile_len * sizeof(Y_T)),
            0,
            0,
            0,
        };
        DataCopyPad(yGm_[y_offset], yLocal, copy_params);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueW_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferW_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accGateBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accUpBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> siluBuffer_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> weightTGm_;
    AscendC::GlobalTensor<float> thresholdGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    uint32_t inputDim_;
    uint32_t intermediateDim_;
    uint32_t gateUpDim_;
    uint32_t tileCount_;
    uint32_t workItems_;
    uint32_t blockDim_;
    uint32_t outputTile_;
    bool thresholdPerRow_;
    bool inclusive_;
};

template <typename scalar_t>
class ActivationSparseLinear {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    __aicore__ inline ActivationSparseLinear() {}

    __aicore__ inline void Init(__gm__ void* x, __gm__ void* weight,
                                __gm__ void* threshold, __gm__ void* y,
                                uint32_t input_dim, uint32_t output_dim,
                                bool threshold_per_row, bool inclusive,
                                uint32_t work_items, uint32_t block_dim)
    {
        inputDim_ = input_dim;
        outputDim_ = output_dim;
        thresholdPerRow_ = threshold_per_row;
        inclusive_ = inclusive;
        workItems_ = work_items;
        blockDim_ = block_dim;

        xGm_.SetGlobalBuffer((__gm__ X_T*)x);
        weightGm_.SetGlobalBuffer((__gm__ W_T*)weight);
        thresholdGm_.SetGlobalBuffer((__gm__ float*)threshold);
        yGm_.SetGlobalBuffer((__gm__ Y_T*)y);
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t linear_idx = block_idx; linear_idx < workItems_;
             linear_idx += blockDim_) {
            uint32_t row = linear_idx / outputDim_;
            uint32_t out_col = linear_idx - row * outputDim_;
            float threshold = thresholdPerRow_ ? thresholdGm_.GetValue(row)
                                               : thresholdGm_.GetValue(0);
            float acc = 0.0F;

            uint64_t x_offset = static_cast<uint64_t>(row) * inputDim_;
            uint64_t w_offset = static_cast<uint64_t>(out_col) * inputDim_;
            for (uint32_t in_col = 0; in_col < inputDim_; ++in_col) {
                float x_value =
                    SparseScalarToFloat(xGm_.GetValue(x_offset + in_col));
                float magnitude = AbsFloat(x_value);
                bool active = inclusive_ ? (magnitude >= threshold)
                                         : (magnitude > threshold);
                if (active) {
                    float w_value = SparseScalarToFloat(
                        weightGm_.GetValue(w_offset + in_col));
                    acc += x_value * w_value;
                }
            }

            yGm_.SetValue(linear_idx, SparseFloatToScalar<Y_T>(acc));
        }
    }

private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> weightGm_;
    AscendC::GlobalTensor<float> thresholdGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    uint32_t inputDim_;
    uint32_t outputDim_;
    bool thresholdPerRow_;
    bool inclusive_;
    uint32_t workItems_;
    uint32_t blockDim_;
};

#define ACTIVATION_SPARSE_PACK_TYPE_DECLARE(TYPE)                                  \
    extern "C" __global__ __aicore__ void activation_sparse_pack_##TYPE(           \
        __gm__ void* x, __gm__ void* threshold, __gm__ void* values,               \
        __gm__ void* indices, __gm__ void* counts, uint32_t batch_size,            \
        uint32_t input_dim, uint32_t block_dim, bool threshold_per_row,            \
        bool inclusive)                                                           \
    {                                                                              \
        ActivationSparsePack<TYPE> op;                                             \
        op.Init(x, threshold, values, indices, counts, batch_size, input_dim,       \
                block_dim, threshold_per_row, inclusive);                          \
        op.Process();                                                              \
    }

#define ACTIVATION_SPARSE_TOPK_THRESHOLD_TYPE_DECLARE(TYPE)                        \
    extern "C" __global__ __aicore__ void                                         \
    activation_sparse_topk_threshold_##TYPE(                                       \
        __gm__ void* x, __gm__ void* threshold, uint32_t batch_size,               \
        uint32_t input_dim, uint32_t keep, uint32_t block_dim)                     \
    {                                                                              \
        AscendC::TPipe pipe;                                                       \
        ActivationSparseTopkThreshold<TYPE> op(&pipe);                             \
        op.Init(x, threshold, batch_size, input_dim, keep, block_dim);              \
        op.Process();                                                              \
    }

#define ACTIVATION_SPARSE_LINEAR_PACKED_TYPE_DECLARE(TYPE)                         \
    extern "C" __global__ __aicore__ void activation_sparse_linear_packed_##TYPE(  \
        __gm__ void* values, __gm__ void* indices, __gm__ void* counts,            \
        __gm__ void* weight, __gm__ void* y, uint32_t input_dim,                   \
        uint32_t output_dim, uint32_t work_items, uint32_t block_dim)              \
    {                                                                              \
        ActivationSparseLinearPacked<TYPE> op;                                     \
        op.Init(values, indices, counts, weight, y, input_dim, output_dim,         \
                work_items, block_dim);                                            \
        op.Process();                                                              \
    }

#define ACTIVATION_SPARSE_LINEAR_PACKED_T_TYPE_DECLARE(TYPE)                       \
    extern "C" __global__ __aicore__ void                                          \
    activation_sparse_linear_packed_t_##TYPE(                                      \
        __gm__ void* values, __gm__ void* indices, __gm__ void* counts,            \
        __gm__ void* weight_t, __gm__ void* y, uint32_t input_dim,                 \
        uint32_t output_dim, uint32_t tile_count, uint32_t work_items,             \
        uint32_t block_dim, uint32_t output_tile)                                  \
    {                                                                              \
        AscendC::TPipe pipe;                                                       \
        ActivationSparseLinearPackedTransposed<TYPE> op(&pipe);                    \
        op.Init(values, indices, counts, weight_t, y, input_dim, output_dim,       \
                tile_count, work_items, block_dim, output_tile);                   \
        op.Process();                                                              \
    }

#define ACTIVATION_SPARSE_SILU_AND_MUL_PACKED_T_TYPE_DECLARE(TYPE)                 \
    extern "C" __global__ __aicore__ void                                          \
    activation_sparse_silu_and_mul_packed_t_##TYPE(                                \
        __gm__ void* values, __gm__ void* indices, __gm__ void* counts,            \
        __gm__ void* weight_t, __gm__ void* y, uint32_t input_dim,                 \
        uint32_t intermediate_dim, uint32_t tile_count, uint32_t work_items,        \
        uint32_t block_dim, uint32_t output_tile)                                  \
    {                                                                              \
        AscendC::TPipe pipe;                                                       \
        ActivationSparseSiluAndMulPackedTransposed<TYPE> op(&pipe);                \
        op.Init(values, indices, counts, weight_t, y, input_dim,                  \
                intermediate_dim, tile_count, work_items, block_dim, output_tile);  \
        op.Process();                                                              \
    }

#define ACTIVATION_SPARSE_LINEAR_DIRECT_T_TYPE_DECLARE(TYPE)                       \
    extern "C" __global__ __aicore__ void                                          \
    activation_sparse_linear_direct_t_##TYPE(                                      \
        __gm__ void* x, __gm__ void* weight_t, __gm__ void* threshold,             \
        __gm__ void* y, uint32_t input_dim, uint32_t output_dim,                   \
        uint32_t tile_count, uint32_t work_items, uint32_t block_dim,              \
        uint32_t output_tile, bool threshold_per_row, bool inclusive)              \
    {                                                                              \
        AscendC::TPipe pipe;                                                       \
        ActivationSparseLinearDirectTransposed<TYPE> op(&pipe);                    \
        op.Init(x, weight_t, threshold, y, input_dim, output_dim, tile_count,       \
                work_items, block_dim, output_tile, threshold_per_row, inclusive);  \
        op.Process();                                                              \
    }

#define ACTIVATION_SPARSE_SILU_AND_MUL_DIRECT_T_TYPE_DECLARE(TYPE)                 \
    extern "C" __global__ __aicore__ void                                          \
    activation_sparse_silu_and_mul_direct_t_##TYPE(                                \
        __gm__ void* x, __gm__ void* weight_t, __gm__ void* threshold,             \
        __gm__ void* y, uint32_t input_dim, uint32_t intermediate_dim,             \
        uint32_t tile_count, uint32_t work_items, uint32_t block_dim,              \
        uint32_t output_tile, bool threshold_per_row, bool inclusive)              \
    {                                                                              \
        AscendC::TPipe pipe;                                                       \
        ActivationSparseSiluAndMulDirectTransposed<TYPE> op(&pipe);                \
        op.Init(x, weight_t, threshold, y, input_dim, intermediate_dim,            \
                tile_count, work_items, block_dim, output_tile,                    \
                threshold_per_row, inclusive);                                     \
        op.Process();                                                              \
    }

#define ACTIVATION_SPARSE_LINEAR_TYPE_DECLARE(TYPE)                                \
    extern "C" __global__ __aicore__ void activation_sparse_linear_##TYPE(         \
        __gm__ void* x, __gm__ void* weight, __gm__ void* threshold,               \
        __gm__ void* y, uint32_t input_dim, uint32_t output_dim,                   \
        bool threshold_per_row, bool inclusive, uint32_t work_items,               \
        uint32_t block_dim)                                                        \
    {                                                                              \
        ActivationSparseLinear<TYPE> op;                                           \
        op.Init(x, weight, threshold, y, input_dim, output_dim,                    \
                threshold_per_row, inclusive, work_items, block_dim);              \
        op.Process();                                                              \
    }

ACTIVATION_SPARSE_PACK_TYPE_DECLARE(half)
ACTIVATION_SPARSE_TOPK_THRESHOLD_TYPE_DECLARE(half)
ACTIVATION_SPARSE_LINEAR_PACKED_TYPE_DECLARE(half)
ACTIVATION_SPARSE_LINEAR_PACKED_T_TYPE_DECLARE(half)
ACTIVATION_SPARSE_SILU_AND_MUL_PACKED_T_TYPE_DECLARE(half)
ACTIVATION_SPARSE_LINEAR_DIRECT_T_TYPE_DECLARE(half)
ACTIVATION_SPARSE_SILU_AND_MUL_DIRECT_T_TYPE_DECLARE(half)
ACTIVATION_SPARSE_LINEAR_TYPE_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
ACTIVATION_SPARSE_PACK_TYPE_DECLARE(bfloat16_t)
ACTIVATION_SPARSE_TOPK_THRESHOLD_TYPE_DECLARE(bfloat16_t)
ACTIVATION_SPARSE_LINEAR_PACKED_TYPE_DECLARE(bfloat16_t)
ACTIVATION_SPARSE_LINEAR_PACKED_T_TYPE_DECLARE(bfloat16_t)
ACTIVATION_SPARSE_SILU_AND_MUL_PACKED_T_TYPE_DECLARE(bfloat16_t)
ACTIVATION_SPARSE_LINEAR_DIRECT_T_TYPE_DECLARE(bfloat16_t)
ACTIVATION_SPARSE_SILU_AND_MUL_DIRECT_T_TYPE_DECLARE(bfloat16_t)
ACTIVATION_SPARSE_LINEAR_TYPE_DECLARE(bfloat16_t)
#endif

} // namespace

namespace vllm_ascend {

extern void activation_sparse_topk_threshold_impl(
    AscendType type, void* stream, void* x, void* threshold,
    uint32_t batch_size, uint32_t input_dim, uint32_t keep,
    uint32_t block_dim)
{
    if (batch_size == 0 || input_dim == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_topk_threshold_half<<<block_dim, nullptr, stream>>>(
            x, threshold, batch_size, input_dim, keep, block_dim);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_topk_threshold_bfloat16_t<<<block_dim, nullptr,
                                                      stream>>>(
            x, threshold, batch_size, input_dim, keep, block_dim);
#endif
    }
}

extern void activation_sparse_pack_impl(AscendType type, void* stream, void* x,
                                        void* threshold, void* values,
                                        void* indices, void* counts,
                                        uint32_t batch_size,
                                        uint32_t input_dim,
                                        uint32_t block_dim,
                                        bool threshold_per_row,
                                        bool inclusive)
{
    if (batch_size == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_pack_half<<<block_dim, nullptr, stream>>>(
            x, threshold, values, indices, counts, batch_size, input_dim,
            block_dim, threshold_per_row, inclusive);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_pack_bfloat16_t<<<block_dim, nullptr, stream>>>(
            x, threshold, values, indices, counts, batch_size, input_dim,
            block_dim, threshold_per_row, inclusive);
#endif
    } else {
        return;
    }
}

extern void activation_sparse_linear_packed_impl(
    AscendType type, void* stream, void* values, void* indices, void* counts,
    void* weight, void* y, uint32_t batch_size, uint32_t input_dim,
    uint32_t output_dim, uint32_t block_dim)
{
    uint32_t work_items = batch_size * output_dim;
    if (work_items == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_linear_packed_half<<<block_dim, nullptr, stream>>>(
            values, indices, counts, weight, y, input_dim, output_dim,
            work_items, block_dim);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_linear_packed_bfloat16_t<<<block_dim, nullptr,
                                                    stream>>>(
            values, indices, counts, weight, y, input_dim, output_dim,
            work_items, block_dim);
#endif
    } else {
        return;
    }
}

extern void activation_sparse_linear_packed_t_impl(
    AscendType type, void* stream, void* values, void* indices, void* counts,
    void* weight_t, void* y, uint32_t batch_size, uint32_t input_dim,
    uint32_t output_dim, uint32_t block_dim, uint32_t output_tile)
{
    uint32_t tile_count = (output_dim + output_tile - 1) / output_tile;
    uint32_t work_items = batch_size * tile_count;
    if (work_items == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_linear_packed_t_half<<<block_dim, nullptr, stream>>>(
            values, indices, counts, weight_t, y, input_dim, output_dim,
            tile_count, work_items, block_dim, output_tile);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_linear_packed_t_bfloat16_t<<<block_dim, nullptr,
                                                      stream>>>(
            values, indices, counts, weight_t, y, input_dim, output_dim,
            tile_count, work_items, block_dim, output_tile);
#endif
    } else {
        return;
    }
}

extern void activation_sparse_silu_and_mul_packed_t_impl(
    AscendType type, void* stream, void* values, void* indices, void* counts,
    void* weight_t, void* y, uint32_t batch_size, uint32_t input_dim,
    uint32_t intermediate_dim, uint32_t block_dim, uint32_t output_tile)
{
    uint32_t tile_count = (intermediate_dim + output_tile - 1) / output_tile;
    uint32_t work_items = batch_size * tile_count;
    if (work_items == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_silu_and_mul_packed_t_half<<<block_dim, nullptr,
                                                       stream>>>(
            values, indices, counts, weight_t, y, input_dim, intermediate_dim,
            tile_count, work_items, block_dim, output_tile);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_silu_and_mul_packed_t_bfloat16_t<<<block_dim, nullptr,
                                                            stream>>>(
            values, indices, counts, weight_t, y, input_dim, intermediate_dim,
            tile_count, work_items, block_dim, output_tile);
#endif
    } else {
        return;
    }
}

extern void activation_sparse_linear_direct_t_impl(
    AscendType type, void* stream, void* x, void* weight_t, void* threshold,
    void* y, uint32_t batch_size, uint32_t input_dim, uint32_t output_dim,
    bool threshold_per_row, bool inclusive, uint32_t block_dim,
    uint32_t output_tile)
{
    uint32_t tile_count = (output_dim + output_tile - 1) / output_tile;
    uint32_t work_items = batch_size * tile_count;
    if (work_items == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_linear_direct_t_half<<<block_dim, nullptr, stream>>>(
            x, weight_t, threshold, y, input_dim, output_dim, tile_count,
            work_items, block_dim, output_tile, threshold_per_row, inclusive);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_linear_direct_t_bfloat16_t<<<block_dim, nullptr,
                                                      stream>>>(
            x, weight_t, threshold, y, input_dim, output_dim, tile_count,
            work_items, block_dim, output_tile, threshold_per_row, inclusive);
#endif
    } else {
        return;
    }
}

extern void activation_sparse_silu_and_mul_direct_t_impl(
    AscendType type, void* stream, void* x, void* weight_t, void* threshold,
    void* y, uint32_t batch_size, uint32_t input_dim,
    uint32_t intermediate_dim, bool threshold_per_row, bool inclusive,
    uint32_t block_dim, uint32_t output_tile)
{
    uint32_t tile_count = (intermediate_dim + output_tile - 1) / output_tile;
    uint32_t work_items = batch_size * tile_count;
    if (work_items == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_silu_and_mul_direct_t_half<<<block_dim, nullptr,
                                                       stream>>>(
            x, weight_t, threshold, y, input_dim, intermediate_dim, tile_count,
            work_items, block_dim, output_tile, threshold_per_row, inclusive);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_silu_and_mul_direct_t_bfloat16_t<<<block_dim, nullptr,
                                                            stream>>>(
            x, weight_t, threshold, y, input_dim, intermediate_dim, tile_count,
            work_items, block_dim, output_tile, threshold_per_row, inclusive);
#endif
    } else {
        return;
    }
}

extern void activation_sparse_linear_impl(AscendType type, void* stream, void* x,
                                          void* weight, void* threshold,
                                          void* y, uint32_t batch_size,
                                          uint32_t input_dim,
                                          uint32_t output_dim,
                                          bool threshold_per_row,
                                          bool inclusive,
                                          uint32_t block_dim)
{
    uint32_t work_items = batch_size * output_dim;
    if (work_items == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_linear_half<<<block_dim, nullptr, stream>>>(
            x, weight, threshold, y, input_dim, output_dim, threshold_per_row,
            inclusive, work_items, block_dim);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_linear_bfloat16_t<<<block_dim, nullptr, stream>>>(
            x, weight, threshold, y, input_dim, output_dim, threshold_per_row,
            inclusive, work_items, block_dim);
#endif
    } else {
        return;
    }
}

} // namespace vllm_ascend

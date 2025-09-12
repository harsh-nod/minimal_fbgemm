/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using namespace fbgemm_gpu;

template <
    typename grad_t,
    typename cache_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH,
    bool kUseVecBlocking
>
DEVICE_INLINE void compute_grad_sum_unweighted(
    Vec4TAcc<cache_t>* grad_sum,
    Vec4TAcc<cache_t>* smem_grad_sum,
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>& grad_output,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& D_offsets,
    const int32_t D,
    const int32_t T,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& sorted_infos,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    const int32_t segment_start,
    const int32_t sl_start,
    const int32_t sl_end,
    const unsigned int shfl_sync_mask,
    const int32_t num_vecs
) {
    // Simplified implementation for testing
    // Copy value to vecs to make num_vecs known at compile time when
    // kUseVecBlocking == false
    const int32_t max_vecs =
        kUseVecBlocking ? num_vecs : kFixedMaxVecsPerThread;
    
    // Basic gradient sum computation
    for (int32_t d = 0; d < D; d += VEC_WIDTH) {
        Vec4TAcc<cache_t> grad_vec;
        grad_vec.clear();
        
        // Accumulate gradients
        for (int32_t i = sl_start; i < sl_end; ++i) {
            const int32_t info = sorted_infos[i];
            const int32_t t = info >> info_B_num_bits;
            const int32_t b = info & info_B_mask;
            
            if (d + VEC_WIDTH <= D) {
                grad_vec.acc.x += grad_output[b][D_offsets[t] + d];
                if (VEC_WIDTH > 1) grad_vec.acc.y += grad_output[b][D_offsets[t] + d + 1];
                if (VEC_WIDTH > 2) grad_vec.acc.z += grad_output[b][D_offsets[t] + d + 2];
                if (VEC_WIDTH > 3) grad_vec.acc.w += grad_output[b][D_offsets[t] + d + 3];
            }
        }
        
        grad_sum[d / VEC_WIDTH] = grad_vec;
    }
}

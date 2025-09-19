////////////////////////////////////////////////////////////////////////////////
// GENERATED FILE INFO
//
// Template Source: training/backward/embedding_backward_split_device_kernel_template.cuh
////////////////////////////////////////////////////////////////////////////////



/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using namespace fbgemm_gpu;

template<
    typename emb_t,
    typename cache_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    int32_t VEC_WIDTH,
    bool kUseVecBlocking
>
DEVICE_INLINE void store_grad_sum(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& grad_dev_weights,
    const Vec4TAcc<cache_t>* grad_sum,
    const Vec4TAcc<cache_t>* smem_grad_sum,
    const int32_t D,
    const int64_t weights_offset,
    const int64_t idx,
    const int32_t max_vecs_per_thread
) {
    // Copy value to max_vecs to make max_vecs_per_thread known at compile time
    // when kUseVecBlocking == false
    const int32_t max_vecs =
        kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;
    
    if constexpr (kUseVecBlocking) {
        // max_vecs is not known at compile time
        for (int32_t vec = 0;
            vec < max_vecs &&
            (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
            
            auto& grad = smem_grad_sum[d_vec];
            grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
            
        }
    
    } else {
        // kFixedMaxVecsPerThread is known at compile time
        #pragma unroll kFixedMaxVecsPerThread
        for (int32_t vec = 0;
            vec < kFixedMaxVecsPerThread
                && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
            
            auto& grad = grad_sum[vec];
            grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
            
        }
    }
    
}

    // clang-format on
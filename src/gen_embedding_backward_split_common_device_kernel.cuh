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
    
    // Store gradient sum to device weights
    for (int32_t d = 0; d < D; d += VEC_WIDTH) {
        const Vec4TAcc<cache_t>& grad = grad_sum[d / VEC_WIDTH];
        grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
    }
}

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
    typename emb_t,
    typename cache_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH,
    bool kUseVecBlocking
>
DEVICE_INLINE void split_rowwise_adagrad_table_update_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>& lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>& weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& sorted_lxu_cache_locations,
    Vec4TAcc<cache_t>* grad_sum,
    Vec4TAcc<cache_t>* smem_grad_sum,
    Vec4TAcc<cache_t>* shared_weight_update_row,
    const bool stochastic_rounding,
    const at::PhiloxCudaState& stochastic_rounding_philox_args,
    const uint32_t run_id,
    const uint32_t cache_loc_run_id,
    const int32_t D,
    const int32_t t,
    const int64_t idx,
    const float global_weight_decay,
    const uint32_t shfl_sync_mask,
    const int32_t max_vecs_per_thread,
    const float learning_rate,
    const float eps,
    const float weight_decay,
    const float momentum,
    const bool weight_decay_mode,
    const bool stochastic_rounding_mode
) {
    constexpr auto kIsInt8 = std::is_same<emb_t, uint8_t>::value;
    // Copy value to max_vecs to make max_vecs_per_thread known at compile time
    // when kUseVecBlocking == false
    const int32_t max_vecs =
        kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;
    const int64_t weights_offset = weights_offsets[t];
    emb_t* __restrict__ weights {nullptr};
    cache_t* __restrict__ cache_weights {nullptr};
    int32_t D_emb = D;
    if (kIsInt8) {
        D_emb += kINT8QparamsBytes;
    }
    const auto weights_placement = static_cast<PlacementType>(weights_placements[t]);
    if (weights_placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset + idx * D_emb];
    } else {
        weights = &uvm_weights[weights_offset + idx * D_emb];
    }
    if (weights_placement == PlacementType::MANAGED_CACHING) {
        const auto cache_idx = sorted_lxu_cache_locations[cache_loc_run_id];
        if (cache_idx != kCacheLocationMissing) {
          cache_weights = &lxu_cache_weights[cache_idx][0];
        }
    }

    // Simplified row-wise Adagrad update
    for (int32_t d = 0; d < D; d += VEC_WIDTH) {
        Vec4TAcc<cache_t> grad_vec = grad_sum[d / VEC_WIDTH];
        
        // Apply weight decay
        if (weight_decay_mode) {
            grad_vec.acc.x += weight_decay * weights[d];
            if (VEC_WIDTH > 1) grad_vec.acc.y += weight_decay * weights[d + 1];
            if (VEC_WIDTH > 2) grad_vec.acc.z += weight_decay * weights[d + 2];
            if (VEC_WIDTH > 3) grad_vec.acc.w += weight_decay * weights[d + 3];
        }
        
        // Update weights using row-wise Adagrad
        weights[d] -= learning_rate * grad_vec.acc.x / (sqrt(grad_vec.acc.x * grad_vec.acc.x) + eps);
        if (VEC_WIDTH > 1) weights[d + 1] -= learning_rate * grad_vec.acc.y / (sqrt(grad_vec.acc.y * grad_vec.acc.y) + eps);
        if (VEC_WIDTH > 2) weights[d + 2] -= learning_rate * grad_vec.acc.z / (sqrt(grad_vec.acc.z * grad_vec.acc.z) + eps);
        if (VEC_WIDTH > 3) weights[d + 3] -= learning_rate * grad_vec.acc.w / (sqrt(grad_vec.acc.w * grad_vec.acc.w) + eps);
    }
}

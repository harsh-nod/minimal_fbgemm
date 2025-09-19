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

template <
    typename grad_t,
    typename cache_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH,
    int32_t unrollCount
>
DEVICE_INLINE void compute_grad_sum_unweighted_unroll(
    Vec4TAcc<cache_t>* grad_sum,
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
    const int32_t vec_start
) {
    for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            auto sl_j = sl + threadIdx.x;
            const auto b_t = sl_j < sl_end
                ? reinterpret_cast<const uint32_t*>(
                    &sorted_infos[0])[segment_start + sl_j]
                : 0;
            const auto b = b_t & info_B_mask;
            const auto t = b_t >> info_B_num_bits; // if vbe
            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0; // if vbe // if not nobag
            for (int32_t j = 0; j < kThreadGroupSize/unrollCount && sl+unrollCount*j<sl_end ; ++j) {
                int32_t b_ids[unrollCount];
                int32_t D_startIds[unrollCount];

                #pragma unroll unrollCount
                for (int32_t i = 0; i < unrollCount; ++i) {
                    int32_t id = unrollCount*j+i;
                    int32_t b_id = SHFL_SYNC(b, id);
                    int32_t D_start_id = SHFL_SYNC(D_start, id);
                    b_ids[i]=b_id;
                    D_startIds[i]=D_start_id;
                }

                for (int32_t vec = 0; vec < kFixedMaxVecsPerThread && (((vec + vec_start) * kThreadGroupSize + threadIdx.x) * VEC_WIDTH) < D; ++vec) {
                    const int32_t d = (((vec + vec_start) * kThreadGroupSize + threadIdx.x) * VEC_WIDTH);
                    for (int32_t i = 0; i < unrollCount; ++i) {
                        int32_t id = unrollCount*j+i;
                        Vec4TAcc<grad_t> grad_out_vec(
                            &grad_output[b_ids[i]][0] + D_startIds[i] + d // if nobag
                        );
                        grad_sum[vec].add_(grad_out_vec);
                    }
                }
            }
        }
    }

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
    // Copy value to vecs to make num_vecs known at compile time when
    // kUseVecBlocking == false
    const int32_t vecs = kUseVecBlocking ? num_vecs : kFixedMaxVecsPerThread;
    for (int32_t vec_start = 0;
         vec_start < vecs;
         vec_start += kFixedMaxVecsPerThread) {

        // Reset grad_sum vectors
        #pragma unroll kFixedMaxVecsPerThread
        for (int32_t vec = 0; vec < kFixedMaxVecsPerThread; vec++) {
            grad_sum[vec].acc.x = 0;
            grad_sum[vec].acc.y = 0;
            grad_sum[vec].acc.z = 0;
            grad_sum[vec].acc.w = 0;
        }

        int32_t sl_length = sl_end - sl_start;
        const int32_t unroll_factors[] = {8, 4, 2};
        const size_t num_factors = sizeof(unroll_factors) / sizeof(unroll_factors[0]);
        int32_t start[num_factors], end[num_factors];
        // Calculate start and end indices
        int32_t prev_end = sl_start;
        for (int i = 0; i < num_factors; ++i) {
            start[i] = prev_end;
            end[i] = sl_end - sl_length % unroll_factors[i];
            prev_end = end[i];
        }

        // Lambda for unroll call
        auto call_unroll = [&](int unroll, int sl_start, int sl_end) {
            switch (unroll) {
                case 8:
                    compute_grad_sum_unweighted_unroll<grad_t,cache_t,kFixedMaxVecsPerThread,kThreadGroupSize,VEC_WIDTH,8>(
                        grad_sum, grad_output,
                        D_offsets,
                        D, T, sorted_infos, 
                        info_B_num_bits, info_B_mask,
                        segment_start, sl_start, sl_end, shfl_sync_mask, vec_start
                    );
                    break;
                case 4:
                    compute_grad_sum_unweighted_unroll<grad_t,cache_t,kFixedMaxVecsPerThread,kThreadGroupSize,VEC_WIDTH,4>(
                        grad_sum, grad_output,
                        D_offsets,
                        D, T, sorted_infos, 
                        info_B_num_bits, info_B_mask,
                        segment_start, sl_start, sl_end, shfl_sync_mask, vec_start
                    );
                    break;
                case 2:
                    compute_grad_sum_unweighted_unroll<grad_t,cache_t,kFixedMaxVecsPerThread,kThreadGroupSize,VEC_WIDTH,2>(
                        grad_sum, grad_output,
                        D_offsets,
                        D, T, sorted_infos, 
                        info_B_num_bits, info_B_mask,
                        segment_start, sl_start, sl_end, shfl_sync_mask, vec_start
                    );
                    break;
                case 1:
                    compute_grad_sum_unweighted_unroll<grad_t,cache_t,kFixedMaxVecsPerThread,kThreadGroupSize,VEC_WIDTH,1>(
                        grad_sum, grad_output,
                        D_offsets,
                        D, T, sorted_infos, 
                        info_B_num_bits, info_B_mask,
                        segment_start, sl_start, sl_end, shfl_sync_mask, vec_start
                    );
                    break;
            }
        };

        for (int i = 0; i < num_factors; ++i) {
            call_unroll(unroll_factors[i], start[i], end[i]);
        }
       
        call_unroll(1, end[num_factors-1], sl_end);

        if (smem_grad_sum) {
            // Store grad_sum in smem_grad_sum
            #pragma unroll kFixedMaxVecsPerThread
            for (int32_t vec = 0;
                 (vec < kFixedMaxVecsPerThread) && ((vec + vec_start) * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                 ++vec) {
                const int32_t d_vec = ((vec + vec_start) * kThreadGroupSize + threadIdx.x);
                smem_grad_sum[d_vec] = grad_sum[vec];
            }
        }
    }
}

    // clang-format on
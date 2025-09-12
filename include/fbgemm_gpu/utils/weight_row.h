/*******************************************************************************
 * Copyright (c) 2016 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>

 #include "fbgemm_gpu/utils/cuda_prelude.cuh"
 #include "fbgemm_gpu/utils/stochastic_rounding.h"
 #include "fbgemm_gpu/utils/vec2.h"
 #include "fbgemm_gpu/utils/stochastic_rounding.h"

namespace fbgemm_gpu::rocm {
template <typename dst_t, typename src_t>
DEVICE_INLINE void quantize_store(
    dst_t* output,
    const Vec2T<src_t>& value,
    StochasticRoundingRNGState* state,
    const float2 qparams) {
  if (!state) {
    nearest_rounding_vector<dst_t, src_t>(output, value, qparams);
  } else {
    stochastic_rounding_vector<dst_t, src_t>(output, value, *state, qparams);
  }
}

template <typename emb_t>
DEVICE_INLINE float2 load_qparams_from_row(const emb_t* row) {
  if constexpr (std::is_same_v<emb_t, uint8_t>) {
    // For uint8_t, qparams are stored at the end of the row
    const float* qparams_ptr = reinterpret_cast<const float*>(row);
    return make_float2(qparams_ptr[0], qparams_ptr[1]);
  } else {
    return make_float2(0.0f, 0.0f);
  }
}

 template <typename dst_t, typename src_t>
 DEVICE_INLINE Vec2T<dst_t> dequantize_load(
     const src_t* value,
     const float2 /* unused */) {
   return Vec2T<dst_t>(value);
 }

 template <>
 DEVICE_INLINE Vec2T<float> dequantize_load(
     const uint8_t* value,
     const float2 qparams) {
   Vec2T<float> out;
   out.acc.x = value[0] * qparams.x + qparams.y;
   out.acc.y = value[1] * qparams.x + qparams.y;

   return out;
 }

 template <>
 DEVICE_INLINE Vec2T<at::Half> dequantize_load(
     const uint8_t* value,
     const float2 qparams) {
   Vec2T<at::Half> out;
   out.acc.x = value[0] * qparams.x + qparams.y;
   out.acc.y = value[1] * qparams.x + qparams.y;

   return out;
 }

 ////////////////////////////////////////////////////////////////////////////////
 // Weight Row Accessor for Vec2T
 ////////////////////////////////////////////////////////////////////////////////

 template <typename emb_t, typename cache_t, typename dst_t, bool uses_cache>
 struct WeightRowAccessorVec2 {
   const emb_t* row_;
   const cache_t* cache_row_;
   const int dim_;

   DEVICE_INLINE
   WeightRowAccessorVec2(
       const emb_t* row,
       const cache_t* cache_row,
       const int dim)
       : row_(row), cache_row_(cache_row), dim_(dim) {}

   DEVICE_INLINE Vec2T<dst_t> load(const int32_t d, const float2 qparams) const {
     if constexpr (uses_cache) {
       return rocm::dequantize_load<dst_t, cache_t>(cache_row_ + d, qparams);
     } else {
       return rocm::dequantize_load<dst_t, emb_t>(row_ + d, qparams);
     }
   }

   DEVICE_INLINE float2 load_qparams() const {
     if constexpr (std::is_same_v<emb_t, uint8_t>) {
       return load_qparams_from_row<emb_t>(row_ + dim_);
     } else {
       return make_float2(0.0f, 0.0f);
     }
  }
};

} // namespace fbgemm_gpu::rocm

namespace fbgemm_gpu {

////////////////////////////////////////////////////////////////////////////////
// WeightRow - Main weight row accessor class
////////////////////////////////////////////////////////////////////////////////

template <typename emb_t, typename cache_t, typename dst_t>
struct WeightRow {
  const emb_t* weights_;
  const cache_t* cache_weights_;
  const int dim_;
  const bool stochastic_rounding_;
  const void* stochastic_rounding_args_;
  const int thread_id_;

  DEVICE_INLINE WeightRow(
      const emb_t* weights,
      const cache_t* cache_weights,
      const int dim,
      const bool stochastic_rounding,
      const void* stochastic_rounding_args,
      const int thread_id)
      : weights_(weights),
        cache_weights_(cache_weights),
        dim_(dim),
        stochastic_rounding_(stochastic_rounding),
        stochastic_rounding_args_(stochastic_rounding_args),
        thread_id_(thread_id) {}

  DEVICE_INLINE float2 load_qparams() const {
    if constexpr (std::is_same_v<emb_t, uint8_t>) {
      return rocm::load_qparams_from_row<emb_t>(weights_ + dim_);
    } else {
      return make_float2(0.0f, 0.0f);
    }
  }

  // Load method that returns Vec4TAcc<cache_t>
  DEVICE_INLINE Vec4TAcc<cache_t> load(int d, const float2& qparams) const {
    Vec4TAcc<cache_t> result;
    // Load 4 elements at offset d
    const emb_t* src = weights_ + d;
    if constexpr (std::is_same_v<emb_t, float>) {
      result.acc.x = *reinterpret_cast<const float*>(src);
      result.acc.y = *reinterpret_cast<const float*>(src + 1);
      result.acc.z = *reinterpret_cast<const float*>(src + 2);
      result.acc.w = *reinterpret_cast<const float*>(src + 3);
    } else if constexpr (std::is_same_v<emb_t, c10::Half>) {
      result.acc.x = static_cast<float>(*reinterpret_cast<const c10::Half*>(src));
      result.acc.y = static_cast<float>(*reinterpret_cast<const c10::Half*>(src + 1));
      result.acc.z = static_cast<float>(*reinterpret_cast<const c10::Half*>(src + 2));
      result.acc.w = static_cast<float>(*reinterpret_cast<const c10::Half*>(src + 3));
    } else if constexpr (std::is_same_v<emb_t, uint8_t>) {
      // Dequantize uint8_t values
      result.acc.x = static_cast<float>(src[0]) * qparams.x + qparams.y;
      result.acc.y = static_cast<float>(src[1]) * qparams.x + qparams.y;
      result.acc.z = static_cast<float>(src[2]) * qparams.x + qparams.y;
      result.acc.w = static_cast<float>(src[3]) * qparams.x + qparams.y;
    } else {
      result.acc.x = result.acc.y = result.acc.z = result.acc.w = 0;
    }
    return result;
  }

  // Store method that accepts Vec4TAcc<cache_t>
  DEVICE_INLINE void store(const Vec4TAcc<cache_t>& value, int d, const float2& qparams) const {
    emb_t* dst = const_cast<emb_t*>(weights_ + d);
    if constexpr (std::is_same_v<emb_t, float>) {
      *reinterpret_cast<float*>(dst) = value.acc.x;
      *reinterpret_cast<float*>(dst + 1) = value.acc.y;
      *reinterpret_cast<float*>(dst + 2) = value.acc.z;
      *reinterpret_cast<float*>(dst + 3) = value.acc.w;
    } else if constexpr (std::is_same_v<emb_t, c10::Half>) {
      *reinterpret_cast<c10::Half*>(dst) = static_cast<c10::Half>(value.acc.x);
      *reinterpret_cast<c10::Half*>(dst + 1) = static_cast<c10::Half>(value.acc.y);
      *reinterpret_cast<c10::Half*>(dst + 2) = static_cast<c10::Half>(value.acc.z);
      *reinterpret_cast<c10::Half*>(dst + 3) = static_cast<c10::Half>(value.acc.w);
    } else if constexpr (std::is_same_v<emb_t, uint8_t>) {
      // Quantize to uint8_t
      dst[0] = static_cast<emb_t>(max(0.0f, min(255.0f, (value.acc.x - qparams.y) / qparams.x)));
      dst[1] = static_cast<emb_t>(max(0.0f, min(255.0f, (value.acc.y - qparams.y) / qparams.x)));
      dst[2] = static_cast<emb_t>(max(0.0f, min(255.0f, (value.acc.z - qparams.y) / qparams.x)));
      dst[3] = static_cast<emb_t>(max(0.0f, min(255.0f, (value.acc.w - qparams.y) / qparams.x)));
    }
  }

  template <typename OptimizerState>
  DEVICE_INLINE OptimizerState* optimizer_state_ptr() const {
    // Return a pointer to optimizer state - this is a placeholder implementation
    return nullptr;
  }
};

} // namespace fbgemm_gpu

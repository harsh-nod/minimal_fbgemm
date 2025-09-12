
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

 #include <hip/hip_fp16.h>

 #include <ATen/ATen.h>

#include "fbgemm_gpu/utils/half2.h"
#include "fbgemm_gpu/utils/vec2.h"
#include "fbgemm_gpu/utils/types.h"

// Forward declarations and definitions for stochastic rounding
namespace fbgemm_gpu::rocm {

struct StochasticRoundingRNGState {
  uint64_t seed;
  uint64_t offset;

  DEVICE_INLINE float4 rand4() {
    // Simple random number generation for stochastic rounding
    // This is a simplified implementation
    float4 result;
    result.x = static_cast<float>(seed & 0xFFFF) / 65536.0f;
    result.y = static_cast<float>((seed >> 16) & 0xFFFF) / 65536.0f;
    result.z = static_cast<float>((seed >> 32) & 0xFFFF) / 65536.0f;
    result.w = static_cast<float>((seed >> 48) & 0xFFFF) / 65536.0f;

    // Update seed (simple LCG)
    seed = seed * 1103515245ULL + 12345ULL;
    return result;
  }
};

DEVICE_INLINE at::Half stochastic_rounding_scalar(float value, float random) {
  float rounded = floorf(value + random);
  return static_cast<at::Half>(rounded);
}

DEVICE_INLINE uint8_t stochastic_rounding_scalar_uint8(float value, float random) {
  float rounded = floorf(value + random);
  return static_cast<uint8_t>(max(0.0f, min(255.0f, rounded)));
}
 template <typename dst_t, typename src_t>
 DEVICE_INLINE void stochastic_rounding_vector(
     dst_t* output,
     const Vec2T<src_t>& value,
     StochasticRoundingRNGState& state,
     const float2 /* not used */) {
   value.store(output);
 }

 template <>
 DEVICE_INLINE void stochastic_rounding_vector(
     at::Half* output,
     const Vec2T<at::Half>& value,
     StochasticRoundingRNGState& state,
     const float2 /* not used */) {
   const auto random_bits = state.rand4();
   Half2 v;
   v.a = __halves2half2(
       stochastic_rounding_scalar(value.acc.x, random_bits.x),
       stochastic_rounding_scalar(value.acc.y, random_bits.y));

   v.store(output);
 }

 template <>
 DEVICE_INLINE void stochastic_rounding_vector(
     at::Half* output,
     const Vec2T<float>& value,
     StochasticRoundingRNGState& state,
     const float2 /* not used */) {
   const auto random_bits = state.rand4();
   Half2 v;
   v.a = __halves2half2(
       stochastic_rounding_scalar(value.acc.x, random_bits.x),
       stochastic_rounding_scalar(value.acc.y, random_bits.y));

   v.store(output);
 }

 template <>
 DEVICE_INLINE void stochastic_rounding_vector(
     uint8_t* output,
     const Vec2T<float>& value,
     StochasticRoundingRNGState& state,
     const float2 qparams) {
   const auto random_bits = state.rand4();
   const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
   output[0] = stochastic_rounding_scalar_uint8(
       (value.acc.x - qparams.y) * inv_scale, random_bits.x);
   output[1] = stochastic_rounding_scalar_uint8(
       (value.acc.y - qparams.y) * inv_scale, random_bits.y);
 }

 template <>
 DEVICE_INLINE void stochastic_rounding_vector(
     uint8_t* output,
     const Vec2T<at::Half>& value,
     StochasticRoundingRNGState& state,
     const float2 qparams) {
   const auto random_bits = state.rand4();
   const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
   output[0] = stochastic_rounding_scalar_uint8(
       (value.acc.x - qparams.y) * inv_scale, random_bits.x);
   output[1] = stochastic_rounding_scalar_uint8(
       (value.acc.y - qparams.y) * inv_scale, random_bits.y);
 }

 template <typename dst_t, typename src_t>
 DEVICE_INLINE void nearest_rounding_vector(
     dst_t* output,
     const Vec2T<src_t>& value,
     const float2 /* not used */) {
   value.store(output);
 }

 template <>
 DEVICE_INLINE void nearest_rounding_vector(
     uint8_t* output,
     const Vec2T<float>& value,
     const float2 qparams) {
   const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
   output[0] = lrintf((value.acc.x - qparams.y) * inv_scale);
   output[1] = lrintf((value.acc.y - qparams.y) * inv_scale);
 }

 template <>
 DEVICE_INLINE void nearest_rounding_vector(
     uint8_t* output,
     const Vec2T<at::Half>& value,
     const float2 qparams) {
   const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
   output[0] = lrintf((value.acc.x - qparams.y) * inv_scale);
   output[1] = lrintf((value.acc.y - qparams.y) * inv_scale);
 }

 } // namespace fbgemm_gpu::rocm

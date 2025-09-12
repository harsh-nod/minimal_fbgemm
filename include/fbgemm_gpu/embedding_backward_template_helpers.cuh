/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <c10/hip/HIPGuard.h>
#include <ATen/native/hip/Math.cuh>
#include <ATen/hip/detail/PhiloxCudaStateRaw.cuh>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <mutex>

#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/find_qparams.cuh"
#include "fbgemm_gpu/utils/fixed_divisor.cuh"
#include "fbgemm_gpu/utils/shared_memory.cuh"
#include "fbgemm_gpu/utils/tensor_utils.h"
#include "fbgemm_gpu/utils/vec4.cuh"
#include "fbgemm_gpu/utils/weight_row.h"

#define SHFL_SYNC(val, srcLane) \
  shfl_sync(val, srcLane, kThreadGroupSize, shfl_sync_mask)

#define DEVICE_INLINE __device__ __forceinline__

// Constants are defined in cuda_prelude.cuh

// Vec4TAcc and SharedMemory are defined in vec4.cuh and shared_memory.cuh respectively

// Utility functions
__device__ __forceinline__ int div_round_up(int a, int b) {
    return (a + b - 1) / b;
}

template<typename T>
__device__ __forceinline__ T gpuAtomicAdd(T* address, T val) {
    return atomicAdd(address, val);
}

template<>
__device__ __forceinline__ float gpuAtomicAdd<float>(float* address, float val) {
    return atomicAdd(address, val);
}

template<>
__device__ __forceinline__ double gpuAtomicAdd<double>(double* address, double val) {
    return atomicAdd(address, val);
}

// shfl_sync function is defined in cuda_prelude.cuh

// Vec4TAcc constructor function
template<typename T>
__device__ __forceinline__ fbgemm_gpu::Vec4TAcc<T> vec4_acc(T x, T y, T z, T w) {
    return fbgemm_gpu::Vec4TAcc<T>(x, y, z, w);
}

template<typename T>
__device__ __forceinline__ fbgemm_gpu::Vec4TAcc<T> vec4_acc(T x, T y) {
    return fbgemm_gpu::Vec4TAcc<T>(x, y, 0, 0);
}

// PlacementType enum
enum class PlacementType : int32_t {
    DEVICE = 0,
    MANAGED = 1,
    MANAGED_CACHING = 2
};

// Constants are defined in cuda_prelude.cuh

constexpr size_t kBackwardMaxThreads = 512;
constexpr int32_t kCacheLocationMissing = -1;

DEVICE_INLINE int64_t gpuAtomicIncrement(int64_t* p) {
  static_assert(
      sizeof(int64_t) == sizeof(unsigned long long),
      "expected int64_t to be unsigned long long");
  return static_cast<int64_t>(atomicAdd(
      reinterpret_cast<unsigned long long int*>(p),
      static_cast<unsigned long long int>(1)));
}

namespace fbgemm_gpu {
namespace {

// Based on the empirical study, max grid size that is 64x larger than the
// number of SMs gives good performance across the board
constexpr int MAX_THREAD_BLOCKS_FACTOR = 64;

inline int get_max_thread_blocks_() {
  return MAX_THREAD_BLOCKS_FACTOR * 64; // Simplified for testing
}
} // namespace
} // namespace fbgemm_gpu

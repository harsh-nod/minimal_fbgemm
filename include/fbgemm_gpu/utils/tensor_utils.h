#pragma once

namespace fbgemm_gpu {
namespace utils {

// Simplified tensor utilities
template<typename T>
__device__ __host__ T* get_tensor_data_ptr(T* ptr) {
    return ptr;
}

template<typename T>
__device__ __host__ const T* get_tensor_data_ptr(const T* ptr) {
    return ptr;
}

} // namespace utils
} // namespace fbgemm_gpu
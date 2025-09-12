#pragma once

namespace fbgemm_gpu {
namespace utils {

// Simplified quantization parameters
template<typename T>
__device__ __host__ void find_qparams(T min_val, T max_val, T& scale, int32_t& zero_point) {
    // Simplified implementation
    scale = (max_val - min_val) / 255.0f;
    zero_point = static_cast<int32_t>(-min_val / scale);
}

} // namespace utils
} // namespace fbgemm_gpu

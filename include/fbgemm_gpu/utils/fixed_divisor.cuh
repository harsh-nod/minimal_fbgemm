#pragma once

namespace fbgemm_gpu {
namespace utils {

// Simplified fixed divisor
template<typename T>
class FixedDivisor {
public:
    FixedDivisor(T divisor) : divisor_(divisor) {}

    __device__ __host__ T div(T dividend) const {
        return dividend / divisor_;
    }

    __device__ __host__ T mod(T dividend) const {
        return dividend % divisor_;
    }

private:
    T divisor_;
};

} // namespace utils
} // namespace fbgemm_gpu

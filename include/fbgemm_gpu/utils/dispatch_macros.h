#pragma once

namespace fbgemm_gpu {
namespace utils {

// Simplified dispatch macros
#define FBGEMM_GPU_DISPATCH_FLOAT_TYPES(TYPE, NAME, ...) \
    [&] { \
        switch (TYPE) { \
            case ScalarType::Float: { \
                using scalar_t = float; \
                return __VA_ARGS__(); \
            } \
            case ScalarType::Double: { \
                using scalar_t = double; \
                return __VA_ARGS__(); \
            } \
            default: \
                AT_ERROR("Unsupported scalar type"); \
        } \
    }()

} // namespace utils
} // namespace fbgemm_gpu
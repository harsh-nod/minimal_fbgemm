#pragma once

namespace fbgemm_gpu {

// Simplified embedding common definitions
enum class EmbeddingSpMDMWeightChoice {
    UNIFORM,
    VARIABLE
};

enum class EmbeddingSpMDMCornerCase {
    NONE,
    EMPTY_INDICES,
    EMPTY_WEIGHTS
};

} // namespace fbgemm_gpu

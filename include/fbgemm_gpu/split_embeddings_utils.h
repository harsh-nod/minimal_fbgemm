#pragma once

namespace fbgemm_gpu {

// Simplified split embeddings utilities
struct SplitEmbeddingArgs {
    int32_t batch_size;
    int32_t num_tables;
    int32_t embedding_dim;
};

} // namespace fbgemm_gpu

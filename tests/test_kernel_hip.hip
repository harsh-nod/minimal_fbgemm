/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <memory>
#include <algorithm>

// Include ATen first
#include <ATen/ATen.h>

// Include the kernel
#include "../src/gen_embedding_backward_rowwise_adagrad_split_unweighted_kernel_cta.hip"

// Helper function to check HIP errors
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Helper function to generate random data
template<typename T>
void generateRandomData(std::vector<T>& data, T min_val = -1.0f, T max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min_val, max_val);

    for (auto& val : data) {
        val = dis(gen);
    }
}

template<>
void generateRandomData<int32_t>(std::vector<int32_t>& data, int32_t min_val, int32_t max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dis(min_val, max_val);

    for (auto& val : data) {
        val = dis(gen);
    }
}

template<>
void generateRandomData<int64_t>(std::vector<int64_t>& data, int64_t min_val, int64_t max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dis(min_val, max_val);

    for (auto& val : data) {
        val = dis(gen);
    }
}

template<>
void generateRandomData<uint32_t>(std::vector<uint32_t>& data, uint32_t min_val, uint32_t max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(min_val, max_val);

    for (auto& val : data) {
        val = dis(gen);
    }
}

int main() {
    std::cout << "Testing split_embedding_backward_codegen_rowwise_adagrad_unweighted_kernel_cta_per_row_1 on AMD GPU" << std::endl;

    // Use direct HIP API to check GPU availability
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!" << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " HIP devices" << std::endl;

    HIP_CHECK(hipSetDevice(0));
    std::cout << "Using device: AMD Instinct MI300X" << std::endl;
    std::cout << "Compute capability: 11.0" << std::endl;

    // Test parameters
    const int32_t B = 32;           // batch size
    const int32_t T = 4;            // number of tables
    const int32_t D = 64;           // embedding dimension
    const int32_t num_indices = 1000; // number of indices
    const int32_t cache_size = 100; // cache size

    std::cout << "Test parameters:" << std::endl;
    std::cout << "  Batch size (B): " << B << std::endl;
    std::cout << "  Number of tables (T): " << T << std::endl;
    std::cout << "  Embedding dimension (D): " << D << std::endl;
    std::cout << "  Number of indices: " << num_indices << std::endl;
    std::cout << "  Cache size: " << cache_size << std::endl;

    // Allocate host memory
    std::vector<float> grad_output_host(B * D);
    std::vector<float> dev_weights_host(T * D * 1000); // Large enough for indices
    std::vector<float> uvm_weights_host(T * D * 1000);
    std::vector<float> lxu_cache_weights_host(cache_size * D);

    std::vector<int32_t> weights_placements_host(T);
    std::vector<int64_t> weights_offsets_host(T);
    std::vector<int32_t> D_offsets_host(T);
    std::vector<int64_t> hash_size_cumsum_host(T);
    std::vector<int32_t> sorted_linear_indices_run_host(num_indices);
    std::vector<int32_t> sorted_linear_indices_cumulative_run_lengths_host(num_indices);
    std::vector<int32_t> long_run_ids_host(num_indices);
    std::vector<int32_t> num_long_run_ids_host(1);
    std::vector<int32_t> sorted_infos_host(num_indices);
    std::vector<int32_t> sorted_lxu_cache_locations_host(num_indices);
    std::vector<int32_t> table_unique_indices_offsets_host(T);
    std::vector<int32_t> long_run_id_to_really_long_run_ids_host(num_indices);
    std::vector<float> temp_grad_accum_host(B * D);
    std::vector<int32_t> grad_accum_counter_host(1);
    std::vector<float> momentum1_dev_host(T * D * 1000);
    std::vector<float> momentum1_uvm_host(T * D * 1000);
    std::vector<int32_t> momentum1_placements_host(T);
    std::vector<int64_t> momentum1_offsets_host(T);

    // Generate random data
    std::cout << "Generating random test data..." << std::endl;
    generateRandomData(grad_output_host, -0.1f, 0.1f);
    generateRandomData(dev_weights_host, -0.5f, 0.5f);
    generateRandomData(uvm_weights_host, -0.5f, 0.5f);
    generateRandomData(lxu_cache_weights_host, -0.5f, 0.5f);

    generateRandomData(weights_placements_host, 0, 2); // PlacementType values
    generateRandomData(weights_offsets_host, 0L, 1000L);
    generateRandomData(D_offsets_host, 0, D);
    generateRandomData(hash_size_cumsum_host, 0L, 1000L);
    generateRandomData(sorted_linear_indices_run_host, 0, num_indices-1);
    generateRandomData(sorted_linear_indices_cumulative_run_lengths_host, 0, num_indices);
    generateRandomData(long_run_ids_host, 0, num_indices-1);
    generateRandomData(num_long_run_ids_host, 0, num_indices);
    generateRandomData(sorted_infos_host, 0, (1 << 20) - 1); // Large range for info
    generateRandomData(sorted_lxu_cache_locations_host, -1, cache_size-1); // -1 for missing
    generateRandomData(table_unique_indices_offsets_host, 0, num_indices);
    generateRandomData(long_run_id_to_really_long_run_ids_host, 0, num_indices-1);
    generateRandomData(temp_grad_accum_host, -0.1f, 0.1f);
    generateRandomData(grad_accum_counter_host, 0, 100);
    generateRandomData(momentum1_dev_host, 0.0f, 1.0f);
    generateRandomData(momentum1_uvm_host, 0.0f, 1.0f);
    generateRandomData(momentum1_placements_host, 0, 2);
    generateRandomData(momentum1_offsets_host, 0L, 1000L);

    // Set up some meaningful values
    for (int i = 0; i < T; ++i) {
        weights_offsets_host[i] = i * D * 1000;
        D_offsets_host[i] = i * D;
        hash_size_cumsum_host[i] = i * 1000;
        momentum1_offsets_host[i] = i * D * 1000;
    }

    num_long_run_ids_host[0] = std::min(10, num_indices);
    grad_accum_counter_host[0] = 0;

    // Allocate device memory
    std::cout << "Allocating device memory..." << std::endl;

    float* grad_output_dev;
    float* dev_weights_dev;
    float* uvm_weights_dev;
    float* lxu_cache_weights_dev;
    int32_t* weights_placements_dev;
    int64_t* weights_offsets_dev;
    int32_t* D_offsets_dev;
    int64_t* hash_size_cumsum_dev;
    int32_t* sorted_linear_indices_run_dev;
    int32_t* sorted_linear_indices_cumulative_run_lengths_dev;
    int32_t* long_run_ids_dev;
    int32_t* num_long_run_ids_dev;
    int32_t* sorted_infos_dev;
    int32_t* sorted_lxu_cache_locations_dev;
    int32_t* table_unique_indices_offsets_dev;
    int32_t* long_run_id_to_really_long_run_ids_dev;
    float* temp_grad_accum_dev;
    int32_t* grad_accum_counter_dev;
    float* momentum1_dev_dev;
    float* momentum1_uvm_dev;
    int32_t* momentum1_placements_dev;
    int64_t* momentum1_offsets_dev;

    HIP_CHECK(hipMalloc(&grad_output_dev, grad_output_host.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dev_weights_dev, dev_weights_host.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&uvm_weights_dev, uvm_weights_host.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&lxu_cache_weights_dev, lxu_cache_weights_host.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&weights_placements_dev, weights_placements_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&weights_offsets_dev, weights_offsets_host.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&D_offsets_dev, D_offsets_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&hash_size_cumsum_dev, hash_size_cumsum_host.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&sorted_linear_indices_run_dev, sorted_linear_indices_run_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&sorted_linear_indices_cumulative_run_lengths_dev, sorted_linear_indices_cumulative_run_lengths_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&long_run_ids_dev, long_run_ids_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&num_long_run_ids_dev, num_long_run_ids_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&sorted_infos_dev, sorted_infos_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&sorted_lxu_cache_locations_dev, sorted_lxu_cache_locations_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&table_unique_indices_offsets_dev, table_unique_indices_offsets_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&long_run_id_to_really_long_run_ids_dev, long_run_id_to_really_long_run_ids_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&temp_grad_accum_dev, temp_grad_accum_host.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&grad_accum_counter_dev, grad_accum_counter_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&momentum1_dev_dev, momentum1_dev_host.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&momentum1_uvm_dev, momentum1_uvm_host.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&momentum1_placements_dev, momentum1_placements_host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&momentum1_offsets_dev, momentum1_offsets_host.size() * sizeof(int64_t)));

    // Copy data to device
    std::cout << "Copying data to device..." << std::endl;
    HIP_CHECK(hipMemcpy(grad_output_dev, grad_output_host.data(), grad_output_host.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_weights_dev, dev_weights_host.data(), dev_weights_host.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(uvm_weights_dev, uvm_weights_host.data(), uvm_weights_host.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(lxu_cache_weights_dev, lxu_cache_weights_host.data(), lxu_cache_weights_host.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(weights_placements_dev, weights_placements_host.data(), weights_placements_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(weights_offsets_dev, weights_offsets_host.data(), weights_offsets_host.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(D_offsets_dev, D_offsets_host.data(), D_offsets_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(hash_size_cumsum_dev, hash_size_cumsum_host.data(), hash_size_cumsum_host.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(sorted_linear_indices_run_dev, sorted_linear_indices_run_host.data(), sorted_linear_indices_run_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(sorted_linear_indices_cumulative_run_lengths_dev, sorted_linear_indices_cumulative_run_lengths_host.data(), sorted_linear_indices_cumulative_run_lengths_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(long_run_ids_dev, long_run_ids_host.data(), long_run_ids_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(num_long_run_ids_dev, num_long_run_ids_host.data(), num_long_run_ids_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(sorted_infos_dev, sorted_infos_host.data(), sorted_infos_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(sorted_lxu_cache_locations_dev, sorted_lxu_cache_locations_host.data(), sorted_lxu_cache_locations_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(table_unique_indices_offsets_dev, table_unique_indices_offsets_host.data(), table_unique_indices_offsets_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(long_run_id_to_really_long_run_ids_dev, long_run_id_to_really_long_run_ids_host.data(), long_run_id_to_really_long_run_ids_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(temp_grad_accum_dev, temp_grad_accum_host.data(), temp_grad_accum_host.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(grad_accum_counter_dev, grad_accum_counter_host.data(), grad_accum_counter_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(momentum1_dev_dev, momentum1_dev_host.data(), momentum1_dev_host.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(momentum1_uvm_dev, momentum1_uvm_host.data(), momentum1_uvm_host.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(momentum1_placements_dev, momentum1_placements_host.data(), momentum1_placements_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(momentum1_offsets_dev, momentum1_offsets_host.data(), momentum1_offsets_host.size() * sizeof(int64_t), hipMemcpyHostToDevice));

    // Create tensor accessors
    std::cout << "Creating tensor accessors..." << std::endl;

    // Create size and stride arrays
    int64_t grad_output_sizes[2] = {B, D};
    int64_t grad_output_strides[2] = {D, 1};
    auto grad_output_acc = at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits>(
        grad_output_dev, grad_output_sizes, grad_output_strides);

    int64_t dev_weights_size[1] = {static_cast<int64_t>(dev_weights_host.size())};
    int64_t dev_weights_stride[1] = {1};
    auto dev_weights_acc = at::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits>(
        dev_weights_dev, dev_weights_size, dev_weights_stride);

    int64_t uvm_weights_size[1] = {static_cast<int64_t>(uvm_weights_host.size())};
    int64_t uvm_weights_stride[1] = {1};
    auto uvm_weights_acc = at::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits>(
        uvm_weights_dev, uvm_weights_size, uvm_weights_stride);

    int64_t lxu_cache_sizes[2] = {cache_size, D};
    int64_t lxu_cache_strides[2] = {D, 1};
    auto lxu_cache_weights_acc = at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits>(
        lxu_cache_weights_dev, lxu_cache_sizes, lxu_cache_strides);

    int32_t weights_placements_size[1] = {T};
    int32_t weights_placements_stride[1] = {1};
    auto weights_placements_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        weights_placements_dev, weights_placements_size, weights_placements_stride);

    int32_t weights_offsets_size[1] = {T};
    int32_t weights_offsets_stride[1] = {1};
    auto weights_offsets_acc = at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>(
        weights_offsets_dev, weights_offsets_size, weights_offsets_stride);

    int32_t D_offsets_size[1] = {T};
    int32_t D_offsets_stride[1] = {1};
    auto D_offsets_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        D_offsets_dev, D_offsets_size, D_offsets_stride);

    int32_t hash_size_cumsum_size[1] = {T};
    int32_t hash_size_cumsum_stride[1] = {1};
    auto hash_size_cumsum_acc = at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>(
        hash_size_cumsum_dev, hash_size_cumsum_size, hash_size_cumsum_stride);
    int32_t sorted_linear_indices_run_size[1] = {num_indices};
    int32_t sorted_linear_indices_run_stride[1] = {1};
    auto sorted_linear_indices_run_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        sorted_linear_indices_run_dev, sorted_linear_indices_run_size, sorted_linear_indices_run_stride);

    int32_t sorted_linear_indices_cumulative_run_lengths_size[1] = {num_indices};
    int32_t sorted_linear_indices_cumulative_run_lengths_stride[1] = {1};
    auto sorted_linear_indices_cumulative_run_lengths_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        sorted_linear_indices_cumulative_run_lengths_dev, sorted_linear_indices_cumulative_run_lengths_size, sorted_linear_indices_cumulative_run_lengths_stride);

    int32_t long_run_ids_size[1] = {num_indices};
    int32_t long_run_ids_stride[1] = {1};
    auto long_run_ids_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        long_run_ids_dev, long_run_ids_size, long_run_ids_stride);

    int32_t num_long_run_ids_size[1] = {1};
    int32_t num_long_run_ids_stride[1] = {1};
    auto num_long_run_ids_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        num_long_run_ids_dev, num_long_run_ids_size, num_long_run_ids_stride);

    int32_t sorted_infos_size[1] = {num_indices};
    int32_t sorted_infos_stride[1] = {1};
    auto sorted_infos_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        sorted_infos_dev, sorted_infos_size, sorted_infos_stride);

    int32_t sorted_lxu_cache_locations_size[1] = {num_indices};
    int32_t sorted_lxu_cache_locations_stride[1] = {1};
    auto sorted_lxu_cache_locations_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        sorted_lxu_cache_locations_dev, sorted_lxu_cache_locations_size, sorted_lxu_cache_locations_stride);

    int32_t table_unique_indices_offsets_size[1] = {T};
    int32_t table_unique_indices_offsets_stride[1] = {1};
    auto table_unique_indices_offsets_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        table_unique_indices_offsets_dev, table_unique_indices_offsets_size, table_unique_indices_offsets_stride);

    int32_t long_run_id_to_really_long_run_ids_size[1] = {num_indices};
    int32_t long_run_id_to_really_long_run_ids_stride[1] = {1};
    auto long_run_id_to_really_long_run_ids_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        long_run_id_to_really_long_run_ids_dev, long_run_id_to_really_long_run_ids_size, long_run_id_to_really_long_run_ids_stride);
    int32_t temp_grad_accum_sizes[2] = {B, D};
    int32_t temp_grad_accum_strides[2] = {D, 1};
    auto temp_grad_accum_acc = at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits>(
        temp_grad_accum_dev, temp_grad_accum_sizes, temp_grad_accum_strides);

    int32_t grad_accum_counter_sizes[1] = {1};
    int32_t grad_accum_counter_strides[1] = {1};
    auto grad_accum_counter_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        grad_accum_counter_dev, grad_accum_counter_sizes, grad_accum_counter_strides);
    int64_t momentum1_dev_sizes[1] = {static_cast<int64_t>(momentum1_dev_host.size())};
    int64_t momentum1_dev_strides[1] = {1};
    auto momentum1_dev_acc = at::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits>(
        momentum1_dev_dev, momentum1_dev_sizes, momentum1_dev_strides);

    int64_t momentum1_uvm_sizes[1] = {static_cast<int64_t>(momentum1_uvm_host.size())};
    int64_t momentum1_uvm_strides[1] = {1};
    auto momentum1_uvm_acc = at::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits>(
        momentum1_uvm_dev, momentum1_uvm_sizes, momentum1_uvm_strides);
    int32_t momentum1_placements_sizes[1] = {T};
    int32_t momentum1_placements_strides[1] = {1};
    auto momentum1_placements_acc = at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>(
        momentum1_placements_dev, momentum1_placements_sizes, momentum1_placements_strides);

    int32_t momentum1_offsets_sizes[1] = {T};
    int32_t momentum1_offsets_strides[1] = {1};
    auto momentum1_offsets_acc = at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>(
        momentum1_offsets_dev, momentum1_offsets_sizes, momentum1_offsets_strides);

    // Kernel parameters
    const bool use_uniq_cache_locations = false;
    const bool stochastic_rounding = false;
    const at::PhiloxCudaState stochastic_rounding_philox_args(0ULL, 0ULL);
    const int32_t info_B_num_bits = 20;
    const uint32_t info_B_mask = (1U << info_B_num_bits) - 1;
    const int32_t max_segment_length_per_cta = 32;
    const bool use_deterministic_algorithms = false;
    const int32_t max_vecs_per_thread = 2;
    const float learning_rate = 0.01f;
    const float eps = 1e-8f;
    const float weight_decay = 0.0f;
    const int64_t weight_decay_mode = 0;
    const float max_norm = 0.0f;

    // Launch kernel
    std::cout << "Launching kernel on AMD GPU..." << std::endl;

    dim3 blockDim(64, 4); // kThreadGroupSize=64, warp_id dimension
    dim3 gridDim(1, 1);   // Single block for testing

    auto start_time = std::chrono::high_resolution_clock::now();

    // Call the kernel with template parameters: float, float, float, int32_t, 2, 64, true
    split_embedding_backward_codegen_rowwise_adagrad_unweighted_kernel_cta_per_row_1
        <float, float, float, int32_t, 2, 64, true>
        <<<gridDim, blockDim>>>(
            grad_output_acc,
            dev_weights_acc,
            uvm_weights_acc,
            lxu_cache_weights_acc,
            weights_placements_acc,
            weights_offsets_acc,
            D_offsets_acc,
            hash_size_cumsum_acc,
            sorted_linear_indices_run_acc,
            sorted_linear_indices_cumulative_run_lengths_acc,
            long_run_ids_acc,
            num_long_run_ids_acc,
            sorted_infos_acc,
            sorted_lxu_cache_locations_acc,
            use_uniq_cache_locations,
            table_unique_indices_offsets_acc,
            stochastic_rounding,
            stochastic_rounding_philox_args,
            info_B_num_bits,
            info_B_mask,
            long_run_id_to_really_long_run_ids_acc,
            temp_grad_accum_acc,
            grad_accum_counter_acc,
            max_segment_length_per_cta,
            use_deterministic_algorithms,
            max_vecs_per_thread,
            momentum1_dev_acc,
            momentum1_uvm_acc,
            momentum1_placements_acc,
            momentum1_offsets_acc,
            learning_rate,
            eps,
            weight_decay,
            weight_decay_mode,
            max_norm
        );

    HIP_CHECK(hipDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Kernel execution completed successfully on AMD GPU!" << std::endl;
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    // Copy results back to host for verification
    std::cout << "Copying results back to host..." << std::endl;
    HIP_CHECK(hipMemcpy(dev_weights_host.data(), dev_weights_dev, dev_weights_host.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(momentum1_dev_host.data(), momentum1_dev_dev, momentum1_dev_host.size() * sizeof(float), hipMemcpyDeviceToHost));

    // Print some sample results
    std::cout << "Sample updated weights (first 10 elements):" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(dev_weights_host.size())); ++i) {
        std::cout << "  dev_weights[" << i << "] = " << dev_weights_host[i] << std::endl;
    }

    std::cout << "Sample momentum values (first 10 elements):" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(momentum1_dev_host.size())); ++i) {
        std::cout << "  momentum1_dev[" << i << "] = " << momentum1_dev_host[i] << std::endl;
    }

    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    HIP_CHECK(hipFree(grad_output_dev));
    HIP_CHECK(hipFree(dev_weights_dev));
    HIP_CHECK(hipFree(uvm_weights_dev));
    HIP_CHECK(hipFree(lxu_cache_weights_dev));
    HIP_CHECK(hipFree(weights_placements_dev));
    HIP_CHECK(hipFree(weights_offsets_dev));
    HIP_CHECK(hipFree(D_offsets_dev));
    HIP_CHECK(hipFree(hash_size_cumsum_dev));
    HIP_CHECK(hipFree(sorted_linear_indices_run_dev));
    HIP_CHECK(hipFree(sorted_linear_indices_cumulative_run_lengths_dev));
    HIP_CHECK(hipFree(long_run_ids_dev));
    HIP_CHECK(hipFree(num_long_run_ids_dev));
    HIP_CHECK(hipFree(sorted_infos_dev));
    HIP_CHECK(hipFree(sorted_lxu_cache_locations_dev));
    HIP_CHECK(hipFree(table_unique_indices_offsets_dev));
    HIP_CHECK(hipFree(long_run_id_to_really_long_run_ids_dev));
    HIP_CHECK(hipFree(temp_grad_accum_dev));
    HIP_CHECK(hipFree(grad_accum_counter_dev));
    HIP_CHECK(hipFree(momentum1_dev_dev));
    HIP_CHECK(hipFree(momentum1_uvm_dev));
    HIP_CHECK(hipFree(momentum1_placements_dev));
    HIP_CHECK(hipFree(momentum1_offsets_dev));

    std::cout << "HIP test completed successfully!" << std::endl;
    return 0;
}

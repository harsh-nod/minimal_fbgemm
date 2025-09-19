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
#include <chrono>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>

// Include ATen first
#include <ATen/ATen.h>

// Include the kernels
#include "../src/gen_embedding_backward_rowwise_adagrad_split_unweighted_kernel_cta.hip"
#include "../src/gen_embedding_backward_split_indice_weights_codegen.hip"
#include "../include/fbgemm_gpu/utils/fixed_divisor.cuh"

// Helper function to check HIP errors
#define HIP_CHECK(call)                                                  \
    do                                                                   \
    {                                                                    \
        hipError_t error = call;                                         \
        if (error != hipSuccess)                                         \
        {                                                                \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__  \
                      << " - " << hipGetErrorString(error) << std::endl; \
            exit(1);                                                     \
        }                                                                \
    } while (0)

// Helper function to print sample data
template <typename T>
void print_sample_data(const std::vector<T> &data, const std::string &name, int max_samples = 5)
{
    std::cout << "Sample captured " << name << " (first " << max_samples << " elements):" << std::endl;
    for (int i = 0; i < std::min(max_samples, static_cast<int>(data.size())); ++i)
    {
        std::cout << "  " << name << "[" << i << "] = " << data[i] << std::endl;
    }
}

// Helper function to allocate device memory
template <typename T>
void allocate_device_memory(T **dev_ptr, size_t size)
{
    HIP_CHECK(hipMalloc(dev_ptr, size * sizeof(T)));
}

// Helper function to copy data to device
template <typename T>
void copy_to_device(T *dev_ptr, const std::vector<T> &host_data)
{
    HIP_CHECK(hipMemcpy(dev_ptr, host_data.data(), host_data.size() * sizeof(T), hipMemcpyHostToDevice));
}

// Helper function to copy data from device
template <typename T>
void copy_from_device(std::vector<T> &host_data, T *dev_ptr)
{
    HIP_CHECK(hipMemcpy(host_data.data(), dev_ptr, host_data.size() * sizeof(T), hipMemcpyDeviceToHost));
}

// Helper function to free device memory
template <typename T>
void free_device_memory(T *dev_ptr)
{
    HIP_CHECK(hipFree(dev_ptr));
}

// Helper function to create 1D tensor accessor
template <typename T, int N>
auto create_1d_accessor(T *dev_ptr, int32_t size)
{
    int32_t sizes[N] = {size};
    int32_t strides[N] = {1};
    return at::PackedTensorAccessor32<T, N, at::RestrictPtrTraits>(dev_ptr, sizes, strides);
}

// Helper function to create 2D tensor accessor
template <typename T, int N>
auto create_2d_accessor(T *dev_ptr, int32_t dim0, int32_t dim1)
{
    int32_t sizes[N] = {dim0, dim1};
    int32_t strides[N] = {dim1, 1};
    return at::PackedTensorAccessor32<T, N, at::RestrictPtrTraits>(dev_ptr, sizes, strides);
}

// Helper function to create 1D tensor accessor with int64_t
template <typename T, int N>
auto create_1d_accessor64(T *dev_ptr, int64_t size)
{
    int64_t sizes[N] = {size};
    int64_t strides[N] = {1};
    return at::PackedTensorAccessor64<T, N, at::RestrictPtrTraits>(dev_ptr, sizes, strides);
}

// Helper function to create 2D tensor accessor with int64_t
template <typename T, int N>
auto create_2d_accessor64(T *dev_ptr, int64_t dim0, int64_t dim1)
{
    int64_t sizes[N] = {dim0, dim1};
    int64_t strides[N] = {dim1, 1};
    return at::PackedTensorAccessor64<T, N, at::RestrictPtrTraits>(dev_ptr, sizes, strides);
}

// Structure to hold captured data
struct CapturedData
{
    std::vector<float> grad_output;
    std::vector<float> dev_weights;
    std::vector<float> uvm_weights;
    std::vector<float> lxu_cache_weights;
    std::vector<int32_t> weights_placements;
    std::vector<int64_t> weights_offsets;
    std::vector<int32_t> D_offsets;
    int64_t max_D;
    std::vector<int64_t> indices;
    std::vector<int64_t> offsets;
    std::vector<int32_t> lxu_cache_locations;
    std::vector<int32_t> feature_requires_grad;

    int32_t T;       // number of tables
    int32_t B;       // batch size
    int32_t total_D; // total embedding dimension
};

// Helper function to read all captured data
CapturedData read_captured_data(const std::string &capture_dir)
{
    CapturedData data;

    data.grad_output = read_tensor_from_file<float>(capture_dir + "/grad_output.bin");
    data.dev_weights = read_tensor_from_file<float>(capture_dir + "/dev_weights.bin");
    data.uvm_weights = read_tensor_from_file<float>(capture_dir + "/uvm_weights.bin");
    data.lxu_cache_weights = read_tensor_from_file<float>(capture_dir + "/lxu_cache_weights.bin");
    data.weights_placements = read_tensor_from_file<int32_t>(capture_dir + "/weights_placements.bin");
    data.weights_offsets = read_tensor_from_file<int64_t>(capture_dir + "/weights_offsets.bin");
    data.D_offsets = read_tensor_from_file<int32_t>(capture_dir + "/D_offsets.bin");
    data.max_D = read_scalar_from_file<int64_t>(capture_dir + "/max_D.bin");
    data.indices = read_tensor_from_file<int64_t>(capture_dir + "/indices.bin");
    data.offsets = read_tensor_from_file<int64_t>(capture_dir + "/offsets.bin");
    data.lxu_cache_locations = read_tensor_from_file<int32_t>(capture_dir + "/lxu_cache_locations.bin");
    data.feature_requires_grad = read_tensor_from_file<int32_t>(capture_dir + "/feature_requires_grad.bin");

    // Calculate dimensions
    data.T = data.D_offsets.size() - 1;
    data.B = data.offsets.size() - 1;
    data.total_D = 0;
    for (int i = 0; i < data.T; ++i)
    {
        data.total_D += data.D_offsets[i + 1] - data.D_offsets[i];
    }

    return data;
}

// Helper function to verify captured data
bool verify_captured_data(const CapturedData &data)
{
    if (data.grad_output.empty() || data.dev_weights.empty())
    {
        std::cerr << "Failed to read captured data!" << std::endl;
        return false;
    }

    if (data.grad_output.size() != data.B * data.total_D)
    {
        std::cerr << "Error: grad_output size mismatch. Expected " << data.B * data.total_D
                  << ", got " << data.grad_output.size() << std::endl;
        return false;
    }

    return true;
}

// Helper function to print captured data info
void print_captured_data_info(const CapturedData &data)
{
    std::cout << "Captured data verification:" << std::endl;
    std::cout << "  grad_output size: " << data.grad_output.size() << std::endl;
    std::cout << "  dev_weights size: " << data.dev_weights.size() << std::endl;
    std::cout << "  uvm_weights size: " << data.uvm_weights.size() << std::endl;
    std::cout << "  lxu_cache_weights size: " << data.lxu_cache_weights.size() << std::endl;
    std::cout << "  weights_placements size: " << data.weights_placements.size() << std::endl;
    std::cout << "  weights_offsets size: " << data.weights_offsets.size() << std::endl;
    std::cout << "  D_offsets size: " << data.D_offsets.size() << std::endl;
    std::cout << "  max_D: " << data.max_D << std::endl;
    std::cout << "  indices size: " << data.indices.size() << std::endl;
    std::cout << "  offsets size: " << data.offsets.size() << std::endl;
    std::cout << "  lxu_cache_locations size: " << data.lxu_cache_locations.size() << std::endl;
    std::cout << "  feature_requires_grad size: " << data.feature_requires_grad.size() << std::endl;

    std::cout << "Calculated dimensions from captured data:" << std::endl;
    std::cout << "  B (batch size): " << data.B << std::endl;
    std::cout << "  T (number of tables): " << data.T << std::endl;
    std::cout << "  total_D: " << data.total_D << std::endl;
}

// Helper function to print sample data
void print_sample_captured_data(const CapturedData &data)
{
    print_sample_data(data.grad_output, "grad_output");
    print_sample_data(data.indices, "indices");
    print_sample_data(data.offsets, "offsets");
}

// Structure to hold device pointers
struct DevicePointers
{
    float *grad_output_dev;
    float *dev_weights_dev;
    float *uvm_weights_dev;
    float *lxu_cache_weights_dev;
    int32_t *weights_placements_dev;
    int64_t *weights_offsets_dev;
    int32_t *D_offsets_dev;
    int64_t *indices_dev;
    int64_t *offsets_dev;
    int32_t *lxu_cache_locations_dev;
    int32_t *feature_requires_grad_dev;
    float *grad_indice_weights_dev;
};

// Helper function to allocate all device memory
DevicePointers allocate_device_memory(const CapturedData &data)
{
    DevicePointers ptrs;

    allocate_device_memory(&ptrs.grad_output_dev, data.grad_output.size());
    allocate_device_memory(&ptrs.dev_weights_dev, data.dev_weights.size());
    allocate_device_memory(&ptrs.uvm_weights_dev, data.uvm_weights.size());
    allocate_device_memory(&ptrs.lxu_cache_weights_dev, data.lxu_cache_weights.size());
    allocate_device_memory(&ptrs.weights_placements_dev, data.weights_placements.size());
    allocate_device_memory(&ptrs.weights_offsets_dev, data.weights_offsets.size());
    allocate_device_memory(&ptrs.D_offsets_dev, data.D_offsets.size());
    allocate_device_memory(&ptrs.indices_dev, data.indices.size());
    allocate_device_memory(&ptrs.offsets_dev, data.offsets.size());
    allocate_device_memory(&ptrs.lxu_cache_locations_dev, data.lxu_cache_locations.size());
    allocate_device_memory(&ptrs.feature_requires_grad_dev, data.feature_requires_grad.size());
    allocate_device_memory(&ptrs.grad_indice_weights_dev, data.indices.size());

    return ptrs;
}

// Helper function to copy all data to device
void copy_all_data_to_device(const CapturedData &data, const DevicePointers &ptrs)
{
    copy_to_device(ptrs.grad_output_dev, data.grad_output);
    copy_to_device(ptrs.dev_weights_dev, data.dev_weights);
    copy_to_device(ptrs.uvm_weights_dev, data.uvm_weights);
    copy_to_device(ptrs.lxu_cache_weights_dev, data.lxu_cache_weights);
    copy_to_device(ptrs.weights_placements_dev, data.weights_placements);
    copy_to_device(ptrs.weights_offsets_dev, data.weights_offsets);
    copy_to_device(ptrs.D_offsets_dev, data.D_offsets);
    copy_to_device(ptrs.indices_dev, data.indices);
    copy_to_device(ptrs.offsets_dev, data.offsets);
    copy_to_device(ptrs.lxu_cache_locations_dev, data.lxu_cache_locations);
    copy_to_device(ptrs.feature_requires_grad_dev, data.feature_requires_grad);
}

// Helper function to free all device memory
void free_all_device_memory(const DevicePointers &ptrs)
{
    free_device_memory(ptrs.grad_output_dev);
    free_device_memory(ptrs.dev_weights_dev);
    free_device_memory(ptrs.uvm_weights_dev);
    free_device_memory(ptrs.lxu_cache_weights_dev);
    free_device_memory(ptrs.weights_placements_dev);
    free_device_memory(ptrs.weights_offsets_dev);
    free_device_memory(ptrs.D_offsets_dev);
    free_device_memory(ptrs.indices_dev);
    free_device_memory(ptrs.offsets_dev);
    free_device_memory(ptrs.lxu_cache_locations_dev);
    free_device_memory(ptrs.feature_requires_grad_dev);
    free_device_memory(ptrs.grad_indice_weights_dev);
}

// Helper function to create all tensor accessors
auto create_all_tensor_accessors(const CapturedData &data, const DevicePointers &ptrs)
{
    struct TensorAccessors
    {
        decltype(create_2d_accessor64<float, 2>(nullptr, 0, 0)) grad_output_acc;
        decltype(create_1d_accessor64<float, 1>(nullptr, 0)) dev_weights_acc;
        decltype(create_1d_accessor64<float, 1>(nullptr, 0)) uvm_weights_acc;
        decltype(create_2d_accessor64<float, 2>(nullptr, 0, 0)) lxu_cache_weights_acc;
        decltype(create_1d_accessor<int32_t, 1>(nullptr, 0)) weights_placements_acc;
        decltype(create_1d_accessor<int64_t, 1>(nullptr, 0)) weights_offsets_acc;
        decltype(create_1d_accessor<int32_t, 1>(nullptr, 0)) D_offsets_acc;
        decltype(create_1d_accessor<int64_t, 1>(nullptr, 0)) indices_acc;
        decltype(create_1d_accessor<int64_t, 1>(nullptr, 0)) offsets_acc;
        decltype(create_1d_accessor<int32_t, 1>(nullptr, 0)) lxu_cache_locations_acc;
        decltype(create_1d_accessor<int32_t, 1>(nullptr, 0)) feature_requires_grad_acc;
        decltype(create_1d_accessor<float, 1>(nullptr, 0)) grad_indice_weights_acc;
    };

    TensorAccessors accs;

    accs.grad_output_acc = create_2d_accessor64<float, 2>(ptrs.grad_output_dev, data.B, data.total_D);
    accs.dev_weights_acc = create_1d_accessor64<float, 1>(ptrs.dev_weights_dev, data.dev_weights.size());
    accs.uvm_weights_acc = create_1d_accessor64<float, 1>(ptrs.uvm_weights_dev, data.uvm_weights.size());
    accs.lxu_cache_weights_acc = create_2d_accessor64<float, 2>(ptrs.lxu_cache_weights_dev,
                                                                data.lxu_cache_weights.size() / data.max_D,
                                                                data.max_D);
    accs.weights_placements_acc = create_1d_accessor<int32_t, 1>(ptrs.weights_placements_dev, data.T);
    accs.weights_offsets_acc = create_1d_accessor<int64_t, 1>(ptrs.weights_offsets_dev, data.T);
    accs.D_offsets_acc = create_1d_accessor<int32_t, 1>(ptrs.D_offsets_dev, data.T + 1);
    accs.indices_acc = create_1d_accessor<int64_t, 1>(ptrs.indices_dev, data.indices.size());
    accs.offsets_acc = create_1d_accessor<int64_t, 1>(ptrs.offsets_dev, data.offsets.size());
    accs.lxu_cache_locations_acc = create_1d_accessor<int32_t, 1>(ptrs.lxu_cache_locations_dev, data.lxu_cache_locations.size());
    accs.feature_requires_grad_acc = create_1d_accessor<int32_t, 1>(ptrs.feature_requires_grad_dev, data.T);
    accs.grad_indice_weights_acc = create_1d_accessor<float, 1>(ptrs.grad_indice_weights_dev, data.indices.size());

    return accs;
}

// Helper function to read tensor data from files
template <typename T>
std::vector<T> read_tensor_from_file(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }

    // Read tensor metadata
    c10::ScalarType dtype;
    c10::DeviceType device_type;
    int32_t device_index;

    file.read(reinterpret_cast<char *>(&dtype), sizeof(dtype));
    file.read(reinterpret_cast<char *>(&device_type), sizeof(device_type));
    file.read(reinterpret_cast<char *>(&device_index), sizeof(device_index));

    // Read tensor dimensions
    int64_t num_dims;
    file.read(reinterpret_cast<char *>(&num_dims), sizeof(num_dims));

    std::vector<int64_t> dims(num_dims);
    for (int64_t i = 0; i < num_dims; ++i)
    {
        file.read(reinterpret_cast<char *>(&dims[i]), sizeof(dims[i]));
    }

    // Calculate total elements
    int64_t total_elements = 1;
    for (int64_t dim : dims)
    {
        total_elements *= dim;
    }

    // Read tensor data
    std::vector<T> data(total_elements);
    file.read(reinterpret_cast<char *>(data.data()), total_elements * sizeof(T));

    file.close();

    std::cout << "Read tensor from " << filename << ": dtype=" << dtype
              << ", device=" << device_type << ":" << device_index
              << ", dims=[";
    for (int64_t i = 0; i < num_dims; ++i)
    {
        std::cout << dims[i];
        if (i < num_dims - 1)
            std::cout << ", ";
    }
    std::cout << "], elements=" << total_elements << std::endl;

    return data;
}

// Helper function to read scalar values from files
template <typename T>
T read_scalar_from_file(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return T{};
    }

    T value;
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    file.close();

    std::cout << "Read scalar from " << filename << ": " << value << std::endl;
    return value;
}

// Helper function to find the most recent capture directory
std::string find_latest_capture_directory()
{
    std::string current_dir = ".";
    std::string latest_dir;
    auto latest_time = std::filesystem::file_time_type::min();

    for (const auto &entry : std::filesystem::directory_iterator(current_dir))
    {
        if (entry.is_directory())
        {
            std::string dirname = entry.path().filename().string();
            if (dirname.find("embedding_grad_indice_weights_capture_") == 0)
            {
                auto dir_time = entry.last_write_time();
                if (dir_time > latest_time)
                {
                    latest_time = dir_time;
                    latest_dir = dirname;
                }
            }
        }
    }

    if (latest_dir.empty())
    {
        std::cerr << "No capture directory found!" << std::endl;
        return "";
    }

    std::cout << "Found latest capture directory: " << latest_dir << std::endl;
    return latest_dir;
}

// Helper function to save results to file
void save_results_to_file(const std::string &capture_dir, const std::vector<float> &results)
{
    std::string results_file = capture_dir + "/grad_indice_weights_results.bin";
    std::ofstream results_out(results_file, std::ios::binary);
    if (results_out.is_open())
    {
        results_out.write(reinterpret_cast<const char *>(results.data()), results.size() * sizeof(float));
        results_out.close();
        std::cout << "Results saved to: " << results_file << std::endl;
    }
}

int main()
{
    std::cout << "Testing grad indice weights kernel with captured data on AMD GPU" << std::endl;

    // Use direct HIP API to check GPU availability
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "No HIP devices found!" << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " HIP devices" << std::endl;

    HIP_CHECK(hipSetDevice(1));
    std::cout << "Using device: AMD Instinct MI300X" << std::endl;
    std::cout << "Compute capability: 11.0" << std::endl;

    // Test the grad indice weights kernel
    std::cout << "\n=== Testing Grad Indice Weights Kernel ===" << std::endl;

    // Check if capture directory exists
    std::string capture_dir = find_latest_capture_directory();
    if (capture_dir.empty())
    {
        std::cerr << "No capture directory found! Please run with CAPTURE_ENABLED=1 first to capture embedding inputs." << std::endl;
        return 1;
    }

    std::cout << "Found capture directory: " << capture_dir << std::endl;
    std::cout << "Reading captured data and running grad indice weights kernel..." << std::endl;

    // Read and verify captured data
    CapturedData data = read_captured_data(capture_dir);
    if (!verify_captured_data(data))
    {
        return 1;
    }

    std::cout << "Successfully read captured data. Running grad indice weights kernel with captured inputs..." << std::endl;

    // Print captured data information
    print_captured_data_info(data);
    print_sample_captured_data(data);

    // Allocate device memory
    DevicePointers ptrs = allocate_device_memory(data);

    // Copy data to device
    copy_all_data_to_device(data, ptrs);

    // Create tensor accessors
    auto accs = create_all_tensor_accessors(data, ptrs);

    // Launch kernel
    std::cout << "Launching grad indice weights kernel with captured data..." << std::endl;

    dim3 blockDim(64, 4, 1);
    dim3 gridDim(16384, 1, 1);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Call the grad indice weights kernel with captured data
    split_embedding_codegen_grad_indice_weights_kernel<float, float, float, int64_t, 2, 64, true>
        <<<gridDim, blockDim>>>(
            accs.grad_output_acc,
            accs.dev_weights_acc,
            accs.uvm_weights_acc,
            accs.lxu_cache_weights_acc,
            accs.weights_placements_acc,
            accs.weights_offsets_acc,
            accs.D_offsets_acc,
            accs.indices_acc,
            accs.offsets_acc,
            accs.lxu_cache_locations_acc,
            accs.feature_requires_grad_acc,
            accs.grad_indice_weights_acc,
            fbgemm_gpu::FixedDivisor(data.B));

    HIP_CHECK(hipDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Grad indice weights kernel execution completed successfully!" << std::endl;
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    // Copy results back to host
    std::vector<float> grad_indice_weights_host(data.indices.size());
    copy_from_device(grad_indice_weights_host, ptrs.grad_indice_weights_dev);

    // Print sample results
    print_sample_data(grad_indice_weights_host, "grad_indice_weights", 10);

    // Save results to file
    save_results_to_file(capture_dir, grad_indice_weights_host);

    // Cleanup device memory
    free_all_device_memory(ptrs);

    std::cout << "Kernel execution with captured data completed successfully!" << std::endl;
    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
}

# FBGEMM Kernel Test Project with HIP Support

This is a self-contained project that tests the `split_embedding_backward_codegen_rowwise_adagrad_unweighted_kernel_cta_per_row_1` kernel from FBGEMM, with support for both CUDA (via HIP conversion) and CPU execution.

## Project Structure

```
self_contained_project/
├── CMakeLists.txt                    # Build configuration with HIP support
├── README.md                         # This file
├── build.sh                          # Build script with HIP support
├── include/                          # Header files
│   ├── ATen/
│   │   └── ATen.h                    # Simplified ATen implementation
│   └── fbgemm_gpu/
│       ├── embedding_backward_template_helpers.cuh
│       ├── split_embeddings_utils.cuh
│       └── utils/
│           ├── tensor_accessor_builder.h
│           └── tensor_accessor.h
├── src/                             # Source files
│   ├── gen_embedding_backward_rowwise_adagrad_split_unweighted_kernel_cta.cu
│   ├── gen_embedding_backward_split_unweighted_device_kernel_hip.cuh
│   ├── gen_embedding_optimizer_rowwise_adagrad_split_device_kernel_hip.cuh
│   ├── gen_embedding_backward_split_unweighted_device_kernel.cuh
│   ├── gen_embedding_backward_split_common_device_kernel.cuh
│   └── gen_embedding_optimizer_rowwise_adagrad_split_device_kernel.cuh
└── tests/
    ├── test_kernel.cu                # CUDA test file
    ├── test_kernel_hip.cu            # HIP-optimized test file
    └── test_kernel_cpu.cpp           # CPU-only test file
```

## Building the Project

### Prerequisites

#### For AMD GPU (HIP) Support:
- ROCm (ROCm 5.0 or later recommended)
- HIP compiler (`hipcc`)
- hipify-perl tool
- CMake (version 3.18 or later)
- C++17 compatible compiler

#### For CPU-only Support:
- CMake (version 3.18 or later)
- C++17 compatible compiler
- PyTorch (optional, for full functionality)

### Build Instructions

1. **For AMD GPU (HIP) Support:**
   ```bash
   # Install ROCm (Ubuntu/Debian example)
   wget https://repo.radeon.com/amdgpu-install/5.7/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
   sudo dpkg -i amdgpu-install_5.7.50700-1_all.deb
   sudo amdgpu-install --usecase=rocm

   # Build the project with default architecture (gfx942)
   ./build.sh

   # Build with specific architecture
   ./build.sh gfx942    # For MI300 series
   ./build.sh gfx100    # For MI200 series
   ./build.sh gfx90a    # For MI100 series
   ```

2. **For CPU-only Support:**
   ```bash
   # Build without HIP
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ```

3. **Run the test:**
   ```bash
   # For AMD GPU
   ./test_kernel

   # For CPU-only
   ./test_kernel_cpu
   ```

## What This Project Tests

This project contains a comprehensive test of the FBGEMM embedding backward kernel with the following components:

### Main Kernel
- **`split_embedding_backward_codegen_rowwise_adagrad_unweighted_kernel_cta_per_row_1`**
  - Main CUDA/HIP kernel that performs embedding backward computation
  - Uses row-wise Adagrad optimization
  - Handles unweighted embeddings
  - Automatically converted from CUDA to HIP using hipify-perl

### Device Kernels
- **`compute_grad_sum_unweighted`**: Computes gradient sums for unweighted embeddings
- **`store_grad_sum`**: Stores computed gradients
- **`split_rowwise_adagrad_table_update_kernel`**: Updates weights using row-wise Adagrad

### Test Features
- **Random Input Generation**: Creates realistic test data with proper distributions
- **Memory Management**: Proper allocation and deallocation of GPU memory
- **Performance Timing**: Measures kernel execution time
- **Result Verification**: Copies results back to host and displays sample outputs
- **Error Handling**: Comprehensive error checking for all GPU operations

## HIP Conversion Process

The project automatically converts CUDA code to HIP using the following process:

1. **hipify-perl**: Converts `.cu` files to `.hip` files
2. **Architecture Targeting**: Compiles for AMD GPU architecture `gfx942`
3. **API Translation**: Converts CUDA runtime API calls to HIP equivalents
4. **Memory Management**: Uses HIP memory allocation functions

## Target Architecture

The project supports multiple AMD GPU architectures. You can specify the target architecture when building:

- **gfx942** (default): MI300 series (AMD Instinct MI300X, etc.)
- **gfx908**: MI200 series (AMD Instinct MI200, etc.)
- **gfx90a**: MI100 series (AMD Instinct MI100, etc.)

### Usage Examples:
```bash
# Build for MI300 series (default)
./build.sh

# Build for MI200 series
./build.sh gfx908

# Build for MI100 series
./build.sh gfx90a
```

## Notes

- This is a simplified version for testing purposes
- Some complex optimizations and features from the full FBGEMM implementation may be missing
- The kernel is designed to compile and run basic functionality tests
- For production use, refer to the full FBGEMM library
- The HIP conversion maintains compatibility with the original CUDA kernel interface

## Troubleshooting

### HIP/ROCm Issues:
1. Ensure ROCm is properly installed and in your PATH
2. Check that `hipcc` and `hipify-perl` are available
3. Verify your AMD GPU supports the target architecture (gfx942)
4. Make sure you have sufficient GPU memory

### Compilation Issues:
1. Ensure all required header files are present
2. Check that CMake can find HIP
3. Verify C++17 support in your compiler
4. Make sure all dependencies are properly installed

### Runtime Issues:
1. Check GPU memory availability
2. Verify HIP device initialization
3. Ensure proper error handling in kernel launches
4. Check that tensor dimensions are valid

## Performance Expectations

- **AMD GPU (gfx942)**: Optimized performance for MI300 series
- **CPU Fallback**: Reasonable performance for testing and development
- **Memory Bandwidth**: Utilizes GPU memory bandwidth efficiently
- **Kernel Launch Overhead**: Minimal overhead for small test cases


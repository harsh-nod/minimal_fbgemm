cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fbgemm_kernel_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable HIP language
enable_language(HIP)
message(STATUS "HIP enabled")
set(HAVE_HIP TRUE)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${TORCH_INCLUDE_DIRS})

# Find PyTorch CMake path from Python package
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_CMAKE_RESULT
)

if(TORCH_CMAKE_RESULT EQUAL 0)
    message(STATUS "Found PyTorch CMake path: ${TORCH_CMAKE_PREFIX_PATH}")
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
else()
    message(FATAL_ERROR "Failed to find PyTorch CMake path from Python package")
endif()

# Find PyTorch package
find_package(Torch REQUIRED)

# Allow user to specify GPU architecture, default to gfx942
if(NOT DEFINED GPU_ARCH)
    set(GPU_ARCH "gfx942")
    message(STATUS "No GPU architecture specified, defaulting to: ${GPU_ARCH}")
else()
    message(STATUS "Using specified GPU architecture: ${GPU_ARCH}")
endif()

# Set HIP properties for the specified GPU architecture
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY HIP_ARCHITECTURES ${GPU_ARCH})

# PyTorch CUDA Extensions are normally compiled with the flags below. However we
# disabled -D__CUDA_NO_HALF_CONVERSIONS__ here as it caused "error: no suitable
# constructor exists to convert from "int" to "__half" errors in
# gen_embedding_forward_quantized_split_[un]weighted_codegen_cuda.cu
#
set(TORCH_CUDA_OPTIONS
  --expt-relaxed-constexpr
  -D__CUDA_NO_HALF_OPERATORS__
  # -D__CUDA_NO_HALF_CONVERSIONS__
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__
  -D__CUDA_NO_HALF2_OPERATORS__)

# Set HIP compiler flags (adapted from CUDA options for HIP)
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -std=c++17 -O3 --offload-arch=${GPU_ARCH} -DFBGEMM_USE_SUBWARP_SHUFFLE=1 -D__HIP_NO_HALF_OPERATORS__ -D__HIP_NO_HALF_CONVERSIONS__ -D__HIP_NO_HALF2_OPERATORS__ -D__HIP_PLATFORM_AMD__ -DHIP_ARCH_${GPU_ARCH}")

# Create HIP versions of CUDA files using hipify
set(HIPIFY_SOURCES)

# HIP source files (already in HIP format)
set(HIP_SOURCES
    src/gen_embedding_backward_rowwise_adagrad_split_unweighted_kernel_cta.hip
)

# Add kernel library with HIP sources
hip_add_library(fbgemm_kernels STATIC
    ${HIP_SOURCES}
    src/gen_embedding_backward_split_unweighted_device_kernel_hip.cuh
    src/gen_embedding_optimizer_rowwise_adagrad_split_device_kernel_hip.cuh
    src/gen_embedding_backward_split_unweighted_device_kernel.cuh
    src/gen_embedding_backward_split_common_device_kernel.cuh
    src/gen_embedding_optimizer_rowwise_adagrad_split_device_kernel.cuh
)

# Link with PyTorch
target_link_libraries(fbgemm_kernels ${TORCH_LIBRARIES})

# Test HIP source file (already in HIP format)
set(TEST_HIP_SOURCE tests/test_kernel_hip.hip)

# Add test executable
hip_add_executable(test_kernel
    ${TEST_HIP_SOURCE}
)

target_link_libraries(test_kernel
    fbgemm_kernels
    ${TORCH_LIBRARIES}
    ${HIP_LIBRARIES}
)

# Set target properties
set_target_properties(fbgemm_kernels PROPERTIES
    HIP_SEPARABLE_COMPILATION ON
)

set_target_properties(test_kernel PROPERTIES
    HIP_SEPARABLE_COMPILATION ON
)

# Print configuration info
message(STATUS "HIP version: ${HIP_VERSION}")
message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
message(STATUS "HIP flags: ${CMAKE_HIP_FLAGS}")
message(STATUS "Target architecture: ${GPU_ARCH}")

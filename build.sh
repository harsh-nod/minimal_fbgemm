#!/bin/bash

# Build script for FBGEMM Kernel Test Project with HIP support
# Usage: ./build.sh [GPU_ARCH]
# Example: ./build.sh gfx942
# Example: ./build.sh gfx908

set -e

# Get GPU architecture from command line argument, default to gfx942
GPU_ARCH=${1:-gfx942}

echo "Building FBGEMM Kernel Test Project with HIP support..."
echo "Target GPU architecture: ${GPU_ARCH}"

# Check if hipify-perl is available
if ! command -v hipify-perl &> /dev/null; then
    echo "Warning: hipify-perl not found. Make sure ROCm is installed."
    echo "You can install ROCm from: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
fi

# Check if HIP is available
if ! command -v hipcc &> /dev/null; then
    echo "Warning: hipcc not found. Make sure ROCm is installed."
    echo "You can install ROCm from: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_ARCH=${GPU_ARCH}

# Build the project
echo "Building..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Run './test_kernel' to test the kernel on AMD GPU (${GPU_ARCH})"

#!/bin/bash
# Build llama.cpp with CUDA support
# Run from the QuatNet root directory
set -e

cd llama.cpp

# Auto-detect CUDA
if command -v nvcc &>/dev/null; then
    echo "CUDA detected, building with GPU support"
    CUDA_FLAG="-DGGML_CUDA=ON"

    # Detect nvcc version and set safe architecture targets
    # Older CUDA toolkits don't support newest GPU architectures
    NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    NVCC_MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
    NVCC_MINOR=$(echo "$NVCC_VER" | cut -d. -f2)

    if [ "$NVCC_MAJOR" -lt 12 ] || { [ "$NVCC_MAJOR" -eq 12 ] && [ "$NVCC_MINOR" -lt 8 ]; }; then
        # CUDA < 12.8: no Blackwell (sm_120) support
        # Use native to compile for the GPU in this machine, with common fallbacks
        ARCH_FLAG="-DCMAKE_CUDA_ARCHITECTURES=native"
        echo "CUDA $NVCC_VER: using native GPU architecture"
    else
        ARCH_FLAG=""
        echo "CUDA $NVCC_VER: using default architecture detection"
    fi
else
    echo "No CUDA found, building CPU-only"
    CUDA_FLAG=""
    ARCH_FLAG=""
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release $CUDA_FLAG $ARCH_FLAG
cmake --build build --config Release -j$(nproc)

echo "Build complete. Binaries in llama.cpp/build/bin/"

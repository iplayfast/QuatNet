#!/bin/bash
# Build llama.cpp with CUDA support
# Run from the QuatNet root directory
set -e

cd llama.cpp

# Auto-detect CUDA
if command -v nvcc &>/dev/null; then
    echo "CUDA detected, building with GPU support"
    CUDA_FLAG="-DGGML_CUDA=ON"
else
    echo "No CUDA found, building CPU-only"
    CUDA_FLAG=""
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release $CUDA_FLAG
cmake --build build --config Release -j$(nproc)

echo "Build complete. Binaries in llama.cpp/build/bin/"

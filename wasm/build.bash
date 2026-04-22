#!/bin/bash
set -euo pipefail

# Resolve repo root and build dir from the script's own location so the
# script behaves the same regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Create a temporary directory and preserve .gitignore
mkdir -p ../temp
mv .gitignore ../temp/

# Clean the build directory
rm -rf *

# Restore .gitignore and clean up temp directory
mv ../temp/.gitignore .
rm -rf ../temp
clear

# Configure and build the project using Emscripten
# -DCMAKE_BUILD_TYPE="Release" sets the build type to Release mode
# -j4 enables parallel compilation using 4 cores
# -msimd128: enable WebAssembly SIMD (128-bit vector ops) for faster matrix math
# -DNAM_USE_INLINE_GEMM: use inline GEMM optimizations from NeuralAmpModelerCore for better performance
emcmake cmake .. -DCMAKE_BUILD_TYPE="Release" -DCMAKE_CXX_FLAGS="${CXX_FLAGS:-} -msimd128 -DNAM_USE_INLINE_GEMM" && cmake --build . --config=release -j4

# Format the generated JavaScript file for to prep for patching
cd wasm
npx prettier --write t3k-wasm-module.js

# Apply custom patch to the generated JavaScript file
patch -p0 < ../../wasm/t3k-wasm-module.patch

# Minify the JavaScript file to reduce its size
npx terser t3k-wasm-module.js -o t3k-wasm-module.js
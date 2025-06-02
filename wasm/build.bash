#!/bin/bash

# Change to the build directory
cd ../build

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
emcmake cmake .. -DCMAKE_BUILD_TYPE="Release" && cmake --build . --config=release -j4

# Format the generated JavaScript file for to prep for patching
cd wasm
npx prettier --write t3k-wasm-model.js

# Apply custom patch to the generated JavaScript file
patch -p0 < ../../wasm/t3k-wasm-model.patch

# Minify the JavaScript file to reduce its size
npx terser t3k-wasm-model.js -o t3k-wasm-model.js
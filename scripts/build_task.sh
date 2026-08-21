#!/usr/bin/env bash

# Copyright (c) 2025 HomomorphicEncryption.org
# All rights reserved.
#
# This software is licensed under the terms of the Apache v2 License.
# See the LICENSE.md file for details.
#
# Modified 2026 by CryptoLab, Inc.: build steps for the HEaaN2 MNIST submission.

# Usage: ./scripts/build_task.sh <TASK_DIR>(/submissions/mnist)
set -euo pipefail

ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
TASK_DIR="$1"
BUILD="$TASK_DIR/build"
NPROC=$(nproc)

if [[ "$(basename "$TASK_DIR")" == "mnist" ]]; then
    HEAAN2_INSTALL="$TASK_DIR/install"

    if [[ ! -f "$HEAAN2_INSTALL/lib/cmake/HEaaN2/HEaaN2Config.cmake" ]]; then
        echo "[build_task] ERROR: no vendored HEaaN2 install at $HEAAN2_INSTALL." >&2
        echo "             See submissions/mnist/README.md." >&2
        exit 1
    fi

    # find_dependency(CUDAToolkit) searches PATH only, so an activated conda
    # toolkit has to outrank a system one exported from a login profile.
    NVCC=""
    for _cand in "${CUDACXX:-}" \
                 "${CONDA_PREFIX:+$CONDA_PREFIX/bin/nvcc}" \
                 "$(command -v nvcc 2>/dev/null || true)"; do
        if [[ -n "$_cand" && -x "$_cand" ]]; then
            NVCC="$_cand"
            break
        fi
    done
    if [[ -z "$NVCC" ]]; then
        echo "[build_task] ERROR: nvcc not found." >&2
        echo "             Put the CUDA toolkit on PATH or set CUDACXX." >&2
        exit 1
    fi

    # Warn, not abort: the build itself is fine on a GPU-less submit node.
    if command -v nvidia-smi >/dev/null 2>&1; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
        if [[ -n "$COMPUTE_CAP" && "$COMPUTE_CAP" != "12.0" ]]; then
            echo "[build_task] WARNING: GPU compute capability is $COMPUTE_CAP, expected 12.0." >&2
            echo "             The vendored HEaaN2 is sm_120 only; stage 7 will fail here." >&2
        fi
    fi

    # CMake resolves a relative CMAKE_PREFIX_PATH against the build directory.
    HEAAN2_INSTALL="$(cd -- "$HEAAN2_INSTALL" && pwd)"

    echo "[build_task] Using vendored HEaaN2 install at $HEAAN2_INSTALL"
    echo "[build_task] CUDA compiler: $NVCC"
    echo "[build_task] Configuring the HEaaN2 MNIST submission..."
    cmake -S "$TASK_DIR" -B "$BUILD" \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_WITH_CUDA=ON \
          -DCUDAToolkit_ROOT="$(dirname "$(dirname "$NVCC")")" \
          -DCMAKE_PREFIX_PATH="$HEAAN2_INSTALL"

    echo "[build_task] Compiling with $NPROC cores..."
    cmake --build "$BUILD" -j"$NPROC"

    # harness/utils.py silently skips a stage whose binary is missing.
    for stage in client_key_generation client_preprocess_input \
                 client_encode_encrypt_input server_preprocess_model \
                 server_encrypted_compute client_decrypt_decode \
                 client_postprocess; do
        if [[ ! -x "$BUILD/$stage" ]]; then
            echo "[build_task] ERROR: stage binary $BUILD/$stage was not built." >&2
            exit 1
        fi
    done

    echo "[build_task] Build complete."
    exit 0
fi

# Original OpenFHE + libtorch path (submissions/cifar10, ...)

# --- 1. LibTorch (PyTorch C++ distribution) ---
LIBTORCH_DIR="$ROOT/third_party/libtorch"
LIBTORCH_ZIP_NAME="libtorch_temp.zip"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip"

if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Downloading LibTorch..."
    mkdir -p "$ROOT/third_party"
    cd "$ROOT/third_party"

    # Use -O to force the output filename and avoid ".zip.1" duplicates
    # We also remove any existing partial downloads first to be safe
    rm -f "$LIBTORCH_ZIP_NAME"
    wget -O "$LIBTORCH_ZIP_NAME" "$LIBTORCH_URL"

    echo "Unzipping LibTorch..."
    unzip -q "$LIBTORCH_ZIP_NAME"
    rm "$LIBTORCH_ZIP_NAME"

    cd "$ROOT"
    echo "LibTorch successfully set up at $LIBTORCH_DIR"
fi

# --- 2. nlohmann/json ---
NLOHMANN_DIR="$ROOT/third_party/nlohmann"
NLOHMANN_HEADER="$NLOHMANN_DIR/json.hpp"
NLOHMANN_URL="https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp"

if [[ ! -f "$NLOHMANN_HEADER" ]]; then
      echo "Downloading nlohmann/json..."
      mkdir -p "$NLOHMANN_DIR"
      curl -L -o "$NLOHMANN_HEADER" "$NLOHMANN_URL"
fi

# --- 3. Build Process ---
# We assume OpenFHE is in /third_party/openfhe or provided via CMAKE_PREFIX_PATH.
echo "Configuring project with CMake..."
cmake -S "$TASK_DIR" -B "$BUILD" \
      -DCMAKE_PREFIX_PATH="$ROOT/third_party/openfhe;$ROOT/third_party/libtorch"

echo "Compiling with $NPROC cores..."
cd "$BUILD"
make -j"$NPROC"

echo "Build complete."

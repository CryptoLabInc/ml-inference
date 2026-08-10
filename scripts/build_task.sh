#!/usr/bin/env bash

# Copyright (c) 2025 HomomorphicEncryption.org
# All rights reserved.
#
# This software is licensed under the terms of the Apache v2 License.
# See the LICENSE.md file for details.

# ------------------------------------------------------------
# Usage: ./scripts/build_task.sh <TASK_DIR>
# Compiles the files in the source directory.
#
# Dispatches on the task directory:
#   submissions/mnist -> the HEaaN2 submission (no OpenFHE, no libtorch)
#   anything else     -> the original OpenFHE + libtorch path, unchanged
# ------------------------------------------------------------
set -euo pipefail

# Define core paths
ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
TASK_DIR="$1"
BUILD="$TASK_DIR/build"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu || echo 4)

# ============================================================
# HEaaN2 path (submissions/mnist)
# ============================================================
if [[ "$(basename "$TASK_DIR")" == "mnist" ]]; then
    # HEaaN2 is not redistributable, so it is not fetched here: point
    # HEAAN2_ROOT at a checkout, or HEAAN2_DIR at an existing install tree.
    HEAAN2_ROOT="${HEAAN2_ROOT:-$HOME/HEaaN2}"
    HEAAN2_INSTALL="${HEAAN2_DIR:-$HEAAN2_ROOT/install}"
    HEAAN2_BUILD_CUDA="${HEAAN2_BUILD_CUDA:-ON}"

    if [[ ! -f "$HEAAN2_INSTALL/lib/cmake/HEaaN2/HEaaN2Config.cmake" ]]; then
        if [[ -f "$HEAAN2_ROOT/CMakeLists.txt" ]]; then
            echo "[build_task] Installing HEaaN2 from $HEAAN2_ROOT -> $HEAAN2_INSTALL"
            cmake -S "$HEAAN2_ROOT" -B "$HEAAN2_ROOT/build" \
                  -DCMAKE_BUILD_TYPE=Release \
                  -DBUILD_WITH_CUDA="$HEAAN2_BUILD_CUDA" \
                  -DCMAKE_INSTALL_PREFIX="$HEAAN2_INSTALL"
            cmake --build "$HEAAN2_ROOT/build" --target install -j"$NPROC"
        else
            echo "[build_task] ERROR: no HEaaN2 install at $HEAAN2_INSTALL and" >&2
            echo "             no source tree at $HEAAN2_ROOT." >&2
            echo "             Set HEAAN2_ROOT to a HEaaN2 checkout, or HEAAN2_DIR" >&2
            echo "             to an existing install prefix. See" >&2
            echo "             submissions/mnist/README.md." >&2
            exit 1
        fi
    else
        echo "[build_task] Using HEaaN2 install at $HEAAN2_INSTALL"
    fi

    echo "[build_task] Configuring the HEaaN2 MNIST submission..."
    cmake -S "$TASK_DIR" -B "$BUILD" \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_WITH_CUDA="$HEAAN2_BUILD_CUDA" \
          -DCMAKE_PREFIX_PATH="$HEAAN2_INSTALL"

    echo "[build_task] Compiling with $NPROC cores..."
    cmake --build "$BUILD" -j"$NPROC"

    # harness/utils.py silently SKIPS a stage whose binary is missing, which
    # would produce a green run over stale output. Fail loudly here instead.
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

# ============================================================
# Original OpenFHE + libtorch path (submissions/cifar10, ...)
# ============================================================

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

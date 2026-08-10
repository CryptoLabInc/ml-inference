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

    # nvcc is frequently not on PATH when this script runs (CMake picks it up
    # from CUDACXX or a previous cache), yet both configures below need it: the
    # HEaaN2 build compiles .cu, and the submission calls find_package(CUDAToolkit).
    # Look in the same places CMake would, once, and reuse the answer for both.
    if [[ "$HEAAN2_BUILD_CUDA" == "ON" && -z "${HEAAN2_NVCC:-}" ]]; then
        for _cand in "${CUDACXX:-}" \
                     "$(command -v nvcc 2>/dev/null || true)" \
                     "${CONDA_PREFIX:-}/bin/nvcc" \
                     "$(sed -n 's/^CMAKE_CUDA_COMPILER:[A-Z]*=//p' \
                          "$HEAAN2_ROOT/build/CMakeCache.txt" 2>/dev/null | head -1)"; do
            if [[ -n "$_cand" && -x "$_cand" ]]; then
                HEAAN2_NVCC="$_cand"
                break
            fi
        done
    fi

    # nvcc probes bare "gcc" for a version, then prepends its own bin/ to PATH
    # before running it. If this script runs without the HEaaN2 conda toolchain
    # env active, those are two different gccs: it probes the system one but
    # preprocesses with conda's, whose libstdc++ headers use builtins the probed
    # version does not advertise, so even CUDA compiler *detection* fails. Pin
    # the host compiler to the toolchain that ships alongside nvcc so they agree.
    if [[ -z "${HEAAN2_CUDA_HOST_COMPILER:-}" && -n "${HEAAN2_NVCC:-}" ]] \
       && [[ -x "$(dirname "$HEAAN2_NVCC")/g++" ]]; then
        HEAAN2_CUDA_HOST_COMPILER="$(dirname "$HEAAN2_NVCC")/g++"
    fi

    # HEaaN2 pulls private CryptoLabInc deps (HEaven, hem) that CPM asks for
    # over https, which cannot prompt for a password in a non-interactive build.
    # If an SSH key can reach GitHub, rewrite those URLs for the duration of
    # this build only -- via GIT_CONFIG_*, so the user's git config is untouched.
    # Set HEAAN2_GIT_SSH=0 to skip (e.g. if you use a credential helper instead).
    # ("|| true" because a successful "ssh -T git@github.com" still exits 1, which
    # pipefail would otherwise read as no SSH access.)
    if [[ "${HEAAN2_GIT_SSH:-1}" == "1" ]] \
       && { ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 \
                -T git@github.com 2>&1 || true; } | grep -q 'successfully authenticated'; then
        echo "[build_task] Routing github.com fetches over SSH for this build."
        export GIT_CONFIG_COUNT=1
        export GIT_CONFIG_KEY_0='url.git@github.com:.insteadOf'
        export GIT_CONFIG_VALUE_0='https://github.com/'
    fi

    if [[ ! -f "$HEAAN2_INSTALL/lib/cmake/HEaaN2/HEaaN2Config.cmake" ]]; then
        if [[ -f "$HEAAN2_ROOT/CMakeLists.txt" ]]; then
            echo "[build_task] Installing HEaaN2 from $HEAAN2_ROOT -> $HEAAN2_INSTALL"
            [[ -n "${HEAAN2_CUDA_HOST_COMPILER:-}" ]] \
                && echo "[build_task] CUDA host compiler: $HEAAN2_CUDA_HOST_COMPILER"
            # Note: setting CMAKE_INSTALL_RPATH here would NOT stick -- HEaaN2's
            # own install rules run file(RPATH_REMOVE) on libheaan2.so, so it ends
            # up with no RPATH regardless. The submission compensates by linking
            # its executables with DT_RPATH; see submissions/mnist/CMakeLists.txt.
            cmake -S "$HEAAN2_ROOT" -B "$HEAAN2_ROOT/build" \
                  -DCMAKE_BUILD_TYPE=Release \
                  -DBUILD_WITH_CUDA="$HEAAN2_BUILD_CUDA" \
                  ${HEAAN2_CUDA_HOST_COMPILER:+-DCMAKE_CUDA_HOST_COMPILER="$HEAAN2_CUDA_HOST_COMPILER"} \
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
    # HEaaN2Config.cmake does find_dependency(CUDAToolkit), which only searches
    # PATH for nvcc -- point it at the toolkit we already located instead.
    cmake -S "$TASK_DIR" -B "$BUILD" \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_WITH_CUDA="$HEAAN2_BUILD_CUDA" \
          ${HEAAN2_NVCC:+-DCUDAToolkit_ROOT="$(dirname "$(dirname "$HEAAN2_NVCC")")"} \
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

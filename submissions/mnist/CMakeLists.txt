# Copyright (c) 2026 Crypto Lab Inc.
#
# This software is licensed under the terms of the Apache v2 License.
# See the LICENSE.md file for details.
# ---------------------------------------------------------------------------
# HEaaN2 MNIST MLP submission.
#
# Built against the HEaaN2 *public* API only. A prebuilt HEaaN2 install (public
# headers + libheaan2.so.0.2.0) is vendored under install/ -- see LICENSE and
# LICENSE-THIRD-PARTY in this directory -- so a bare clone of this fork builds
# with no private-repo access or SSH key needed. Point CMAKE_PREFIX_PATH
# elsewhere (scripts/build_task.sh does this from $HEAAN2_DIR / $HEAAN2_ROOT)
# to use a different install.

cmake_minimum_required(VERSION 3.23)
project(
  heaan2_mnist_mlp
  VERSION 0.1.0
  DESCRIPTION "MNIST MLP FHE inference benchmark submission built on HEaaN2"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/install")
endif()

option(BUILD_WITH_CUDA "Build the submission against the CUDA HEaaN2 backend" ON)

find_package(HEaaN2 REQUIRED)
find_package(OpenMP REQUIRED)

# --- shared pipeline -------------------------------------------------------
add_library(mlp_pipeline STATIC src/mlp_pipeline.cpp src/mlp_pcmm.cpp)
target_include_directories(mlp_pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mlp_pipeline PUBLIC HEaaN2::HEaaN2 OpenMP::OpenMP_CXX)
set_target_properties(mlp_pipeline PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # libheaan2.so has DT_NEEDED on the *shared* libcudart.so.12/libcublas.so.12
  # and does not re-export them (its interface carries only OpenMP), so the
  # static cudart cannot resolve its symbols -- link the shared pair here.
  target_link_libraries(mlp_pipeline PUBLIC CUDA::cudart CUDA::cublas)
  # Selects Device::GPU_CUDA in mlp::targetDevice().
  target_compile_definitions(mlp_pipeline PUBLIC MLP_WITH_CUDA)
endif()

# --- the seven stage executables -------------------------------------------
# These names are the benchmark contract: harness/utils.py resolves each stage
# to submissions/mnist/build/<name> and SILENTLY SKIPS a stage whose binary is
# missing. Do not rename any of them.
set(MLP_STAGES
    client_key_generation
    client_preprocess_input
    client_encode_encrypt_input
    server_preprocess_model
    server_encrypted_compute
    client_decrypt_decode
    client_postprocess)

foreach(stage ${MLP_STAGES})
  add_executable(${stage} src/${stage}.cpp)
  target_link_libraries(${stage} PRIVATE mlp_pipeline)
  # The harness runs the binaries straight out of build/, with no
  # LD_LIBRARY_PATH set, so the HEaaN2 location has to be baked in.
  set_target_properties(${stage} PROPERTIES
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  # ...and baked in as DT_RPATH rather than the modern default DT_RUNPATH.
  # libheaan2.so needs libcudart/libcublas/libtcmalloc but carries no RPATH of
  # its own, and DT_RUNPATH is not consulted when resolving a *dependency's*
  # dependencies, so the loader would fail on those three. DT_RPATH is
  # inherited down the whole chain, which covers them.
  target_link_options(${stage} PRIVATE "LINKER:--disable-new-dtags")
endforeach()

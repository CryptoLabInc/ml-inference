# Copyright (c) 2026 Crypto Lab Inc.
#
# This software is licensed under the terms of the Apache v2 License.
# See the LICENSE.md file for details.
# ---------------------------------------------------------------------------

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

add_library(mlp_pipeline STATIC src/mlp_pipeline.cpp src/mlp_pcmm.cpp)
target_include_directories(mlp_pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mlp_pipeline PUBLIC HEaaN2::HEaaN2 OpenMP::OpenMP_CXX)
set_target_properties(mlp_pipeline PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(mlp_pipeline PUBLIC CUDA::cudart CUDA::cublas)
  target_compile_definitions(mlp_pipeline PUBLIC MLP_WITH_CUDA)
endif()

# Stage names are the benchmark contract: harness/utils.py resolves each stage
# to submissions/mnist/build/<name> and silently skips a missing binary.
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
  set_target_properties(${stage} PROPERTIES
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_link_options(${stage} PRIVATE "LINKER:--disable-new-dtags")
endforeach()

// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// Stage 3: server-side model preprocessing.
//
// The harness invokes this stage with NO arguments (run_submission.py:97), so
// it cannot know the instance size and therefore cannot reach the instance's
// io/<size>/public_keys directory. Everything key-dependent -- in particular
// MatrixVectorEval, which encodes its diagonals in its constructor on the
// device of the rotation keys -- must therefore happen in stage 7.
//
// What is left here is the genuinely size- and key-independent half: read the
// CSV weights, validate their shapes, and write them out in the padded p x q
// layout the packing wants. See README.md for how the two halves are reported.

#include "mlp_pipeline.hpp"

#include <iostream>

using namespace mlp;

namespace {
constexpr const char *WEIGHT_DIR = "submissions/mnist/weights";
constexpr const char *CACHE_DIR = "submissions/mnist/build/model_cache";
} // namespace

int main() try {
    const std::string wdir = WEIGHT_DIR;

    auto W1 = readMatrixCsv(wdir + "/W1_128x484.csv");
    auto b1 = readFlatCsv(wdir + "/b1_128.csv");
    auto W2 = readMatrixCsv(wdir + "/W2_10x128.csv");
    auto b2 = readFlatCsv(wdir + "/b2_10.csv");

    if (W1.size() != HIDDEN_DIM || W1[0].size() != INPUT_DIM)
        throw std::runtime_error("W1 must be 128x484");
    if (W2.size() != LABEL_DIM || W2[0].size() != HIDDEN_DIM)
        throw std::runtime_error("W2 must be 10x128");

    const auto fc1 = padWeights(FC1, W1, b1);
    const auto fc2 = padWeights(FC2, W2, b2);

    fs::create_directories(CACHE_DIR);
    writeLayerWeights(fc1, std::string(CACHE_DIR) + "/fc1.bin");
    writeLayerWeights(fc2, std::string(CACHE_DIR) + "/fc2.bin");

    std::cout << "         [server] model cached to " << CACHE_DIR << "\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << "server_preprocess_model: " << e.what() << "\n";
    return 1;
}

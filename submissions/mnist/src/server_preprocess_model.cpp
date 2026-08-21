// Copyright (c) 2026 CryptoLab, Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#include "mlp_pcmm.hpp"
#include "mlp_pipeline.hpp"

#include <iostream>

using namespace heaan;
using namespace mlp;

namespace {
constexpr const char *WEIGHT_DIR = "submissions/mnist/weights";
constexpr const char *CACHE_DIR = MODEL_CACHE_DIR;
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

    fs::create_directories(CACHE_DIR);

    pcmm::writeRawModel({W1, b1, W2, b2}, std::string(CACHE_DIR) + "/pcmm.bin");

    std::cout << "         [server] model cached to " << CACHE_DIR << "\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << "server_preprocess_model: " << e.what() << "\n";
    return 1;
}

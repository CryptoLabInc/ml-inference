// Copyright (c) 2026 Crypto Lab Inc.
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

    const auto marked = readInstanceMarker();
    if (marked && usePcmm(*marked)) {
        std::cout << "         [server] model cached to " << CACHE_DIR
                  << " (PCMM instance: HS diagonals not needed)\n";
        return 0;
    }

    //HS specific
    const auto fc1 = padWeights(FC1, W1, b1);
    const auto fc2 = padWeights(FC2, W2, b2);

    const Levels levels = buildLevels();
    const u32 top = levels.top();
    const Device dev = targetDevice();

    const auto enc1 = encodeDiags(FC1, fc1.W, levels, top - FC1_IN_DROP, dev);
    serial::save(std::string(CACHE_DIR) + "/fc1_diags.bin", enc1);
    const auto enc2 = encodeDiags(FC2, fc2.W, levels, top - FC2_IN_DROP, dev);
    serial::save(std::string(CACHE_DIR) + "/fc2_diags.bin", enc2);

    std::cout << "         [server] model cached to " << CACHE_DIR << " ("
              << enc1.numDiags() << " + " << enc2.numDiags()
              << " diagonals encoded)\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << "server_preprocess_model: " << e.what() << "\n";
    return 1;
}

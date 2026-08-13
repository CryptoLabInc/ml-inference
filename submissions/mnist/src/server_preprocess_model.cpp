// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// Stage 3: server-side model preprocessing.
//
// The harness invokes this stage with NO arguments (run_submission.py:97), so
// it cannot know the instance size and cannot reach io/<size>/public_keys --
// nothing here may depend on the instance size or on any key material.
//
// For the HS scheme that is not a limitation. Encoding the layer diagonals is
// the expensive part of building the model, and it depends only on the
// weights, the layer geometry and the level/scale -- all compile-time
// constants -- plus the switching-key gadget decomposition, which is a *shape*
// carrying no key material (mlp::makeSwkParams). So it happens here, and
// stage 7 only binds the keys to the result. Encoding runs on the device the
// evaluation later runs on.
//
// PCMM's model encoding cannot move here -- its shapes depend on the batch
// size -- but it is cheap. Its raw weights are cached here so the CSVs are
// parsed once.
//
// See "Stage split" in DESIGN.md for how the timing is reported.

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

    // HS: padded p x q diagonal layout. Only the bias is read back in stage 7
    // now, but the padded weights are what the diagonals are built from here.
    const auto fc1 = padWeights(FC1, W1, b1);
    const auto fc2 = padWeights(FC2, W2, b2);
    writeLayerWeights(fc1, std::string(CACHE_DIR) + "/fc1.bin");
    writeLayerWeights(fc2, std::string(CACHE_DIR) + "/fc2.bin");

    // PCMM: raw CSV shapes, unpadded.
    pcmm::writeRawModel({W1, b1, W2, b2}, std::string(CACHE_DIR) + "/pcmm.bin");

    // HS: the expensive part -- encode both layers' diagonals, no keys needed.
    // Several seconds of work, skipped on a PCMM instance, which never looks
    // at them; the marker is the only way this stage can know. An absent
    // marker means prepare everything.
    const auto marked = readInstanceMarker();
    if (marked && usePcmm(*marked)) {
        std::cout << "         [server] model cached to " << CACHE_DIR
                  << " (PCMM instance: HS diagonals not needed)\n";
        return 0;
    }

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

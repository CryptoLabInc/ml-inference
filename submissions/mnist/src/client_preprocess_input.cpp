// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// Stage 5: cleartext client-side input preparation.
//
// Two operations, both in the clear, both documented in DESIGN.md:
//   1. normalize (p - MNIST_MEAN) / MNIST_STD -- the harness writes pixels
//      already scaled to [0,1]; this is the same affine map the harness's own
//      reference model applies in harness/mnist/test.py.
//   2. center-crop 28x28 -> 22x22 (784 -> 484) -- the model was TRAINED on
//      cropped input, so this is part of the model definition. It also halves
//      the padded coordinate dimension, which is what fits 128 images per
//      ciphertext instead of 64. See "The crop" in DESIGN.md.

#include "mlp_pipeline.hpp"

#include <iostream>

using namespace mlp;

int main(int argc, char *argv[]) try {
    const auto size = parseInstanceSize(argc, argv);
    const InstanceParams prms(size);

    auto dataset = readSamples(prms.test_input_file().string(), MNIST_DIM);
    if (dataset.size() != prms.getBatchSize())
        throw std::runtime_error("dataset size " +
                                 std::to_string(dataset.size()) +
                                 " does not match instance batch size " +
                                 std::to_string(prms.getBatchSize()));

    std::vector<std::vector<double>> out;
    out.reserve(dataset.size());
    for (const auto &img : dataset) {
        std::vector<double> cropped(INPUT_DIM);
        for (u32 r = 0; r < CROP_DIM; ++r)
            for (u32 c = 0; c < CROP_DIM; ++c) {
                const double p = img[(r + CROP) * IMG_DIM + (c + CROP)];
                cropped[r * CROP_DIM + c] = (p - MNIST_MEAN) / MNIST_STD;
            }
        out.push_back(std::move(cropped));
    }

    fs::create_directories(prms.iointermdir());
    writeSamples(out, prms.preprocessed_input_file().string());
    return 0;
} catch (const std::exception &e) {
    std::cerr << "client_preprocess_input: " << e.what() << "\n";
    return 1;
}

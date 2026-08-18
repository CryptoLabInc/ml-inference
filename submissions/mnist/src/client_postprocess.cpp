// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#include "mlp_pipeline.hpp"

#include <fstream>
#include <iostream>

using namespace mlp;

int main(int argc, char *argv[]) try {
    const auto size = parseInstanceSize(argc, argv);
    const InstanceParams prms(size);

    const auto scores =
        readSamples(prms.model_scores_file().string(), LABEL_DIM);
    if (scores.size() != prms.getBatchSize())
        throw std::runtime_error("score rows " + std::to_string(scores.size()) +
                                 " do not match instance batch size " +
                                 std::to_string(prms.getBatchSize()));

    std::ofstream out(prms.encrypted_model_predictions_file());
    if (!out.good())
        throw std::runtime_error(
            "cannot write " + prms.encrypted_model_predictions_file().string());

    for (const auto &row : scores) {
        size_t best = 0;
        for (size_t r = 1; r < row.size(); ++r)
            if (row[r] > row[best])
                best = r;
        out << best << '\n';
    }
    if (!out)
        throw std::runtime_error("short write to predictions file");
    return 0;
} catch (const std::exception &e) {
    std::cerr << "client_postprocess: " << e.what() << "\n";
    return 1;
}

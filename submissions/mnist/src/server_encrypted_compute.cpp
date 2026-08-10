// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// Stage 7: the encrypted inference. Everything the model computes happens
// here, on ciphertext:
//
//     fc1 matvec -> fold -> +b1 -> x^2 -> rescale -> fc2 matvec -> +b2
//
// The server holds no secret key. It loads the evaluation keys the client
// published and the cleartext model it owns.
//
// Setup (loading keys, constructing the two MatrixVectorEvals -- which is
// where the weight diagonals get encoded) is timed separately from evaluation
// and both are reported through io/<size>/server_reported_steps.json. Stage 3
// could not do the setup: it never learns the instance size. See README.md.

#include "mlp_pipeline.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace heaan;
using namespace mlp;

namespace {

class Timer {
public:
    Timer() : beg_(std::chrono::high_resolution_clock::now()) {}
    double seconds() const {
        return std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - beg_)
            .count();
    }

private:
    std::chrono::high_resolution_clock::time_point beg_;
};

constexpr const char *CACHE_DIR = "submissions/mnist/build/model_cache";

} // namespace

int main(int argc, char *argv[]) try {
    const auto size = parseInstanceSize(argc, argv);
    const InstanceParams prms(size);
    const Device dev = targetDevice();

    const Timer t_setup;

    const Levels levels = buildLevels();
    const u32 top = levels.top();
    const EnDecoder encoder = makeEncoder(levels);
    const HomEval eval{HomEvalParams{levels}};
    // adjust() lives on HomEvalFlexible; it is stateless and cannot be merged
    // with HomEval, whose 3-argument tensor() the 4-argument one would hide.
    const HomEvalFlexible flex;

    const u32 fc1_in = top - FC1_IN_DROP, fc1_out = top - FC1_OUT_DROP;
    const u32 fc2_in = top - FC2_IN_DROP, fc2_out = top - FC2_OUT_DROP;

    auto rot_fc1 = std::make_unique<RotKeyPtrs>(serial::load<RotKeyPtrs>(
        (prms.pubkeydir() / ROT_KEY_FC1_FILE).string(), dev));
    auto rot_fc2 = std::make_unique<RotKeyPtrs>(serial::load<RotKeyPtrs>(
        (prms.pubkeydir() / ROT_KEY_FC2_FILE).string(), dev));
    auto relin_key = serial::loadAsPtr<ISwKey>(
        (prms.pubkeydir() / RELIN_KEY_FILE).string(), dev);

    const auto w1 = readLayerWeights(FC1, std::string(CACHE_DIR) + "/fc1.bin");
    const auto w2 = readLayerWeights(FC2, std::string(CACHE_DIR) + "/fc2.bin");

    const Layer fc1 = makeLayer(FC1, w1, levels, fc1_in, fc1_out, encoder,
                                std::move(rot_fc1), std::move(relin_key), dev);
    const Layer fc2 = makeLayer(FC2, w2, levels, fc2_in, fc2_out, encoder,
                                std::move(rot_fc2), KeyPtr{}, dev);

    const double setup_s = t_setup.seconds();

    // ---- evaluate ----
    fs::create_directories(prms.ctxtdowndir());
    const size_t n_ct = numCtxts(prms.getBatchSize());
    double eval_s = 0.0;

    for (size_t j = 0; j < n_ct; ++j) {
        const auto in_path =
            prms.ctxtupdir() / ("cipher_input_" + std::to_string(j) + ".bin");
        auto ct = serial::loadAsPtr<ICiphertext>(in_path.string(), dev);

        const Timer t_eval;
        homLayer(ct, fc1, eval, flex);
        homLayer(ct, fc2, eval, flex);
        eval_s += t_eval.seconds();

        serial::save((prms.ctxtdowndir() /
                      ("cipher_result_" + std::to_string(j) + ".bin"))
                         .string(),
                     *ct);
    }

    std::cout << "         [server] " << n_ct << " ciphertext(s), setup "
              << setup_s << "s, eval " << eval_s << "s\n";

    std::ofstream json(prms.server_reported_steps_file());
    json << std::fixed << std::setprecision(6) << "{\n"
         << "  \"Model setup (key load + diagonal encoding)\": " << setup_s
         << ",\n"
         << "  \"Encrypted computation\": " << eval_s << ",\n"
         << "  \"Total\": " << (setup_s + eval_s) << "\n}\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << "server_encrypted_compute: " << e.what() << "\n";
    return 1;
}

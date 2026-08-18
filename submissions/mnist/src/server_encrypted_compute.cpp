// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#include "mlp_pcmm.hpp"
#include "mlp_pipeline.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>

#ifdef MLP_WITH_CUDA
#include <cuda_runtime.h>
#endif

using namespace heaan;
using namespace mlp;

namespace {

class Timer {
public:
    Timer() {
        sync();
        beg_ = std::chrono::high_resolution_clock::now();
    }
    double seconds() const {
        sync();
        return std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - beg_)
            .count();
    }

private:
    static void sync() {
#ifdef MLP_WITH_CUDA
        ::cudaDeviceSynchronize();
#endif
    }
    std::chrono::high_resolution_clock::time_point beg_;
};

constexpr const char *CACHE_DIR = MODEL_CACHE_DIR;

struct StageTimes {
    double setup_s = 0.0;
    double warmup_s = 0.0;
    double eval_s = 0.0;
};

StageTimes runHS(const InstanceParams &prms, Device dev) {
    const Timer t_setup;

    const Levels levels = buildLevels();
    const u32 top = levels.top();
    const EnDecoder encoder = makeEncoder(levels);
    const HomEval eval{HomEvalParams{levels}};
    const HomEvalFlexible flex;

    const u32 fc1_in = top - FC1_IN_DROP, fc1_out = top - FC1_OUT_DROP;
    const u32 fc2_in = top - FC2_IN_DROP, fc2_out = top - FC2_OUT_DROP;

    auto rot_fc1 = std::make_unique<RotKeyPtrs>(serial::load<RotKeyPtrs>(
        (prms.pubkeydir() / ROT_KEY_FC1_FILE).string(), dev));
    auto rot_fc2 = std::make_unique<RotKeyPtrs>(serial::load<RotKeyPtrs>(
        (prms.pubkeydir() / ROT_KEY_FC2_FILE).string(), dev));
    auto relin_key = serial::loadAsPtr<ISwKey>(
        (prms.pubkeydir() / RELIN_KEY_FILE).string(), dev);

    const auto raw =
        pcmm::readRawModel(std::string(CACHE_DIR) + "/pcmm.bin");

    const auto enc1 = serial::load<MatrixVectorEvalEncoded>(
        std::string(CACHE_DIR) + "/fc1_diags.bin", dev);
    const auto enc2 = serial::load<MatrixVectorEvalEncoded>(
        std::string(CACHE_DIR) + "/fc2_diags.bin", dev);

    auto fold_keys = serial::load<RotKeyPtrs>(
        (prms.pubkeydir() / ROT_KEY_FOLD_FILE).string(), dev);

    const Layer fc1 = makeLayer(FC1, raw.b1, enc1, levels, fc1_in, fc1_out,
                                encoder, std::move(rot_fc1),
                                std::move(relin_key), dev,
                                std::move(fold_keys));
    const Layer fc2 = makeLayer(FC2, raw.b2, enc2, levels, fc2_in, fc2_out,
                                encoder, std::move(rot_fc2), KeyPtr{}, dev);

    StageTimes tm;
    tm.setup_s = t_setup.seconds();

    fs::create_directories(prms.ctxtdowndir());
    const size_t n_ct = numCtxts(prms.getBatchSize());

    {
        const Timer t_warm;
        auto warm = serial::loadAsPtr<ICiphertext>(
            (prms.ctxtupdir() / "cipher_input_0.bin").string(), dev);
        homLayer(warm, fc1, eval, flex);
        homLayer(warm, fc2, eval, flex);
        tm.warmup_s = t_warm.seconds();
    }

    for (size_t j = 0; j < n_ct; ++j) {
        const auto in_path =
            prms.ctxtupdir() / ("cipher_input_" + std::to_string(j) + ".bin");
        auto ct = serial::loadAsPtr<ICiphertext>(in_path.string(), dev);

        const Timer t_eval;
        homLayer(ct, fc1, eval, flex);
        homLayer(ct, fc2, eval, flex);
        tm.eval_s += t_eval.seconds();

        serial::save((prms.ctxtdowndir() /
                      ("cipher_result_" + std::to_string(j) + ".bin"))
                         .string(),
                     *ct);
    }

    std::cout << "         [server] " << n_ct << " ciphertext(s), setup "
              << tm.setup_s << "s, warm-up " << tm.warmup_s << "s, eval "
              << tm.eval_s << "s\n";
    return tm;
}

StageTimes runPcmm(const InstanceParams &prms, Device dev, InstanceSize size) {
    const Timer t_setup;

    const auto prof = pcmm::profile(size);
    const Levels levels = pcmm::buildLevels(prof);
    const EnDecoder coeff_encoder = pcmm::makeCoeffEncoder(levels, prof);
    const auto num_images = static_cast<u32>(prms.getBatchSize());

    auto relin_key = serial::loadAsPtr<ISwKey>(
        (prms.pubkeydir() / RELIN_KEY_FILE).string(), dev);

    const auto raw =
        pcmm::readRawModel(std::string(CACHE_DIR) + "/pcmm.bin");
    const auto model = pcmm::buildModel(raw.W1, raw.b1, raw.W2, raw.b2,
                                        coeff_encoder, levels, num_images,
                                        prof);

    StageTimes tm;
    tm.setup_s = t_setup.seconds();

    fs::create_directories(prms.ctxtdowndir());
    auto cx = pcmm::loadCtMatrix(
        (prms.ctxtupdir() / pcmm::INPUT_CTMATRIX_FILE).string(), pcmm::IN_P,
        pcmm::numCols(prof, num_images), dev);

    {
        const Timer t_warm;
        auto warm = ICtMatrix::make();
        pcmm::inference(model, *relin_key, *cx, *warm, levels, num_images,
                        prof);
        tm.warmup_s = t_warm.seconds();
    }

    const Timer t_eval;
    auto cy = ICtMatrix::make();
    pcmm::inference(model, *relin_key, *cx, *cy, levels, num_images, prof);
    tm.eval_s = t_eval.seconds();

    pcmm::saveCtMatrix(
        (prms.ctxtdowndir() / pcmm::RESULT_CTMATRIX_FILE).string(), *cy);

    std::cout << "         [server] 1 ciphertext matrix (" << num_images
              << " images), setup " << tm.setup_s << "s, warm-up "
              << tm.warmup_s << "s, eval " << tm.eval_s << "s\n";
    return tm;
}

} // namespace

int main(int argc, char *argv[]) try {
    const auto size = parseInstanceSize(argc, argv);
    const InstanceParams prms(size);
    const Device dev = targetDevice();

    const StageTimes tm =
        usePcmm(size) ? runPcmm(prms, dev, size) : runHS(prms, dev);

    std::ofstream json(prms.server_reported_steps_file());
    json << std::fixed << std::setprecision(6) << "{\n"
         << "  \"Model setup (key load + diagonal/weight encoding)\": "
         << tm.setup_s << ",\n"
         << "  \"Warm-up (one discarded inference pass)\": " << tm.warmup_s
         << ",\n"
         << "  \"Encrypted computation\": " << tm.eval_s << ",\n"
         << "  \"Total\": " << (tm.setup_s + tm.warmup_s + tm.eval_s)
         << "\n}\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << "server_encrypted_compute: " << e.what() << "\n";
    return 1;
}

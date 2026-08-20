// Copyright (c) 2026 CryptoLab, Inc.
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
        runPcmm(prms, dev, size);

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

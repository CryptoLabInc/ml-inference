// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// Stage 7: the encrypted inference. Everything the model computes happens
// here, on ciphertext. Dispatches on instance size (mlp::usePcmm):
//
//   HS (single):               fc1 matvec -> fold -> +b1 -> x^2 -> rescale ->
//                               fc2 matvec -> +b2
//   PCMM (small/medium/large): fc1 pcmm+b1 -> x^2 -> rescale -> fc2 pcmm -> +b2
//
// The server holds no secret key in either scheme. It loads the evaluation
// keys the client published and the cleartext model it owns.
//
// Setup (loading keys, and -- scheme-dependently -- constructing the two
// MatrixVectorEvals or encoding U1/U2/b2, both of which need the instance
// size) is timed separately from evaluation, and both are reported through
// io/<size>/server_reported_steps.json. Stage 3 could not do this setup: it
// never learns the instance size. See "Stage split" in DESIGN.md.

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

// Elapsed seconds, synchronizing the device at both ends so a GPU timing
// measures work *completed* rather than kernels *launched*. HEaaN2 kernel
// launches are asynchronous: without the sync the evaluation timer closes
// while the GPU is still working, and the stage under-reports by an order of
// magnitude (the cost then silently surfaces in whatever forces completion
// next -- here the result serialization, which the harness still counts). The
// library's own mlp benchmarks sync for the same reason, so this also keeps
// the two sets of numbers comparable.
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

// One full discarded inference pass before the timed evaluation, the same
// warm-up HEaaN2's own mlp benchmarks run: the first use of each CUDA kernel
// pays module loading, the first allocation grows the memory pool, and the
// NTT workspace is built lazily. Without it those one-time costs land inside
// the timed evaluation and the reported number is not comparable to a warm
// server (or to HEaaN2's benchmark figures). The harness times this stage as
// one process, so the warm-up does not move any cost out of the harness's
// "Encrypted computation" -- it costs about one extra warm evaluation there,
// and is reported as its own line in server_reported_steps.json.
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

    // The diagonals were encoded by server_preprocess_model (stage 3), which
    // needs no keys to do it; this stage only binds the rotation keys to them.
    const auto enc1 = serial::load<MatrixVectorEvalEncoded>(
        std::string(CACHE_DIR) + "/fc1_diags.bin", dev);
    const auto enc2 = serial::load<MatrixVectorEvalEncoded>(
        std::string(CACHE_DIR) + "/fc2_diags.bin", dev);

    // fc1's fold keys, present whenever the secret key is not lifted (the
    // shipped configuration). Absent for a lifted key, where fc1 folds with
    // bare automorphisms and makeLayer ignores the empty set.
    RotKeyPtrs fold_keys;
    const auto fold_path = prms.pubkeydir() / ROT_KEY_FOLD_FILE;
    if (fs::exists(fold_path))
        fold_keys = serial::load<RotKeyPtrs>(fold_path.string(), dev);

    const Layer fc1 = makeLayer(FC1, w1.b, enc1, levels, fc1_in, fc1_out,
                                encoder, std::move(rot_fc1),
                                std::move(relin_key), dev,
                                std::move(fold_keys));
    const Layer fc2 = makeLayer(FC2, w2.b, enc2, levels, fc2_in, fc2_out,
                                encoder, std::move(rot_fc2), KeyPtr{}, dev);

    StageTimes tm;
    tm.setup_s = t_setup.seconds();

    fs::create_directories(prms.ctxtdowndir());
    const size_t n_ct = numCtxts(prms.getBatchSize());

    // Warm-up on a throwaway reload of the first input ciphertext; the result
    // is discarded, so re-running the real inputs afterwards stays correct.
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

    // Warm-up: same inference on the real input (which inference() does not
    // modify), result discarded.
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

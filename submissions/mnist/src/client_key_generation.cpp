// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#include "mlp_pcmm.hpp"
#include "mlp_pipeline.hpp"

#include <iostream>
#include <set>

using namespace heaan;
using namespace mlp;

namespace {

void runHS(const InstanceParams &prms) {
    const Levels levels = buildLevels();
    const u32 top = levels.top();

    SKGenerator skgen{SKGenParams{LOG_DEGREE, HW, NTT_ALG}};
    auto sk = skgen.genKey();

    const u32 fc1_stride = IMAGES_PER_CTXT * FC1.p;

    EncKeyGenerator enckeygen{EncKeyGenParams{DiscreteGaussian(NOISE_STDDEV),
                                              POLY_TYPE, levels.mods[top],
                                              NTT_ALG}};
    auto enc_key = enckeygen.genKey(*sk);

    const u32 fc1_in = top - FC1_IN_DROP, fc1_out = top - FC1_OUT_DROP;
    const u32 fc2_in = top - FC2_IN_DROP;

    SwKeyGenerator rot1_gen(makeSwkParams(levels, fc1_in));
    auto rot_keys_fc1 = rot1_gen.genRotKeys(
        *sk, MatrixVectorEval::rotKeyIndices(
                 makeMvParams(FC1, levels, fc1_in, levels.scales[fc1_in])));

    SwKeyGenerator rot2_gen(makeSwkParams(levels, fc2_in));
    auto rot_keys_fc2 = rot2_gen.genRotKeys(
        *sk, MatrixVectorEval::rotKeyIndices(
                 makeMvParams(FC2, levels, fc2_in, levels.scales[fc2_in])));

    SwKeyGenerator relin_gen(makeSwkParams(levels, fc1_out));
    auto relin_key = relin_gen.genRelinKey(*sk);

    std::set<i32> fold_steps;
    for (u32 j = 1; j < FC1.q / FC1.p; ++j)
        fold_steps.insert(static_cast<i32>(fc1_stride * j));
    SwKeyGenerator fold_gen(makeSwkParams(levels, fc1_out));
    auto fold_keys = fold_gen.genRotKeys(*sk, fold_steps);

    fs::create_directories(prms.pubkeydir());
    fs::create_directories(prms.seckeydir());

    serial::save((prms.seckeydir() / SECRET_KEY_FILE).string(), *sk);
    serial::save((prms.pubkeydir() / ENC_KEY_FILE).string(), *enc_key);
    serial::save((prms.pubkeydir() / ROT_KEY_FC1_FILE).string(), rot_keys_fc1);
    serial::save((prms.pubkeydir() / ROT_KEY_FC2_FILE).string(), rot_keys_fc2);
    serial::save((prms.pubkeydir() / ROT_KEY_FOLD_FILE).string(), fold_keys);
    serial::save((prms.pubkeydir() / RELIN_KEY_FILE).string(), *relin_key);
}

void runPcmm(const InstanceParams &prms, InstanceSize size) {
    const auto prof = pcmm::profile(size);
    const Levels levels = pcmm::buildLevels(prof);

    SKGenerator skgen{SKGenParams{prof.log_degree, HW, prof.ntt_alg}};
    auto sk = skgen.genKey();

    auto relin_key = pcmm::genRelinKey(*sk, levels, prof);

    fs::create_directories(prms.pubkeydir());
    fs::create_directories(prms.seckeydir());

    serial::save((prms.seckeydir() / SECRET_KEY_FILE).string(), *sk);
    serial::save((prms.pubkeydir() / RELIN_KEY_FILE).string(), *relin_key);
}

} // namespace

int main(int argc, char *argv[]) try {
    const auto size = parseInstanceSize(argc, argv);
    const InstanceParams prms(size);
    writeInstanceMarker(size);

    if (usePcmm(size))
        runPcmm(prms, size);
    else
        runHS(prms);

    std::cout << "         [client] keys written to " << prms.pubkeydir()
              << "\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << "client_key_generation: " << e.what() << "\n";
    return 1;
}

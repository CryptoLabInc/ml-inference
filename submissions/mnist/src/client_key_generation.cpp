// Copyright (c) 2026 CryptoLab, Inc.
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

    runPcmm(prms, size);

    std::cout << "         [client] keys written to " << prms.pubkeydir()
              << "\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << "client_key_generation: " << e.what() << "\n";
    return 1;
}

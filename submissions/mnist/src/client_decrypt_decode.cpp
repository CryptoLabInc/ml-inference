// Copyright (c) 2026 CryptoLab, Inc.
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

std::vector<std::vector<double>> runPcmm(const InstanceParams &prms, Device dev,
                                         InstanceSize size) {
    const auto prof = pcmm::profile(size);
    const Levels levels = pcmm::buildLevels(prof);
    const EnDecoder coeff_encoder = pcmm::makeCoeffEncoder(levels, prof);
    const EnDecryptor encryptor{EncryptParams{DiscreteGaussian(NOISE_STDDEV)}};

    auto sk = serial::loadAsPtr<ISecretKey>(
        (prms.seckeydir() / SECRET_KEY_FILE).string(), dev);

    const auto num_images = static_cast<u32>(prms.getBatchSize());
    auto cy = pcmm::loadCtMatrix(
        (prms.ctxtdowndir() / pcmm::RESULT_CTMATRIX_FILE).string(),
        LABEL_DIM, pcmm::numCols(prof, num_images), dev);

    pcmm::setDFT(*cy, /*dft=*/true, num_images, prof);
    auto dy = IPtMatrix::make();
    encryptor.decrypt(*cy, *sk, *dy);
    Matrix<Real> yc;
    coeff_encoder.decode(*dy, yc);
    yc.to(Device::CPU);

    return pcmm::unpackLogits(yc, num_images, prof);
}

} // namespace

int main(int argc, char *argv[]) try {
    const auto size = parseInstanceSize(argc, argv);
    const InstanceParams prms(size);
    const Device dev = targetDevice();

    const auto scores =
        runPcmm(prms, dev, size);

    fs::create_directories(prms.iointermdir());
    writeSamples(scores, prms.model_scores_file().string());
    return 0;
} catch (const std::exception &e) {
    std::cerr << "client_decrypt_decode: " << e.what() << "\n";
    return 1;
}

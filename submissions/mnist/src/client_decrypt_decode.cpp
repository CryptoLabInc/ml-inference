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

std::vector<std::vector<double>> runHS(const InstanceParams &prms,
                                       Device dev) {
    const Levels levels = buildLevels();
    const EnDecoder encoder = makeEncoder(levels);
    const EnDecryptor encryptor{EncryptParams{DiscreteGaussian(NOISE_STDDEV)}};

    auto sk = serial::loadAsPtr<ISecretKey>(
        (prms.seckeydir() / SECRET_KEY_FILE).string(), dev);

    const size_t batch = prms.getBatchSize();
    const size_t n_ct = numCtxts(batch);

    std::vector<std::vector<double>> scores;
    scores.reserve(batch);

    for (size_t j = 0; j < n_ct; ++j) {
        const auto path = prms.ctxtdowndir() /
                          ("cipher_result_" + std::to_string(j) + ".bin");
        auto ct = serial::loadAsPtr<ICiphertext>(path.string(), dev);

        auto ptxt = IPlaintext::make(PtxtType::NORMAL);
        Message out;
        encryptor.decrypt(*ct, *sk, *ptxt);
        encoder.decode(*ptxt, out);
        out.to(Device::CPU);

        const size_t first = j * IMAGES_PER_CTXT;
        const size_t count = std::min<size_t>(IMAGES_PER_CTXT, batch - first);
        for (size_t i = 0; i < count; ++i) {
            std::vector<double> logits(LABEL_DIM);
            for (u32 r = 0; r < LABEL_DIM; ++r)
                logits[r] = out[slotOf(r, static_cast<u32>(i))].real();
            scores.push_back(std::move(logits));
        }
    }
    return scores;
}

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
        usePcmm(size) ? runPcmm(prms, dev, size) : runHS(prms, dev);

    fs::create_directories(prms.iointermdir());
    writeSamples(scores, prms.model_scores_file().string());
    return 0;
} catch (const std::exception &e) {
    std::cerr << "client_decrypt_decode: " << e.what() << "\n";
    return 1;
}

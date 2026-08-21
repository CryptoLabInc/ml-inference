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

void run(const InstanceParams &prms, Device dev, InstanceSize size) {
    const auto prof = pcmm::profile(size);
    const Levels levels = pcmm::buildLevels(prof);
    const u32 top = levels.top();
    const EnDecoder slot_encoder = pcmm::makeSlotEncoder(levels, prof);
    EnDecryptor encryptor{EncryptParams{DiscreteGaussian(NOISE_STDDEV)}};

    auto sk = serial::loadAsPtr<ISecretKey>(
        (prms.seckeydir() / SECRET_KEY_FILE).string(), dev);

    auto images =
        readSamples(prms.preprocessed_input_file().string(), INPUT_DIM);
    const u32 num_images = static_cast<u32>(images.size());
    if (num_images != prms.getBatchSize())
        throw std::runtime_error("preprocessed input size does not match "
                                 "instance batch size");

    auto x = pcmm::packImages(images, prof);
    auto px = pcmm::encodeMatrix(slot_encoder, x, top);

    auto cx = ICtMatrix::make();
    encryptor.encrypt(*px, *sk, *cx);
    pcmm::setDFT(*cx, /*dft=*/false, num_images, prof);

    fs::create_directories(prms.ctxtupdir());
    pcmm::saveCtMatrix(
        (prms.ctxtupdir() / pcmm::INPUT_CTMATRIX_FILE).string(), *cx);

    std::cout << "         [client] encrypted " << num_images
              << " images into 1 ciphertext matrix\n";
}

} // namespace

int main(int argc, char *argv[]) try {
    const auto size = parseInstanceSize(argc, argv);
    const InstanceParams prms(size);
    const Device dev = targetDevice();

    run(prms, dev, size);

    return 0;
} catch (const std::exception &e) {
    std::cerr << "client_encode_encrypt_input: " << e.what() << "\n";
    return 1;
}

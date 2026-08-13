// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// Stage 6: pack, encode and encrypt the batch.
//
// HS: IMAGES_PER_CTXT images ride in one ciphertext at slot(c,i) = c*128 + i,
// so a batch of n needs ceil(n/128) ciphertexts rather than n.
//
// PCMM: the whole batch packs into ONE ICtMatrix (its internal "blocks"
// machinery is what absorbs a batch larger than one message), encrypted
// under the client's own secret key -- see the note in
// client_key_generation.cpp on why PCMM has no public-key encrypt available.
//
// Either way, the harness only measures the byte size of ciphertexts_upload/,
// so the file layout is ours to choose.

#include "mlp_pcmm.hpp"
#include "mlp_pipeline.hpp"

#include <iostream>

using namespace heaan;
using namespace mlp;

namespace {

void runHS(const InstanceParams &prms, Device dev) {
    const Levels levels = buildLevels();
    const u32 top = levels.top();
    const EnDecoder encoder = makeEncoder(levels);
    EnDecryptor encryptor{EncryptParams{DiscreteGaussian(NOISE_STDDEV)}};

    auto enc_key = serial::loadAsPtr<IEncKey>(
        (prms.pubkeydir() / ENC_KEY_FILE).string(), dev);

    auto images =
        readSamples(prms.preprocessed_input_file().string(), INPUT_DIM);
    if (images.size() != prms.getBatchSize())
        throw std::runtime_error("preprocessed input size does not match "
                                 "instance batch size");

    fs::create_directories(prms.ctxtupdir());
    const size_t n_ct = numCtxts(images.size());
    for (size_t j = 0; j < n_ct; ++j) {
        const size_t first = j * IMAGES_PER_CTXT;
        const size_t count =
            std::min<size_t>(IMAGES_PER_CTXT, images.size() - first);
        auto msg = packImages(images, first, count);
        msg.to(dev);

        auto ptxt = IPlaintext::make(PtxtType::NORMAL);
        auto ctxt = ICiphertext::make(EncType::RLWE);
        encoder.encode(msg, *ptxt, top);
        encryptor.encrypt(*ptxt, *enc_key, *ctxt);

        serial::save(
            (prms.ctxtupdir() / ("cipher_input_" + std::to_string(j) + ".bin"))
                .string(),
            *ctxt);
    }

    std::cout << "         [client] encrypted " << images.size()
              << " images into " << n_ct << " ciphertext(s)\n";
}

void runPcmm(const InstanceParams &prms, Device dev, InstanceSize size) {
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
    encryptor.encrypt(*px, *sk, *cx); // r_ntt=false -> coeff domain
    // section 6.3: relabel as coefficient-encoded before it ever reaches
    // pcmm on the server (pcmm rejects operands whose dft flag is set).
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

    if (usePcmm(size))
        runPcmm(prms, dev, size);
    else
        runHS(prms, dev);

    return 0;
} catch (const std::exception &e) {
    std::cerr << "client_encode_encrypt_input: " << e.what() << "\n";
    return 1;
}

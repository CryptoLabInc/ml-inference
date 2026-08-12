// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// Stage 2.2: generate all key material at the client.
//
// Dispatches on instance size (mlp::usePcmm): single uses the Halevi-Shoup
// layer scheme below; small/medium/large use PCMM (mlp_pcmm.hpp).
// The two schemes need different key material -- HS wants rotation keys for
// its two matvec layers plus a public encryption key; PCMM needs neither
// rotation keys nor a public encryption key (its matrix encrypt is
// necessarily symmetric-key, see runPcmm) but does need a relinearization
// key at a different level. Each size only ever runs one scheme, so the two
// keygen paths never collide inside the same io/<size>/ directory tree.

#include "mlp_pcmm.hpp"
#include "mlp_pipeline.hpp"

#include <iostream>

using namespace heaan;
using namespace mlp;

namespace {

// The secret key is sampled at 2^SMALL_LOG_DEGREE and lifted to 2^LOG_DEGREE.
// Lifting adds no entropy -- it buys fc1's key-less fold, and the switching
// key budget is sized from the sampled degree. See mlp_params.hpp.
//
// Rotation keys are derived from the layer *shapes* only. The client has no
// model and is not entitled to one, so nothing here may depend on the weights.
void runHS(const InstanceParams &prms) {
    const Levels levels = buildLevels();
    const u32 top = levels.top();

    // ---- secret key: sampled low, lifted high ----
    SKGenerator skgen{SKGenParams{LOG_DEGREE, HW, NTT_ALG}};
    SKGenerator skgen_low{SKGenParams{SMALL_LOG_DEGREE, HW, NTT_ALG}};
    auto sk = skgen.genHighDegreeKey(*skgen_low.genKey());

    // The key-less fold is valid only when fc1's fold stride divides evenly
    // into the lifted key's invariance period. Checked here so a change to the
    // packing fails loudly at key generation rather than decrypting to noise.
    const u32 period = rotInvariantPeriod(SMALL_LOG_DEGREE);
    const u32 fc1_stride = IMAGES_PER_CTXT * FC1.p;
    if (fc1_stride % period != 0)
        throw std::runtime_error(
            "fc1 fold stride " + std::to_string(fc1_stride) +
            " is not a multiple of the rotation-invariance period " +
            std::to_string(period) + "; the key-less fold would be wrong");

    // ---- public encryption key, at the level inputs are encrypted to ----
    EncKeyGenerator enckeygen{EncKeyGenParams{DiscreteGaussian(NOISE_STDDEV),
                                              POLY_TYPE, levels.mods[top],
                                              NTT_ALG}};
    auto enc_key = enckeygen.genKey(*sk);

    // ---- switching keys ----
    // makeSwkParams() is shared with server_preprocess_model, which encodes the
    // layer diagonals against the same gadget decomposition. Keep them going
    // through that one function: if the two disagreed, the encoded diagonals
    // would not bind to these keys.
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

    // Relinearization for the x^2, at the level the squaring happens on.
    SwKeyGenerator relin_gen(makeSwkParams(levels, fc1_out));
    auto relin_key = relin_gen.genRelinKey(*sk);

    // ---- serialize ----
    fs::create_directories(prms.pubkeydir());
    fs::create_directories(prms.seckeydir());

    serial::save((prms.seckeydir() / SECRET_KEY_FILE).string(), *sk);
    serial::save((prms.pubkeydir() / ENC_KEY_FILE).string(), *enc_key);
    serial::save((prms.pubkeydir() / ROT_KEY_FC1_FILE).string(), rot_keys_fc1);
    serial::save((prms.pubkeydir() / ROT_KEY_FC2_FILE).string(), rot_keys_fc2);
    serial::save((prms.pubkeydir() / RELIN_KEY_FILE).string(), *relin_key);
}

// PCMM's ISecretKey is sampled directly at pcmm::LOG_DEGREE (no lifting: pcmm
// has no key-less fold to buy with one) and needs no rotation keys at all --
// only a relinearization key for the x^2 step.
//
// HEaaN2's public EnDecryptor exposes matrix encrypt/decrypt only against a
// secret key (there is no public-encryption-key overload for IPtMatrix /
// ICtMatrix, unlike the plain-ciphertext overload the HS path uses above).
// The client's own sk is therefore what client_encode_encrypt_input encrypts
// with, and what this stage saves to seckeydir() -- still exclusively a
// client-side operation, and sk never leaves seckeydir() (which the harness
// does not measure), but it is a real, disclosed asymmetry against the HS
// path's public-key encryption. See "Encryption: public key vs symmetric key"
// in DESIGN.md.
void runPcmm(const InstanceParams &prms) {
    const Levels levels = pcmm::buildLevels();

    SKGenerator skgen{SKGenParams{pcmm::LOG_DEGREE, pcmm::HW, pcmm::NTT_ALG}};
    auto sk = skgen.genKey();

    auto relin_key = pcmm::genRelinKey(*sk, levels, pcmm::SWK_MARGIN);

    fs::create_directories(prms.pubkeydir());
    fs::create_directories(prms.seckeydir());

    serial::save((prms.seckeydir() / SECRET_KEY_FILE).string(), *sk);
    serial::save((prms.pubkeydir() / RELIN_KEY_FILE).string(), *relin_key);
}

} // namespace

int main(int argc, char *argv[]) try {
    const auto size = parseInstanceSize(argc, argv);
    const InstanceParams prms(size);
    // Stage 3 runs after this one but is given no arguments, so it cannot tell
    // which scheme to prepare. Record the instance size on the way past -- see
    // the instance-marker note in mlp_pipeline.hpp.
    writeInstanceMarker(size);

    // Deliberately no targetDevice() here: key generation runs on the CPU. The
    // client is a separate party and need not own a GPU, and the keys are
    // serialized either way -- server_encrypted_compute is what loads them onto
    // the device. Do not "fix" this by moving the key material to the GPU
    // without also re-checking what the harness then attributes to stage 2.2.

    if (usePcmm(size))
        runPcmm(prms);
    else
        runHS(prms);

    std::cout << "         [client] keys written to " << prms.pubkeydir()
              << "\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << "client_key_generation: " << e.what() << "\n";
    return 1;
}

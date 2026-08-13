// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#include "mlp_pcmm.hpp"
#include "mlp_pipeline.hpp" // mlp::targetDevice()

#include <fstream>
#include <stdexcept>

using namespace heaan;

namespace mlp::pcmm {

//===========================================================================
// Parameter reconstruction
//===========================================================================

Levels buildLevels(const Profile &p) {
    paramsUtils::LevelsBuilder lb;
    lb.setRing(p.log_degree, p.poly_type);
    lb.initMod(p.base_bits);
    return lb.buildAbove(NUM_MULTS, p.rescale_bits);
}

EnDecoder makeCoeffEncoder(const Levels &levels, const Profile &p) {
    return EnDecoder(
        EncodeParams(p.poly_type, p.log_degree, levels, p.ntt_alg,
                    /*coeff_encoding=*/true));
}

EnDecoder makeSlotEncoder(const Levels &levels, const Profile &p) {
    return EnDecoder(
        EncodeParams(p.poly_type, p.log_degree, levels, p.ntt_alg,
                    /*coeff_encoding=*/false));
}

//===========================================================================
// Packing
//===========================================================================

Matrix<Real> packImages(const std::vector<std::vector<double>> &images,
                        const Profile &p) {
    const u32 num_images = static_cast<u32>(images.size());
    const u32 degree = 1U << p.log_degree;
    // Images per message: degree/2 in both rings, NOT ringDim(p) -- see
    // slotsPerMsg() in mlp_params.hpp.
    const u32 slots = slotsPerMsg(p);
    const u32 blocks = numBlocks(p, num_images);

    Matrix<Real> x(p.log_degree, IN_P, blocks * degree);
    for (u32 img = 0; img < num_images; ++img) {
        const auto &row = images[img];
        if (row.size() != INPUT_DIM)
            throw std::runtime_error(
                "packImages expects INPUT_DIM values per image");
        const u32 col = (img / slots) * degree + (img % slots);
        for (u32 r = 0; r < INPUT_DIM; ++r)
            x.at(r, col) = static_cast<Real>(row[r]);
        x.at(INPUT_DIM, col) = 1.0; // ones-row: carries b1 into fc1
    }
    return x;
}

// Bridges a Matrix<Real> through one message per stored chunk into a plaintext
// matrix. Only the first `slots` reals of every chunk are meaningful under CI,
// which is why the client-side Matrix is degree-wide rather than slots-wide.
Ptr<IPtMatrix> encodeMatrix(const EnDecoder &encoder, const Matrix<Real> &m,
                            u32 level) {
    if (m.device() != Device::CPU)
        throw std::runtime_error("encodeMatrix: the matrix must be on the CPU");

    const u32 rows = m.rows();
    const u32 log_slots = m.logDegree() - 1;
    const u32 slots = 1U << log_slots;
    const u32 degree = slots * 2;
    const u32 blocks = (m.cols() + degree - 1) / degree;

    std::vector<Message> msgs;
    msgs.reserve(static_cast<size_t>(rows) * blocks);
    for (u32 stored = 0; stored < rows * blocks; ++stored) {
        Message msg(log_slots);
        const Real *src = m[stored];
        Complex *dst = msg.data();
        for (u32 i = 0; i < slots; ++i)
            dst[i] = Complex(src[i], 0.0);
        msg.to(mlp::targetDevice());
        msgs.push_back(std::move(msg));
    }

    auto bridge = IPlaintext::make(PtxtType::BATCH);
    encoder.encode(msgs, *bridge, level);
    auto out = IPtMatrix::make();
    out->moveFrom(*bridge, rows, m.cols());
    return out;
}

//===========================================================================
// Model
//===========================================================================

namespace {
void writeFlat(std::ofstream &f, const std::vector<double> &v) {
    f.write(reinterpret_cast<const char *>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(double)));
}
void readFlat(std::ifstream &f, std::vector<double> &v) {
    f.read(reinterpret_cast<char *>(v.data()),
           static_cast<std::streamsize>(v.size() * sizeof(double)));
}
} // namespace

void writeRawModel(const RawModel &m, const std::string &path) {
    if (m.W1.size() != HIDDEN_DIM || m.b1.size() != HIDDEN_DIM ||
        m.W2.size() != LABEL_DIM || m.b2.size() != LABEL_DIM)
        throw std::runtime_error("writeRawModel: shape mismatch");
    std::ofstream f(path, std::ios::binary);
    if (!f.good())
        throw std::runtime_error("cannot write " + path);
    for (const auto &row : m.W1) writeFlat(f, row);
    writeFlat(f, m.b1);
    for (const auto &row : m.W2) writeFlat(f, row);
    writeFlat(f, m.b2);
    if (!f)
        throw std::runtime_error("short write to " + path);
}

RawModel readRawModel(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good())
        throw std::runtime_error("cannot open " + path +
                                 " (run server_preprocess_model first)");
    RawModel m;
    m.W1.assign(HIDDEN_DIM, std::vector<double>(INPUT_DIM));
    for (auto &row : m.W1) readFlat(f, row);
    m.b1.resize(HIDDEN_DIM);
    readFlat(f, m.b1);
    m.W2.assign(LABEL_DIM, std::vector<double>(HIDDEN_DIM));
    for (auto &row : m.W2) readFlat(f, row);
    m.b2.resize(LABEL_DIM);
    readFlat(f, m.b2);
    if (!f)
        throw std::runtime_error("truncated model cache " + path);
    return m;
}


Model buildModel(const std::vector<std::vector<double>> &W1,
                 const std::vector<double> &b1,
                 const std::vector<std::vector<double>> &W2,
                 const std::vector<double> &b2, const EnDecoder &coeff,
                 const Levels &levels, u32 num_images, const Profile &p) {
    if (W1.size() != HIDDEN_DIM || W1[0].size() != INPUT_DIM ||
        b1.size() != HIDDEN_DIM)
        throw std::runtime_error("buildModel: W1/b1 shape mismatch");
    if (W2.size() != LABEL_DIM || W2[0].size() != HIDDEN_DIM ||
        b2.size() != LABEL_DIM)
        throw std::runtime_error("buildModel: W2/b2 shape mismatch");

    const u32 top = levels.top();
    Model model;

    // U1 [HIDDEN_DIM x IN_P] = [W1 | b1] @ fc1's input level. b1 rides as an
    // extra weight column against the packed input's ones-row, so fc1 computes
    // W1*X + b1 directly -- no separate plaintext add for fc1's bias.
    Matrix<Real> u1(p.log_degree, HIDDEN_DIM, IN_P);
    for (u32 r = 0; r < HIDDEN_DIM; ++r) {
        for (u32 c = 0; c < INPUT_DIM; ++c)
            u1.at(r, c) = static_cast<Real>(W1[r][c]);
        u1.at(r, INPUT_DIM) = static_cast<Real>(b1[r]);
    }
    model.u1 = encodeMatrix(coeff, u1, top - FC1_IN_DROP);

    // U2 [LABEL_DIM x HIDDEN_DIM] = W2 @ fc2's input level (x^2's output).
    Matrix<Real> u2(p.log_degree, LABEL_DIM, HIDDEN_DIM);
    for (u32 r = 0; r < LABEL_DIM; ++r)
        for (u32 c = 0; c < HIDDEN_DIM; ++c)
            u2.at(r, c) = static_cast<Real>(W2[r][c]);
    model.u2 = encodeMatrix(coeff, u2, top - FC2_IN_DROP);

    // b2 [LABEL_DIM x n_cols] @ fc2's OUTPUT level, added after fc2 rather
    // than folded the way b1 was: that would need the ones-row carried past
    // x^2, taking fc1's output (and with it x^2's batch and fc2's contraction)
    // from HIDDEN_DIM to HIDDEN_DIM+1 rows for one bias vector.
    // b2[r] is one constant over its row's slots, so its inverse DFT is a
    // single nonzero coefficient at the head of each block -- which a
    // zero-filled Matrix already provides, so only that entry is written.
    const u32 degree = 1U << p.log_degree;
    const u32 blocks = numBlocks(p, num_images);
    Matrix<Real> b2m(p.log_degree, LABEL_DIM, blocks * degree);
    for (u32 r = 0; r < LABEL_DIM; ++r)
        for (u32 blk = 0; blk < blocks; ++blk)
            b2m.at(r, blk * degree) = static_cast<Real>(b2[r]);
    auto b2_bridge = encodeMatrix(coeff, b2m, top - B2_DROP);
    // The addition itself is a plain ciphertext-plaintext add, so b2 is kept
    // as a plaintext rather than a plaintext matrix (see inference()). Only
    // the polynomial and its encoding carry over; the shape label is dropped
    // and nothing downstream reads it.
    model.b2 = IPlaintext::make(PtxtType::BATCH);
    b2_bridge->moveTo(*model.b2);

    return model;
}

//===========================================================================
// Client-side key material
//===========================================================================

Ptr<ISwKey> genRelinKey(const ISecretKey &sk, const Levels &levels,
                        const Profile &p) {
    const u32 top = levels.top();
    const bool conj_inv = (p.ntt_alg == NTTAlgorithm::CYC_FOR_CI);

    paramsUtils::SwKeyGenParamsBuilder sb;
    sb.setNoiseDistribution(DiscreteGaussian(NOISE_STDDEV));
    // The key lives in the ring sk itself reports.
    sb.setRing(sk.logDegree(), p.poly_type);
    // The budget is a separate bound: how many bits of raised modulus the
    // key's entropy supports. PCMM's key is sampled directly at the profile's
    // log_degree (no lifting), and CI samples only half its coefficients, so
    // the bound is keyed by logRlweDim(p), not by the degree. Under NORMAL the
    // two are the same number.
    sb.setModUpPrimes(mlp::maxBits128(logRlweDim(p)), SWK_MARGIN);

    return SwKeyGenerator(sb.build(levels.mods[top - SQUARE_OUT_DROP], conj_inv))
        .genRelinKey(sk);
}

//===========================================================================
// Coefficient/slot relabeling
//===========================================================================

void setDFT(ICtMatrix &ct, bool dft, u32 num_images, const Profile &p) {
    const u32 rows = ct.rows();
    auto bridge = ICiphertext::make(EncType::BatchRLWE);
    ct.moveTo(*bridge);
    HomEvalFlexible{}.setDFT(*bridge, dft);
    // Coeff- and slot-encoded columns are counted in different units under CI:
    // a coeff-encoded block stores ringDim() = degree/2 coefficients, its
    // slot-encoded relabeling the full degree. Restating the true count here
    // keeps ct's own reported .cols() meaningful to the next caller. Under
    // NORMAL the two labels coincide.
    const u32 blocks = numBlocks(p, num_images);
    const u32 cols = blocks * (dft ? (1U << p.log_degree) : ringDim(p));
    ct.moveFrom(*bridge, rows, cols);
}

//===========================================================================
// Ciphertext matrix <-> file
//===========================================================================

void saveCtMatrix(const std::string &path, ICtMatrix &ct) {
    auto bridge = ICiphertext::make(EncType::BatchRLWE);
    ct.moveTo(*bridge);
    serial::save(path, *bridge);
}

Ptr<ICtMatrix> loadCtMatrix(const std::string &path, u32 rows, u32 cols,
                            Device dev) {
    auto bridge = serial::loadAsPtr<ICiphertext>(path, dev);
    auto ct = ICtMatrix::make();
    ct->moveFrom(*bridge, rows, cols);
    return ct;
}

//===========================================================================
// Online inference
//===========================================================================

void inference(const Model &model, const ISwKey &relin_key,
               const ICtMatrix &cx, ICtMatrix &cy, const Levels &levels,
               u32 num_images, const Profile &p) {
    const u32 n_cols = numCols(p, num_images);
    HomEvalMatrix mm{HomEvalParams{levels}};
    HomEval he{HomEvalParams{levels}};

    // ----- fc1: U1 * X -> W1 X + b1 -----
    auto ch1 = ICtMatrix::make();
    mm.pcmm(*model.u1, cx, *ch1);
    mm.rescale(*ch1, *ch1, /*r_ntt=*/true);

    // ----- x^2: tensor -> rescale -> relin. Rescaling first puts the relin
    // key one level lower than the usual tensor->relin->rescale order (see
    // SQUARE_OUT_DROP in mlp_params.hpp); the backward NTT then returns the
    // result to the coefficient domain fc2 consumes. -----
    auto hb = ICiphertext::make(EncType::BatchRLWE);
    ch1->moveTo(*hb);
    auto sq = ICiphertext::make(EncType::BatchRLWE);
    he.tensor(*hb, *hb, *sq);
    he.rescale(*sq, *sq);
    he.relin(*sq, relin_key);
    he.backwardNTT(*sq, *sq);
    auto ch2 = ICtMatrix::make();
    ch2->moveFrom(*sq, HIDDEN_DIM, n_cols);

    // ----- fc2: U2 * h^2 -> logits -----
    mm.pcmm(*model.u2, *ch2, cy);
    mm.rescale(cy, cy);

    // ----- +b2: plain ciphertext-plaintext add, no level cost. -----
    auto cyb = ICiphertext::make(EncType::BatchRLWE);
    cy.moveTo(*cyb);
    he.add(*cyb, *model.b2, *cyb);
    cy.moveFrom(*cyb, LABEL_DIM, n_cols);
}

//===========================================================================
// Client-side output unpacking
//===========================================================================

std::vector<std::vector<double>> unpackLogits(const Matrix<Real> &yc,
                                              u32 num_images,
                                              const Profile &p) {
    const u32 degree = 1U << p.log_degree;
    const u32 slots = slotsPerMsg(p); // must mirror packImages exactly
    std::vector<std::vector<double>> logits(num_images,
                                            std::vector<double>(LABEL_DIM));
    for (u32 img = 0; img < num_images; ++img) {
        const u32 col = (img / slots) * degree + (img % slots);
        for (u32 r = 0; r < LABEL_DIM; ++r)
            logits[img][r] = static_cast<double>(yc.at(r, col));
    }
    return logits;
}

} // namespace mlp::pcmm

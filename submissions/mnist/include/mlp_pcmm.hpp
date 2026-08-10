// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// mlp_pcmm.hpp - the PCMM (GEMM-based) inference scheme, used for medium and
// large instances (see mlp::usePcmm). Written independently against HEaaN2's
// public API; no HEaaN2 source is vendored. HEaaN2's own mlp/PCMM benchmark
// was read as a design reference for the algorithm -- the pcmm circuit for
// this exact model, the section-6.3 coeff/slot relabeling trick, and the
// bias-folded-into-a-weight-column layout are its ideas, reimplemented here
// for the submission's own stage contract (separate client/server keys,
// serialization across io/, no plaintext ever touching the server).
//
// Packing (opposite of the HS scheme in mlp_pipeline.hpp): feature = matrix
// row (pcmm's contraction dimension), image = column/slot. One message holds
// ringDim() images; a larger batch splits into further "blocks" inside a
// single ICtMatrix, transparent to callers here.
//
// Bias: b1 rides along as an extra column of U1 against an appended
// "ones" row in the packed input, so fc1's pcmm computes W1*X + b1 directly.
// b2 is added afterwards as a plaintext, once fc2's output no longer carries
// that ones-row (folding it the same way would carry one more row through
// x^2 and fc2, which is not worth it for a single bias vector).
//
// Encoding: pcmm needs its weights coefficient-encoded (its scalars are used
// literally) but its data slot-encoded (so the library's own iDFT applies).
// Both are, bit-for-bit, the same array -- HEaaN2 exposes no direct way to
// build one CtMatrix in "coefficient" form from slot-encoded ciphertext data,
// so encryption goes through the slot encoder and the DFT flag is then
// flipped in place via HomEvalFlexible::setDFT, bridged through a BatchRLWE
// ICiphertext (ICtMatrix itself has no setDFT overload). See setDFT() below.

#ifndef MLP_PCMM_HPP_
#define MLP_PCMM_HPP_

#include "mlp_params.hpp"
#include "params.h"

#include <string>
#include <vector>

namespace mlp::pcmm {

// Filenames inside ctxtupdir()/ctxtdowndir(). PCMM packs the whole batch into
// one ICtMatrix, so (unlike the HS scheme's per-128-image files) there is
// exactly one file each way, regardless of batch size.
constexpr const char *INPUT_CTMATRIX_FILE = "cipher_input_matrix.bin";
constexpr const char *RESULT_CTMATRIX_FILE = "cipher_result_matrix.bin";

//===========================================================================
// Parameter reconstruction. Every PCMM stage calls these and must get the
// same answer -- see the header comment of mlp_params.hpp.
//===========================================================================

heaan::Levels buildLevels();

// Weights, bias and (post relabel) decode all use the coefficient encoder;
// only the input's initial encode uses the slot encoder. See setDFT().
heaan::EnDecoder makeCoeffEncoder(const heaan::Levels &levels);
heaan::EnDecoder makeSlotEncoder(const heaan::Levels &levels);

//===========================================================================
// Packing
//===========================================================================

// Pack a batch of already-cropped, already-normalized images (INPUT_DIM
// values each, row-major, in the format client_preprocess_input writes) into
// the [IN_P x n_cols] matrix pcmm consumes: row r holds feature r across
// every image, image i lands in column (i / ringDim()) * (1 << LOG_DEGREE) +
// (i % ringDim()), and the trailing IN_P-1 row is a constant 1 (the bias
// carrier). Width uses the client-side stored-coefficient count
// (1 << LOG_DEGREE) per block, not numCols() -- see encodeMatrix().
heaan::Matrix<heaan::Real>
packImages(const std::vector<std::vector<double>> &images);

// Encode a Matrix<Real> into an IPtMatrix at the given level. Used for the
// input (through the slot encoder) and for weights/bias (through the
// coefficient encoder) alike -- the message contents are identical either
// way, only the encoder's own params decide the encoding.
heaan::Ptr<heaan::IPtMatrix> encodeMatrix(const heaan::EnDecoder &encoder,
                                          const heaan::Matrix<heaan::Real> &m,
                                          u32 level);

//===========================================================================
// Model (server side; built from cleartext weights alone, no secret key).
//===========================================================================

// Raw (unpadded) CSV-shaped weights, cached by server_preprocess_model and
// read back by server_encrypted_compute -- the plaintext-only half of
// preprocessing stage 3 can do without knowing the instance size (see the
// header comment of server_preprocess_model.cpp). Shapes are the compile-time
// constants above, so the file stores no dimensions of its own.
struct RawModel {
    std::vector<std::vector<double>> W1; // HIDDEN_DIM x INPUT_DIM
    std::vector<double> b1;              // HIDDEN_DIM
    std::vector<std::vector<double>> W2; // LABEL_DIM x HIDDEN_DIM
    std::vector<double> b2;              // LABEL_DIM
};
void writeRawModel(const RawModel &m, const std::string &path);
RawModel readRawModel(const std::string &path);


struct Model {
    heaan::Ptr<heaan::IPtMatrix> u1;  // [HIDDEN_DIM x IN_P] = [W1 | b1] @ top
    heaan::Ptr<heaan::IPtMatrix> u2;  // [LABEL_DIM x HIDDEN_DIM] = W2
    heaan::Ptr<heaan::IPlaintext> b2; // delta-encoded, added after fc2
};

// W1 (HIDDEN_DIM x INPUT_DIM), b1 (HIDDEN_DIM), W2 (LABEL_DIM x HIDDEN_DIM),
// b2 (LABEL_DIM) -- dense, unpadded, straight out of the CSV shapes.
Model buildModel(const std::vector<std::vector<double>> &W1,
                 const std::vector<double> &b1,
                 const std::vector<std::vector<double>> &W2,
                 const std::vector<double> &b2, const heaan::EnDecoder &coeff,
                 const heaan::Levels &levels, u32 num_images);

//===========================================================================
// Client-side key material: only a relinearization key -- pcmm needs no
// rotation keys at all, unlike the HS scheme.
//===========================================================================

heaan::Ptr<heaan::ISwKey> genRelinKey(const heaan::ISecretKey &sk,
                                      const heaan::Levels &levels,
                                      double swk_margin);

//===========================================================================
// The section-6.3 relabeling bridge: flips only the DFT metadata flag on an
// ICtMatrix's data, leaving the coefficients untouched. ICtMatrix has no
// setDFT of its own, so this goes through a BatchRLWE ICiphertext bridge.
// `num_images` is needed to recompute the column count in the target
// labeling (coeff- and slot-encoded columns are counted in different units
// under CI; see mlp_pcmm.cpp).
//===========================================================================

void setDFT(heaan::ICtMatrix &ct, bool dft, u32 num_images);

//===========================================================================
// ICtMatrix <-> file. Bridges through the same BatchRLWE ICiphertext that
// setDFT does (heaan::serial has no ICtMatrix overload), so an ICtMatrix
// crosses io/ exactly like every other object in this submission. Shape is
// not stored in the file: both ends recompute rows/cols from num_images and
// the constants in mlp_params.hpp, so nothing can drift between them.
//===========================================================================

void saveCtMatrix(const std::string &path, heaan::ICtMatrix &ct);
heaan::Ptr<heaan::ICtMatrix> loadCtMatrix(const std::string &path, u32 rows,
                                          u32 cols, heaan::Device dev);

//===========================================================================
// Online inference: fc1 -> x^2 -> fc2 -> +b2. Ciphertext operations only;
// `cy` receives the result. `relin_key` is the client-published key from
// genRelinKey; the server holds no secret key.
//===========================================================================

void inference(const Model &model, const heaan::ISwKey &relin_key,
               const heaan::ICtMatrix &cx, heaan::ICtMatrix &cy,
               const heaan::Levels &levels, u32 num_images);

//===========================================================================
// Client-side output unpacking: the reverse of packImages. `yc` is the
// decoded [LABEL_DIM x n_cols] matrix; returns one 10-entry logit vector per
// image, in image order -- the same shape client_postprocess's argmax
// already expects from the HS path.
//===========================================================================

std::vector<std::vector<double>> unpackLogits(const heaan::Matrix<heaan::Real> &yc,
                                              u32 num_images);

} // namespace mlp::pcmm

#endif // MLP_PCMM_HPP_

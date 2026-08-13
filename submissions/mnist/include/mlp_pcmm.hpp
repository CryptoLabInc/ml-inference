// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// mlp_pcmm.hpp - the PCMM (GEMM-based) inference scheme, used for the small,
// medium and large instances (see mlp::usePcmm). Written against the public
// HEaaN2 API only.
//
// Packing (opposite of the HS scheme in mlp_pipeline.hpp): feature = matrix
// row (the GEMM contraction dimension), image = column/slot. One message holds
// slotsPerMsg() images; a larger batch splits into further "blocks" inside a
// single ciphertext matrix, transparent to callers here. Note slotsPerMsg() is
// not ringDim() -- see their definitions in mlp_params.hpp.
//
// Bias: b1 rides along as an extra column of U1 against an appended "ones" row
// in the packed input, so fc1 computes W1*X + b1 directly. b2 is added
// afterwards as a plaintext, once fc2's output no longer carries that ones-row
// (folding it the same way would carry one more row through x^2 and fc2, which
// is not worth it for a single bias vector).
//
// Encoding: the GEMM consumes coefficient-encoded operands, while encryption
// goes through the slot encoder. The two carry the same array, so the input is
// encrypted slot-encoded and then relabelled as coefficient-encoded in place,
// without touching the data -- see setDFT() below.

#ifndef MLP_PCMM_HPP_
#define MLP_PCMM_HPP_

#include "mlp_params.hpp"
#include "params.h"

#include <string>
#include <vector>

namespace mlp::pcmm {

// Filenames inside ctxtupdir()/ctxtdowndir(). PCMM packs the whole batch into
// one ciphertext matrix, so (unlike the HS scheme's per-ciphertext files)
// there is exactly one file each way, regardless of batch size.
constexpr const char *INPUT_CTMATRIX_FILE = "cipher_input_matrix.bin";
constexpr const char *RESULT_CTMATRIX_FILE = "cipher_result_matrix.bin";

//===========================================================================
// Parameter reconstruction. Every PCMM stage calls these and must get the
// same answer -- see the header comment of mlp_params.hpp.
//===========================================================================

// Each takes the Profile its stage resolved from the instance size
// (mlp::pcmm::profile(size)); large is tuned separately from small/medium. A
// stage passing a different profile than its peers produces objects the others
// cannot read.
heaan::Levels buildLevels(const Profile &p);

// Weights, bias and (post relabel) decode use the coefficient encoder; only
// the input's initial encode uses the slot encoder. See setDFT().
heaan::EnDecoder makeCoeffEncoder(const heaan::Levels &levels,
                                  const Profile &p);
heaan::EnDecoder makeSlotEncoder(const heaan::Levels &levels,
                                 const Profile &p);

//===========================================================================
// Packing
//===========================================================================

// Pack a batch of already-cropped, already-normalized images (INPUT_DIM values
// each, row-major, in the format client_preprocess_input writes) into the
// [IN_P x n_cols] matrix the GEMM consumes: row r holds feature r across every
// image, image i lands in column (i / slotsPerMsg(p)) * (1 << p.log_degree) +
// (i % slotsPerMsg(p)), and the trailing IN_P-1 row is a constant 1 (the bias
// carrier). Width uses the client-side per-block width (1 << p.log_degree),
// not numCols() -- see encodeMatrix().
heaan::Matrix<heaan::Real>
packImages(const std::vector<std::vector<double>> &images, const Profile &p);

// Encode a Matrix<Real> into a plaintext matrix at the given level. Used for
// the input (through the slot encoder) and for weights/bias (through the
// coefficient encoder) alike -- the message contents are identical either way,
// only the encoder decides the encoding.
heaan::Ptr<heaan::IPtMatrix> encodeMatrix(const heaan::EnDecoder &encoder,
                                          const heaan::Matrix<heaan::Real> &m,
                                          u32 level);

//===========================================================================
// Model (server side; built from cleartext weights alone, no secret key).
//===========================================================================

// Raw (unpadded) CSV-shaped weights, cached by server_preprocess_model and
// read back by server_encrypted_compute -- the plaintext-only half of stage 3,
// which does not know the instance size. Shapes are the compile-time constants
// in mlp_params.hpp, so the file stores no dimensions of its own.
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
                 const heaan::Levels &levels, u32 num_images,
                 const Profile &p);

//===========================================================================
// Client-side key material: only a relinearization key -- PCMM needs no
// rotation keys at all, unlike the HS scheme.
//===========================================================================

heaan::Ptr<heaan::ISwKey> genRelinKey(const heaan::ISecretKey &sk,
                                      const heaan::Levels &levels,
                                      const Profile &p);

//===========================================================================
// Coefficient/slot relabeling: flips only the encoding label on a ciphertext
// matrix, leaving the data untouched. `num_images` is needed to recompute the
// column count in the target labeling (coeff- and slot-encoded columns are
// counted in different units under CI, and coincide under NORMAL).
//===========================================================================

void setDFT(heaan::ICtMatrix &ct, bool dft, u32 num_images, const Profile &p);

//===========================================================================
// Ciphertext matrix <-> file. Shape is not stored in the file: both ends
// recompute rows/cols from num_images and the constants in mlp_params.hpp, so
// nothing can drift between them.
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
               const heaan::Levels &levels, u32 num_images, const Profile &p);

//===========================================================================
// Client-side output unpacking: the reverse of packImages. `yc` is the decoded
// [LABEL_DIM x n_cols] matrix; returns one 10-entry logit vector per image, in
// image order -- the same shape client_postprocess's argmax expects from the
// HS path.
//===========================================================================

std::vector<std::vector<double>> unpackLogits(const heaan::Matrix<heaan::Real> &yc,
                                              u32 num_images, const Profile &p);

} // namespace mlp::pcmm

#endif // MLP_PCMM_HPP_

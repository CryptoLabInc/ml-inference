// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// mlp_pipeline.hpp - helpers shared by the seven stage executables: parameter
// reconstruction, text/CSV I/O against the harness file formats, slot packing,
// and the Halevi-Shoup homomorphic layer.

#ifndef MLP_PIPELINE_HPP_
#define MLP_PIPELINE_HPP_

#include "mlp_params.hpp"
#include "params.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlp {

//===========================================================================
// Parameter reconstruction. Every stage calls this and must get the same
// answer; see the header comment of mlp_params.hpp.
//===========================================================================

heaan::Levels buildLevels();
heaan::EnDecoder makeEncoder(const heaan::Levels &levels);

// The matrix-vector evaluation parameters for one layer. Shape-only: no
// weights, so client_key_generation can call it to learn which rotation keys
// to make.
heaan::MatrixVectorEvalParams makeMvParams(const LayerGeom &geom,
                                           const heaan::Levels &levels,
                                           u32 in_level,
                                           heaan::Real128 in_scale);

// The switching-key parameters for one level. Carries no key material -- it is
// the *shape* of a switching key, including the gadget decomposition that
// diagonal encoding needs. client_key_generation builds its rotation keys from
// this and server_preprocess_model encodes diagonals against the same object,
// so the two cannot drift: a mismatch makes the encoded diagonals unusable
// with the keys.
heaan::SwKeyGenParams makeSwkParams(const heaan::Levels &levels, u32 level);

//===========================================================================
// Model
//===========================================================================

// A p x q row-major, zero-padded weight block plus its bias.
struct LayerWeights {
    std::vector<double> W; // p * q, row-major, zero-padded
    std::vector<double> b; // n_out
};

std::vector<std::vector<double>> readMatrixCsv(const std::string &path);
std::vector<double> readFlatCsv(const std::string &path);

LayerWeights padWeights(const LayerGeom &geom,
                        const std::vector<std::vector<double>> &dense,
                        const std::vector<double> &bias);

// Compact binary form written by server_preprocess_model and read back by
// server_encrypted_compute: native-endian doubles, W then b, no header.
void writeLayerWeights(const LayerWeights &lw, const std::string &path);
LayerWeights readLayerWeights(const LayerGeom &geom, const std::string &path);

//===========================================================================
// Packing
//===========================================================================

// Slot index holding coordinate `coord` of image `img` within a ciphertext.
inline size_t slotOf(u32 coord, u32 img) {
    return static_cast<size_t>(coord) * IMAGES_PER_CTXT + img;
}

// Pack up to IMAGES_PER_CTXT already-cropped, already-normalized images
// (INPUT_DIM values each, row-major) into one Message. Unused slots are zero.
heaan::Message packImages(const std::vector<std::vector<double>> &images,
                          size_t first, size_t count);

// The p diagonals of the layer matrix, in the layout the matrix-vector
// evaluation wants:
//   res[i] = sum_d diags[d][i] * op[(i + d) % slots]
// with d running over multiples of IMAGES_PER_CTXT. All p are kept, including
// any that are all-zero -- see bsIndices/gsIndices in mlp_params.hpp.
std::map<i32, heaan::Message> buildDiags(const LayerGeom &geom,
                                         const std::vector<double> &W);

// Bias across the slots: coordinate c carries b[c % p] on a real output row,
// 0 on a padding row.
heaan::Message buildBiasMessage(const LayerGeom &geom,
                                const std::vector<double> &b);

//===========================================================================
// One precomputed homomorphic layer.
//===========================================================================

struct Layer {
    bool activate = false;
    // The matrix-vector evaluator holds a reference to the rotation keys and
    // needs them to outlive it, so both sit behind a pointer and moving a
    // Layer moves only the pointers.
    std::unique_ptr<heaan::RotKeyPtrs> rot_keys;
    std::unique_ptr<heaan::MatrixVectorEval> matvec;
    i32 fold_stride = 0;
    u32 fold_factor = 1;
    heaan::RotKeyPtrs fold_keys;
    std::vector<i32> fold_steps;
    heaan::Ptr<heaan::IPlaintext> bias;
    heaan::KeyPtr relin_key; // only when activate
    heaan::PolyMod mod_to;
    heaan::Real128 scale_to;
};

// Encode a layer's diagonals, WITHOUT any key material. This is the expensive
// half of building a layer, and it depends only on the weights, the layer
// geometry and the level -- all fixed -- so server_preprocess_model runs it and
// serializes the result. Encoding happens on `dev` directly, the device the
// evaluation later runs on.
heaan::MatrixVectorEvalEncoded encodeDiags(const LayerGeom &geom,
                                           const std::vector<double> &W,
                                           const heaan::Levels &levels,
                                           u32 in_level, heaan::Device dev);

// Assemble a layer on the server from loaded keys plus the pre-encoded
// diagonals. `rot_keys` is consumed; `encoded` is shared, not copied, so it may
// be destroyed afterwards. Only the bias is encoded here -- one message per
// layer, negligible next to the diagonals.
// `fold_keys` is consumed too, and is required exactly when the layer folds
// (q/p > 1). Pass an empty RotKeyPtrs for a layer that does not fold.
Layer makeLayer(const LayerGeom &geom, const std::vector<double> &bias,
                const heaan::MatrixVectorEvalEncoded &encoded,
                const heaan::Levels &levels, u32 in_level, u32 out_level,
                const heaan::EnDecoder &encoder,
                std::unique_ptr<heaan::RotKeyPtrs> rot_keys,
                heaan::KeyPtr relin_key, heaan::Device dev,
                heaan::RotKeyPtrs fold_keys = {});

// Online evaluation: ciphertext operations only. `ct` is replaced.
void homLayer(heaan::Ptr<heaan::ICiphertext> &ct, const Layer &lyr,
              const heaan::HomEval &eval, const heaan::HomEvalFlexible &flex);

//===========================================================================
// Server-side model cache, and the instance marker.
//
// server_preprocess_model (stage 3) is invoked with NO arguments, so it cannot
// tell which scheme the run will use -- and encoding the HS diagonals is
// several seconds of pure waste on a PCMM instance. It cannot infer the size
// from io/ either: the harness only clears the *current* instance's directory,
// so stale ones from earlier runs sit alongside it.
//
// So the stage that does know writes it down: client_key_generation (stage
// 2.2) receives <size> and always runs first. This is our own pipeline passing
// a public, already-known quantity between our own stages; nothing about the
// measurement changes, as both schemes still do all of their own work. A
// missing or unreadable marker is not an error -- stage 3 then prepares both
// schemes.
//===========================================================================

constexpr const char *MODEL_CACHE_DIR = "submissions/mnist/build/model_cache";

void writeInstanceMarker(InstanceSize size);
std::optional<InstanceSize> readInstanceMarker();

//===========================================================================
// Harness file formats
//===========================================================================

// One sample per line, `dim` whitespace-separated floats. Used for the
// harness's test_pixels.txt (784) and our preprocessed input (484).
std::vector<std::vector<double>> readSamples(const std::string &path, u32 dim);
void writeSamples(const std::vector<std::vector<double>> &rows,
                  const std::string &path);

// argv parsing shared by every stage: "<exe> <instance-size>".
InstanceSize parseInstanceSize(int argc, char *argv[]);

// Device the stages run on: GPU when built with CUDA, else CPU.
heaan::Device targetDevice();

} // namespace mlp

#endif // MLP_PIPELINE_HPP_

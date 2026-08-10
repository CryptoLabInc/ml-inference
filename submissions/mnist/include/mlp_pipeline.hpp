// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// mlp_pipeline.hpp - helpers shared by the seven stage executables: parameter
// reconstruction, text/CSV I/O against the harness file formats, slot packing,
// and the homomorphic layer.

#ifndef MLP_PIPELINE_HPP_
#define MLP_PIPELINE_HPP_

#include "mlp_params.hpp"
#include "params.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlp {

//===========================================================================
// Parameter reconstruction. Every stage calls this and must get the same
// answer; see the header comment of mlp_params.hpp.
//===========================================================================

heaan::Levels buildLevels();
heaan::EnDecoder makeEncoder(const heaan::Levels &levels);

// The MatrixVectorEval parameters for one layer. Shape-only: no weights, so
// client_key_generation can call it to learn which rotation keys to make.
heaan::MatrixVectorEvalParams makeMvParams(const LayerGeom &geom,
                                           const heaan::Levels &levels,
                                           u32 in_level,
                                           heaan::Real128 in_scale);

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

// The p diagonals of the layer matrix, in the layout MatrixVectorEval wants:
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
    // MatrixVectorEval holds a reference to the rotation keys and needs them
    // to outlive it, so both sit behind a pointer and moving a Layer moves
    // only the pointers.
    std::unique_ptr<heaan::RotKeyPtrs> rot_keys;
    std::unique_ptr<heaan::MatrixVectorEval> matvec;
    bool keyless_fold = false;
    i32 fold_stride = 0;
    u32 fold_factor = 1;
    heaan::RotKeyPtrs fold_keys; // only when the fold must be keyed
    std::vector<i32> fold_steps;
    heaan::Ptr<heaan::IPlaintext> bias;
    heaan::KeyPtr relin_key; // only when activate
    heaan::PolyMod mod_to;
    heaan::Real128 scale_to;
};

// Assemble a layer on the server from loaded keys plus cleartext weights.
// `rot_keys` is consumed.
Layer makeLayer(const LayerGeom &geom, const LayerWeights &lw,
                const heaan::Levels &levels, u32 in_level, u32 out_level,
                const heaan::EnDecoder &encoder,
                std::unique_ptr<heaan::RotKeyPtrs> rot_keys,
                heaan::KeyPtr relin_key, heaan::Device dev);

// Online evaluation: ciphertext operations only. `ct` is replaced.
void homLayer(heaan::Ptr<heaan::ICiphertext> &ct, const Layer &lyr,
              const heaan::HomEval &eval, const heaan::HomEvalFlexible &flex);

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

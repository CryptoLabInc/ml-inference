// Copyright (c) 2026 CryptoLab, Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

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

heaan::Levels buildLevels();
heaan::EnDecoder makeEncoder(const heaan::Levels &levels);

heaan::MatrixVectorEvalParams makeMvParams(const LayerGeom &geom,
                                           const heaan::Levels &levels,
                                           u32 in_level,
                                           heaan::Real128 in_scale);

heaan::SwKeyGenParams makeSwkParams(const heaan::Levels &levels, u32 level);

struct LayerWeights {
    std::vector<double> W;
    std::vector<double> b;
};

std::vector<std::vector<double>> readMatrixCsv(const std::string &path);
std::vector<double> readFlatCsv(const std::string &path);

LayerWeights padWeights(const LayerGeom &geom,
                        const std::vector<std::vector<double>> &dense,
                        const std::vector<double> &bias);

inline size_t slotOf(u32 coord, u32 img) {
    return static_cast<size_t>(coord) * IMAGES_PER_CTXT + img;
}

heaan::Message packImages(const std::vector<std::vector<double>> &images,
                          size_t first, size_t count);

std::map<i32, heaan::Message> buildDiags(const LayerGeom &geom,
                                         const std::vector<double> &W);

heaan::Message buildBiasMessage(const LayerGeom &geom,
                                const std::vector<double> &b);

struct Layer {
    bool activate = false;
    std::unique_ptr<heaan::RotKeyPtrs> rot_keys;
    std::unique_ptr<heaan::MatrixVectorEval> matvec;
    i32 fold_stride = 0;
    u32 fold_factor = 1;
    heaan::RotKeyPtrs fold_keys;
    std::vector<i32> fold_steps;
    heaan::Ptr<heaan::IPlaintext> bias;
    heaan::KeyPtr relin_key;
    heaan::PolyMod mod_to;
    heaan::Real128 scale_to;
};

heaan::MatrixVectorEvalEncoded encodeDiags(const LayerGeom &geom,
                                           const std::vector<double> &W,
                                           const heaan::Levels &levels,
                                           u32 in_level, heaan::Device dev);

Layer makeLayer(const LayerGeom &geom, const std::vector<double> &bias,
                const heaan::MatrixVectorEvalEncoded &encoded,
                const heaan::Levels &levels, u32 in_level, u32 out_level,
                const heaan::EnDecoder &encoder,
                std::unique_ptr<heaan::RotKeyPtrs> rot_keys,
                heaan::KeyPtr relin_key, heaan::Device dev,
                heaan::RotKeyPtrs fold_keys = {});

void homLayer(heaan::Ptr<heaan::ICiphertext> &ct, const Layer &lyr,
              const heaan::HomEval &eval, const heaan::HomEvalFlexible &flex);

constexpr const char *MODEL_CACHE_DIR = "submissions/mnist/build/model_cache";

void writeInstanceMarker(InstanceSize size);
std::optional<InstanceSize> readInstanceMarker();

std::vector<std::vector<double>> readSamples(const std::string &path, u32 dim);
void writeSamples(const std::vector<std::vector<double>> &rows,
                  const std::string &path);

InstanceSize parseInstanceSize(int argc, char *argv[]);

heaan::Device targetDevice();

} // namespace mlp

#endif // MLP_PIPELINE_HPP_

// Copyright (c) 2026 CryptoLab, Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#ifndef MLP_PIPELINE_HPP_
#define MLP_PIPELINE_HPP_

#include "mlp_params.hpp"
#include "params.h"

#include <optional>
#include <string>
#include <vector>

namespace mlp {

std::vector<std::vector<double>> readMatrixCsv(const std::string &path);
std::vector<double> readFlatCsv(const std::string &path);

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

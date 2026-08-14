// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#ifndef MLP_PCMM_HPP_
#define MLP_PCMM_HPP_

#include "mlp_params.hpp"
#include "params.h"

#include <string>
#include <vector>

namespace mlp::pcmm {

constexpr const char *INPUT_CTMATRIX_FILE = "cipher_input_matrix.bin";
constexpr const char *RESULT_CTMATRIX_FILE = "cipher_result_matrix.bin";

heaan::Levels buildLevels(const Profile &prof);

heaan::EnDecoder makeCoeffEncoder(const heaan::Levels &levels,
                                  const Profile &prof);
heaan::EnDecoder makeSlotEncoder(const heaan::Levels &levels,
                                 const Profile &prof);

heaan::Matrix<heaan::Real>
packImages(const std::vector<std::vector<double>> &images, const Profile &prof);

heaan::Ptr<heaan::IPtMatrix> encodeMatrix(const heaan::EnDecoder &encoder,
                                          const heaan::Matrix<heaan::Real> &m,
                                          u32 level);

struct RawModel {
    std::vector<std::vector<double>> W1;
    std::vector<double> b1;
    std::vector<std::vector<double>> W2;
    std::vector<double> b2;
};

void writeRawModel(const RawModel &m, const std::string &path);
RawModel readRawModel(const std::string &path);

struct Model {
    heaan::Ptr<heaan::IPtMatrix> fc1_weights;
    heaan::Ptr<heaan::IPtMatrix> fc2_weights;
    heaan::Ptr<heaan::IPlaintext> fc2_bias;
};

Model buildModel(const std::vector<std::vector<double>> &W1,
                 const std::vector<double> &b1,
                 const std::vector<std::vector<double>> &W2,
                 const std::vector<double> &b2, const heaan::EnDecoder &coeff,
                 const heaan::Levels &levels, u32 num_images,
                 const Profile &prof);

heaan::Ptr<heaan::ISwKey> genRelinKey(const heaan::ISecretKey &sk,
                                      const heaan::Levels &levels,
                                      const Profile &prof);

void setDFT(heaan::ICtMatrix &ct, bool dft, u32 num_images,
            const Profile &prof);

void saveCtMatrix(const std::string &path, heaan::ICtMatrix &ct);
heaan::Ptr<heaan::ICtMatrix> loadCtMatrix(const std::string &path, u32 rows,
                                          u32 cols, heaan::Device dev);

void inference(const Model &model, const heaan::ISwKey &relin_key,
               const heaan::ICtMatrix &cx, heaan::ICtMatrix &cy,
               const heaan::Levels &levels, u32 num_images,
               const Profile &prof);

std::vector<std::vector<double>>
unpackLogits(const heaan::Matrix<heaan::Real> &yc, u32 num_images,
             const Profile &prof);

} // namespace mlp::pcmm

#endif // MLP_PCMM_HPP_

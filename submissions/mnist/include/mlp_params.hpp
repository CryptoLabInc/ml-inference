// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// mlp_params.hpp - the shared "context" for the MNIST MLP submission.
//
// Every stage rebuilds its parameters from the constants below, so THESE
// CONSTANTS ARE THE CONTEXT: if two stages disagree on any of them, keys and
// ciphertexts stop matching. Nothing here may be changed for one stage alone.
//
// The circuit is the BN-folded 2-layer MLP described in weights/manifest.txt:
//
//     fc1 (128x484, +b1)  ->  x^2  ->  fc2 (10x128, +b2)
//

#ifndef MLP_PARAMS_HPP_
#define MLP_PARAMS_HPP_

#include "HEaaN2/HEaaN2.hpp"
#include "params.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace mlp {

using heaan::i32;
using heaan::u32;

constexpr u32 IMG_DIM = 28;
constexpr u32 CROP = 3;
constexpr u32 CROP_DIM = IMG_DIM - 2 * CROP;
constexpr u32 MNIST_DIM = IMG_DIM * IMG_DIM;
constexpr u32 INPUT_DIM = CROP_DIM * CROP_DIM;
constexpr u32 HIDDEN_DIM = 128;
constexpr u32 LABEL_DIM = 10;

inline bool usePcmm(InstanceSize size) { return size >= InstanceSize::SMALL; }

constexpr double MNIST_MEAN = 0.1307;
constexpr double MNIST_STD = 0.3081;

//Encoding params
constexpr u32 LOG_DEGREE = 15;
constexpr u32 LOG_SLOTS = LOG_DEGREE - 1;
constexpr u32 SLOTS = 1U << LOG_SLOTS;
constexpr u32 COORD_DIM = 512;
constexpr u32 IMAGES_PER_CTXT = SLOTS / COORD_DIM;

static_assert(INPUT_DIM <= COORD_DIM, "cropped input must fit the coordinate dim");
static_assert(SLOTS % COORD_DIM == 0, "coordinate dim must divide the slot count");

constexpr heaan::NTTAlgorithm NTT_ALG = heaan::NTTAlgorithm::CYC_FOR_CI;
constexpr heaan::PolyType POLY_TYPE = heaan::PolyType::GRAFTED;

//Single inference 
constexpr u32 BASE_BITS = 30;
constexpr u32 RESCALE_BITS = 25;
constexpr u32 NUM_MULTS = 3;

constexpr u32 HW = 0; // 0 = uniform ternary
constexpr double SWK_MARGIN = 5.0;
constexpr double NOISE_STDDEV = 3.2;

inline u32 maxBits128(u32 log_degree) {
    switch (log_degree) {
    case 12: return 106;
    case 13: return 214;
    case 14: return 430;
    case 15: return 868;
    case 16: return 1747;
    case 17: return 3523;
    default:
        throw std::runtime_error("no 128-bit maxBits entry for N=2^" +
                                 std::to_string(log_degree) + " with hw=0");
    }
}

inline u32 swkMaxBits() {
    return maxBits128(LOG_DEGREE -
                      (NTT_ALG == heaan::NTTAlgorithm::CYC_FOR_CI ? 1 : 0));
}

struct LayerGeom {
    u32 p;
    u32 q;
    u32 n_out;
    u32 num_bs;
    bool activate;
};

constexpr LayerGeom FC1{128, 512, HIDDEN_DIM, 64, true};
constexpr LayerGeom FC2{128, 128, LABEL_DIM, 64, false};

constexpr u32 FC1_IN_DROP = 0, FC1_OUT_DROP = 1;
constexpr u32 FC2_IN_DROP = 2, FC2_OUT_DROP = 3;

inline std::vector<i32> bsIndices(const LayerGeom &g) {
    std::vector<i32> v;
    v.reserve(g.num_bs);
    for (u32 b = 0; b < g.num_bs; ++b)
        v.push_back(static_cast<i32>(IMAGES_PER_CTXT * b));
    return v;
}

inline std::vector<i32> gsIndices(const LayerGeom &g) {
    std::vector<i32> v;
    for (u32 k = 0; k < g.p; k += g.num_bs)
        v.push_back(static_cast<i32>(IMAGES_PER_CTXT * g.num_bs *
                                     (k / g.num_bs)));
    return v;
}

constexpr const char *ENC_KEY_FILE = "enc_key.bin";
constexpr const char *ROT_KEY_FC1_FILE = "rot_keys_fc1.bin";
constexpr const char *ROT_KEY_FC2_FILE = "rot_keys_fc2.bin";
constexpr const char *ROT_KEY_FOLD_FILE = "rot_keys_fold.bin";
constexpr const char *RELIN_KEY_FILE = "relin_key.bin";
constexpr const char *SECRET_KEY_FILE = "sk.bin";

inline size_t numCtxts(size_t n) {
    return (n + IMAGES_PER_CTXT - 1) / IMAGES_PER_CTXT;
}

// The PCMM circuit: one feature per matrix row, one image per column, the
// opposite of the Halevi-Shoup above. No rotations needed, hence
// no rotation keys.
namespace pcmm {

struct Profile {
    heaan::NTTAlgorithm ntt_alg;
    heaan::PolyType poly_type;
    u32 log_degree;
    u32 base_bits;
    u32 rescale_bits;
};

constexpr Profile MEDIUM_PROFILE{heaan::NTTAlgorithm::NORMAL,
                                 heaan::PolyType::SIMPLE32,
                                 /*log_degree=*/12,
                                 /*base_bits=*/34,
                                 /*rescale_bits=*/24};

constexpr Profile LARGE_PROFILE{heaan::NTTAlgorithm::CYC_FOR_CI,
                                heaan::PolyType::SIMPLE32,
                                /*log_degree=*/15,
                                /*base_bits=*/44,
                                /*rescale_bits=*/27};

constexpr Profile profile(InstanceSize size) {
    return size == InstanceSize::LARGE ? LARGE_PROFILE : MEDIUM_PROFILE;
}

// This is for the bias column added to fc1.
constexpr u32 IN_P = INPUT_DIM + 1;


// How much the levels drop from the top of the chain:
//   L3 fc1 PCMM -> L2 square -> L1 fc2 PCMM -> L0 +b2 (no drop), decrypt
// The square relinearizes at L1, which is the level genRelinKey builds for.
constexpr u32 FC1_IN_DROP = 0, FC1_OUT_DROP = 1;
constexpr u32 SQUARE_OUT_DROP = 2;
constexpr u32 FC2_IN_DROP = 2, FC2_OUT_DROP = 3;
constexpr u32 B2_DROP = 3;

//Conj Inv rings are half the size of the usual ring
constexpr u32 logRlweDim(const Profile &p) {
    return p.ntt_alg == heaan::NTTAlgorithm::CYC_FOR_CI ? p.log_degree - 1
                                                        : p.log_degree;
}

constexpr u32 ringDim(const Profile &p) { return 1U << logRlweDim(p); }

constexpr u32 slotsPerMsg(const Profile &p) { return 1U << (p.log_degree - 1); }

constexpr u32 numBlocks(const Profile &p, u32 num_images) {
    return (num_images + slotsPerMsg(p) - 1) / slotsPerMsg(p);
}

constexpr u32 numCols(const Profile &p, u32 num_images) {
    return numBlocks(p, num_images) * ringDim(p);
}

} // namespace pcmm

} // namespace mlp

#endif // MLP_PARAMS_HPP_

// Copyright (c) 2026 Crypto Lab Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================
//
// mlp_params.hpp - the shared "context" for the HEaaN2 MNIST MLP submission.
//
// HEaaN2 has no serializable CryptoContext object the way OpenFHE does. Every
// stage instead rebuilds its parameters deterministically from the constants
// below, so THESE CONSTANTS ARE THE CONTEXT: if two stages disagree on any of
// them, keys and ciphertexts silently stop matching. Nothing here may be
// changed for one stage alone.
//
// The circuit is the BN-folded 2-layer MNIST MLP described in
// weights/manifest.txt:
//
//     fc1 (128x484, +b1)  ->  x^2  ->  fc2 (10x128, +b2)
//
// x^2 is the activation the model was *trained* with, not an approximation of
// something else, so the homomorphic circuit evaluates the network exactly.

#ifndef MLP_PARAMS_HPP_
#define MLP_PARAMS_HPP_

#include "HEaaN2/HEaaN2.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlp {

using heaan::i32;
using heaan::u32;

//===========================================================================
// Model shape. See weights/manifest.txt.
//===========================================================================

constexpr u32 IMG_DIM = 28;                        // MNIST is 28x28
constexpr u32 CROP = 3;                            // pixels trimmed per side
constexpr u32 CROP_DIM = IMG_DIM - 2 * CROP;       // 22
constexpr u32 MNIST_DIM = IMG_DIM * IMG_DIM;       // 784, what the harness hands us
constexpr u32 INPUT_DIM = CROP_DIM * CROP_DIM;     // 484, what fc1 consumes
constexpr u32 HIDDEN_DIM = 128;
constexpr u32 LABEL_DIM = 10;

// Normalization the model was trained under (torchvision ToTensor + Normalize).
// The harness writes pixels already scaled to [0,1], so only the affine part
// applies. Identical to harness/mnist/test.py:60.
constexpr double MNIST_MEAN = 0.1307;
constexpr double MNIST_STD = 0.3081;

//===========================================================================
// Packing.
//
// One image occupies COORD_DIM padded coordinates; coordinate c of image i
// lives in slot c * IMAGES_PER_CTXT + i. 484 pads to 512 rather than 1024,
// which is what fits 128 images in a ciphertext instead of 64 -- see the crop
// discussion in README.md.
//===========================================================================

constexpr u32 LOG_DEGREE = 17;
constexpr u32 LOG_SLOTS = LOG_DEGREE - 1;   // CI pairs log_slots with N/2
constexpr u32 SLOTS = 1U << LOG_SLOTS;      // 65536
constexpr u32 COORD_DIM = 512;              // 484 zero-padded
constexpr u32 IMAGES_PER_CTXT = SLOTS / COORD_DIM; // 128

static_assert(INPUT_DIM <= COORD_DIM, "cropped input must fit the coordinate dim");
static_assert(SLOTS % COORD_DIM == 0, "coordinate dim must divide the slot count");

//===========================================================================
// Scheme parameters.
//
// Conjugate-invariant subring: the model and the inputs are real, so CI
// carries the same 2^LOG_SLOTS slots in half the polynomial arithmetic. The
// secret key, the encoder and every switching key must agree on it.
//
// Level schedule (one level per operation, no bootstrapping):
//   L3  encrypt        -> fc1 matvec -> L2
//   L2  +b1, x^2, rescale                -> L1
//   L1  fc2 matvec     -> L0, +b2, decrypt
//===========================================================================

constexpr heaan::NTTAlgorithm NTT_ALG = heaan::NTTAlgorithm::CYC_FOR_CI;
constexpr heaan::PolyType POLY_TYPE = heaan::PolyType::GRAFTED;

constexpr u32 BASE_BITS = 30;
constexpr u32 RESCALE_BITS = 25;
constexpr u32 NUM_MULTS = 3;

//---------------------------------------------------------------------------
// SECURITY-RELEVANT PARAMETERS -- ANALYSIS PENDING.
//
// The secret key is sampled at 2^SMALL_LOG_DEGREE and lifted to 2^LOG_DEGREE
// by SKGenerator::genHighDegreeKey. Lifting adds no entropy, so the switching
// key budget is sized from the degree actually sampled, halved again by CI
// (SKGenerator samples only half the coefficients there) -- see swkMaxBits().
// What the lifting buys is fc1's key-less fold: a key lifted from 2^l is
// invariant under exactly the rotations whose step is a multiple of
// 2^(l-1), so those rotations need no keys at all.
//
// The >=128-bit claim for this configuration has NOT been signed off yet; the
// numbers in swkMaxBits() are provisional and HW may change. See
// NOTES_FOR_HUMAN.md and the "Security" section of README.md. Everything a
// review would need to change is in this block.
//---------------------------------------------------------------------------

constexpr u32 SMALL_LOG_DEGREE = 15; // degree the secret key is sampled at
constexpr u32 HW = 0;                // 0 = uniform ternary
constexpr double SWK_MARGIN = 5.0;
constexpr double NOISE_STDDEV = 3.2;

// Provisional 128-bit modulus budget for a uniform-ternary secret (hw == 0).
// 2^13..2^15 from HEaven's maxBitsPolicy128(); 2^16 and 2^17 from
// ePrint 2024/463. The entries are hw-specific -- do not substitute a value
// from an hw > 0 table.
inline u32 maxBits128(u32 log_degree) {
    switch (log_degree) {
    case 13: return 214;
    case 14: return 430;
    case 15: return 868;
    case 16: return 1748;
    case 17: return 3523;
    default:
        throw std::runtime_error("no 128-bit maxBits entry for N=2^" +
                                 std::to_string(log_degree) + " with hw=0");
    }
}

// The budget a switching key may spend: that of the degree the secret was
// sampled at, less one for the conjugate-invariant ring.
inline u32 swkMaxBits() {
    return maxBits128(SMALL_LOG_DEGREE -
                      (NTT_ALG == heaan::NTTAlgorithm::CYC_FOR_CI ? 1 : 0));
}

//===========================================================================
// Layer geometry.
//
// A layer applies a p x q matrix, cyclically diagonalised: p diagonals, with
// the q/p cosets summed afterwards ("the fold"). n_out is the number of rows
// that carry a real output; the rest are zero-padding.
//
// fc1 is rectangular 128x512, so it folds 4:1. fc2 is deliberately *squared*
// to 128x128 rather than the natural 16x128: its fold stride would not meet
// the key-less fold's invariance period, and a keyed fold costs one mod-down
// per coset. At 128x128 the cosets become giant steps instead, which the
// double-hoisted BSGS accumulates behind a single mod-down.
//===========================================================================

struct LayerGeom {
    u32 p;        // output period = number of cyclic diagonals
    u32 q;        // input period; p | q and q | COORD_DIM
    u32 n_out;    // rows carrying a real output (n_out <= p)
    u32 num_bs;   // BSGS baby-step count
    bool activate; // x^2 after this layer
};

constexpr LayerGeom FC1{128, 512, HIDDEN_DIM, 64, true};
constexpr LayerGeom FC2{128, 128, LABEL_DIM, 64, false};

// Levels each layer runs at, as offsets below the top level.
constexpr u32 FC1_IN_DROP = 0, FC1_OUT_DROP = 1;
constexpr u32 FC2_IN_DROP = 2, FC2_OUT_DROP = 3;

//===========================================================================
// Key-less rotation helpers.
//
// HomEval::frobMap takes a Galois exponent rather than a slot step, and the
// conversion is arithmetic on the ring degree with no key material in it.
//===========================================================================

// The Galois exponent whose automorphism rotates left by `step`:
// 5^(step mod 2^(log_degree-1)) mod 2^(log_degree+1). Same conversion the
// keyed HomEval::rot performs internally.
inline i32 frobPowForRot(i32 step, u32 log_degree) {
    const uint64_t num_slots = 1ULL << (log_degree - 1);
    const uint64_t modulus = 1ULL << (log_degree + 1);
    auto exp = static_cast<uint64_t>(
        ((static_cast<int64_t>(step) % static_cast<int64_t>(num_slots)) +
         static_cast<int64_t>(num_slots)) %
        static_cast<int64_t>(num_slots));
    uint64_t pow = 1, base = 5;
    for (; exp != 0; exp >>= 1) {
        if (exp & 1)
            pow = (pow * base) % modulus;
        base = (base * base) % modulus;
    }
    return static_cast<i32>(pow);
}

// Rotation-step granularity at which frobMap is valid for a key lifted from
// 2^log_degree_low: such a key is fixed by exactly the rotations whose step is
// a multiple of 2^(log_degree_low - 1). A divisibility, not a lower bound.
inline u32 rotInvariantPeriod(u32 log_degree_low) {
    if (log_degree_low == 0)
        throw std::runtime_error("log_degree_low must be positive");
    return 1U << (log_degree_low - 1);
}

//===========================================================================
// BSGS index sets.
//
// Derived from the layer *shape* alone, never from the weight values, because
// client_key_generation must produce the rotation keys before it has -- or is
// entitled to -- the model. Every diagonal is kept even if a weight matrix
// happened to make one all-zero, so the two sides cannot drift apart.
//===========================================================================

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

//===========================================================================
// Directory layout inside io/<instance>/. The three measured directory names
// (public_keys, ciphertexts_upload, ciphertexts_download) are fixed by
// harness/run_submission.py and must not be renamed.
//===========================================================================

constexpr const char *ENC_KEY_FILE = "enc_key.bin";
constexpr const char *ROT_KEY_FC1_FILE = "rot_keys_fc1.bin";
constexpr const char *ROT_KEY_FC2_FILE = "rot_keys_fc2.bin";
constexpr const char *RELIN_KEY_FILE = "relin_key.bin";
constexpr const char *SECRET_KEY_FILE = "sk.bin";

// Number of ciphertexts a batch of `n` images needs.
inline size_t numCtxts(size_t n) {
    return (n + IMAGES_PER_CTXT - 1) / IMAGES_PER_CTXT;
}

} // namespace mlp

#endif // MLP_PARAMS_HPP_

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
// x^2 is the activation the model was *trained* with, not an approximation of
// something else, so the homomorphic circuit evaluates the network exactly.

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

// Which scheme a stage uses is a function of instance size alone: single
// (1 image) uses the Halevi-Shoup matvec scheme below, small/medium/large
// (100, 1000, 10000) use the PCMM (GEMM-based) scheme in mlp_pcmm.hpp. Every
// stage binary dispatches on this at its own entry point.
//
// PCMM does identical work for any batch up to pcmm::slotsPerMsg(), so small
// inherits medium's cost outright while dropping the rotation keys (~45 MB ->
// ~54 KB of public key material) and the diagonal encoding. Single stays on
// HS: it is the faster of the two on one image, and the one size that
// exercises public-key encryption -- PCMM's matrix encryption is necessarily
// symmetric-key (see client_key_generation.cpp).
inline bool usePcmm(InstanceSize size) { return size >= InstanceSize::SMALL; }

// Normalization the model was trained under (torchvision ToTensor + Normalize).
// The harness writes pixels already scaled to [0,1], so only the affine part
// applies. Identical to harness/mnist/test.py:60.
constexpr double MNIST_MEAN = 0.1307;
constexpr double MNIST_STD = 0.3081;

//===========================================================================
// Packing (HS scheme).
//
// One image occupies COORD_DIM padded coordinates; coordinate c of image i
// lives in slot c * IMAGES_PER_CTXT + i. 484 pads to 512 rather than 1024,
// which fits 32 images per ciphertext instead of 16.
//===========================================================================

constexpr u32 LOG_DEGREE = 15;
constexpr u32 LOG_SLOTS = LOG_DEGREE - 1;   // CI pairs log_slots with N/2
constexpr u32 SLOTS = 1U << LOG_SLOTS;      // 16384
constexpr u32 COORD_DIM = 512;              // 484 zero-padded
constexpr u32 IMAGES_PER_CTXT = SLOTS / COORD_DIM; // 32

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
// The secret key is sampled directly at 2^LOG_DEGREE with uniform-ternary
// coefficients -- no lifting into a larger ring, so there is no separate
// lifting argument for a review to make. The switching-key budget is read at
// that degree, halved once more for CI (which samples only half the
// coefficients) -- see swkMaxBits().
//
// The >=128-bit claim for this configuration has NOT been signed off: the
// numbers in maxBits128() are provisional and HW may change. Everything a
// review would need to change is in this block.
//---------------------------------------------------------------------------

constexpr u32 HW = 0; // 0 = uniform ternary
constexpr double SWK_MARGIN = 5.0;
constexpr double NOISE_STDDEV = 3.2;

// Provisional 128-bit modulus budget for a uniform-ternary secret (hw == 0),
// keyed by RLWE dimension; the 2^16 and 2^17 entries follow ePrint 2024/463.
// The values are hw-specific -- do not substitute one from an hw > 0 table.
// Shared by the HS scheme and the PCMM scheme in mlp_pcmm.hpp, so a review
// edits one table.
inline u32 maxBits128(u32 log_degree) {
    switch (log_degree) {
    case 12: return 106;
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
    return maxBits128(LOG_DEGREE -
                      (NTT_ALG == heaan::NTTAlgorithm::CYC_FOR_CI ? 1 : 0));
}

//===========================================================================
// Layer geometry.
//
// A layer applies a p x q matrix, cyclically diagonalised: p diagonals, with
// the q/p cosets summed afterwards ("the fold"). n_out is the number of rows
// that carry a real output; the rest are zero-padding.
//
// fc1 is rectangular 128x512, so it folds 4:1, with rotation keys. fc2 is
// deliberately *squared* to 128x128 rather than the natural 16x128, which makes
// q/p == 1 so it needs no fold at all: its cosets ride the matvec's giant steps
// instead.
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
constexpr const char *ROT_KEY_FOLD_FILE = "rot_keys_fold.bin";
constexpr const char *RELIN_KEY_FILE = "relin_key.bin";
constexpr const char *SECRET_KEY_FILE = "sk.bin";

// Number of ciphertexts a batch of `n` images needs.
inline size_t numCtxts(size_t n) {
    return (n + IMAGES_PER_CTXT - 1) / IMAGES_PER_CTXT;
}

//===========================================================================
// PCMM (GEMM-based) scheme parameters, for small/medium/large.
//
// feature = matrix row (the GEMM contraction dimension), image = column/slot
// -- the opposite packing from the HS scheme above. One message holds
// slotsPerMsg() images; a larger batch splits into further "blocks" within a
// single ciphertext matrix. See mlp_pcmm.hpp for the circuit itself.
//===========================================================================

namespace pcmm {

//---------------------------------------------------------------------------
// Tuning profile.
//
// The batch size decides how many blocks a matrix row splits into, and that in
// turn decides which ring and which modulus chain come out cheapest -- so one
// setting cannot be right for every instance size. What differs lives in this
// struct; everything below it is shared.
//
// Two profiles, not three: small (100) and medium (1000) both fit a single
// block, so they do identical work and share one setting. Large (10000) is
// tuned on its own.
//---------------------------------------------------------------------------

struct Profile {
    heaan::NTTAlgorithm ntt_alg;
    heaan::PolyType poly_type;
    u32 log_degree;
    u32 base_bits;
    u32 rescale_bits;
};

// SIMPLE32 (32-bit RNS primes) on both profiles: smaller machine words for the
// GEMM, at the cost of more primes to reach the same modulus budget. Both
// chains below are sized against that trade, so neither is valid read back
// against a GRAFTED budget.

// --- small (100) and medium (1000) -----------------------------------------
// One NORMAL message at N=2^12 holds 2^11 = 2048 images, so either batch fits
// a single block already. CI's doubled slot count would remove no block here
// and only costs constant factors, which is why these sizes use the plain ring
// while large does not. Under NORMAL the RLWE dimension is the degree itself,
// 2^12.
constexpr Profile MEDIUM_PROFILE{heaan::NTTAlgorithm::NORMAL,
                                 heaan::PolyType::SIMPLE32,
                                 /*log_degree=*/12,
                                 /*base_bits=*/34,
                                 /*rescale_bits=*/24};

// --- large (10000) ---------------------------------------------------------
// CI packs twice the images per stored coefficient, and at this batch that is
// what removes blocks: 2^14 real slots per message take all 10000 images in
// one. Under CI the RLWE dimension is half the degree -- 2^14, a wider budget,
// which is what lets this chain be wider than medium's.
constexpr Profile LARGE_PROFILE{heaan::NTTAlgorithm::CYC_FOR_CI,
                                heaan::PolyType::SIMPLE32,
                                /*log_degree=*/15,
                                /*base_bits=*/44,
                                /*rescale_bits=*/27};

// Every PCMM stage resolves its parameters through this one call, from the
// instance size it was invoked with. A stage that resolves a different profile
// than its peers produces objects the others cannot read -- see the header
// comment at the top of this file.
constexpr Profile profile(InstanceSize size) {
    return size == InstanceSize::LARGE ? LARGE_PROFILE : MEDIUM_PROFILE;
}

// INPUT_DIM plus one row, which folds b1 in as an extra weight column against
// an appended "ones" row (see mlp_pcmm.hpp).
constexpr u32 IN_P = INPUT_DIM + 1; // 485

//---------------------------------------------------------------------------
// SECURITY-RELEVANT PARAMETERS -- ANALYSIS PENDING, same status as the HS
// scheme's block above. PCMM's secret key is sampled directly at the profile's
// log_degree (no lifting), so its security rests on
// maxBits128(logRlweDim(profile)) alone.
//
// The budget each profile's chain is sized against:
//   small/medium  NORMAL N=2^12 -> RLWE dim 2^12
//   large         CI     N=2^15 -> RLWE dim 2^14
//
// What bounds base_bits/rescale_bits at each end:
//   lower -- the bottom modulus (base_bits - rescale_bits) has to clear the
//     coefficient range of fc2's output (not the logit range itself, but its
//     inverse DFT, whose max coefficient magnitude is materially larger), or
//     one wrapped coefficient smears error across a whole message and
//     collapses accuracy batch-wide.
//   upper -- the relin key. x^2 runs tensor -> rescale -> relin, so the key's
//     modulus is a single prime of base_bits + rescale_bits bits (not
//     base + 2*rescale), and key generation refuses unless the budget above
//     admits it with margin.
// Both settings were chosen by accuracy and timing measurement under SIMPLE32.
//---------------------------------------------------------------------------

constexpr u32 HW = 0; // uniform ternary, same as the HS scheme
constexpr double SWK_MARGIN = 5.0;
constexpr u32 NUM_MULTS = 3; // fc1, x^2, fc2 -- one rescale each

// Level schedule, one rescale per operation, offsets below the top level:
//   L(top)   encrypt X, encode U1 -> fc1 pcmm+rescale        -> L(top-1)
//   L(top-1) x^2: tensor (still L(top-1)) -> rescale         -> L(top-2), relin
//   L(top-2) encode U2 -> fc2 pcmm+rescale                   -> L(top-3)
//   L(top-3) +b2 (plaintext add, no level cost), decrypt
// Because the rescale runs BEFORE the relinearization (tensor -> rescale ->
// relin, not the more usual tensor -> relin -> rescale), the relin key is
// built at x^2's post-rescale level, not at its input level.
constexpr u32 FC1_IN_DROP = 0, FC1_OUT_DROP = 1;
constexpr u32 SQUARE_OUT_DROP = 2; // relin key lives at this level
constexpr u32 FC2_IN_DROP = 2, FC2_OUT_DROP = 3;
constexpr u32 B2_DROP = 3; // b2 is added at fc2's output level, no change

// The RLWE dimension the 128-bit budget is keyed by: the ring degree itself in
// NORMAL, half of it under CI, which samples only its free half.
constexpr u32 logRlweDim(const Profile &p) {
    return p.ntt_alg == heaan::NTTAlgorithm::CYC_FOR_CI ? p.log_degree - 1
                                                        : p.log_degree;
}

// Coefficients one coeff-encoded block stores: the whole block in the NORMAL
// ring, half of it under CI. The server-side column count is counted in these.
constexpr u32 ringDim(const Profile &p) { return 1U << logRlweDim(p); }

// Images one message holds. This is degree/2 in BOTH rings -- NORMAL has N/2
// complex slots, CI has N/2 real ones -- so it is NOT ringDim(p), which the
// two differ in. The two coincide under CI only.
constexpr u32 slotsPerMsg(const Profile &p) { return 1U << (p.log_degree - 1); }

// A batch bigger than one message splits into further blocks within one
// ciphertext matrix. Blocks are counted in images, hence slotsPerMsg.
constexpr u32 numBlocks(const Profile &p, u32 num_images) {
    return (num_images + slotsPerMsg(p) - 1) / slotsPerMsg(p);
}

// Server-side matrix column count: stored coefficients, i.e. blocks *
// ringDim() -- NOT the client-side Matrix<Real> width, which is
// blocks * (1 << log_degree). Under CI the two genuinely differ; under NORMAL
// they coincide. See the packing note in mlp_pcmm.hpp.
constexpr u32 numCols(const Profile &p, u32 num_images) {
    return numBlocks(p, num_images) * ringDim(p);
}

} // namespace pcmm

} // namespace mlp

#endif // MLP_PARAMS_HPP_

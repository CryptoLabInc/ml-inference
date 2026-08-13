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
#include "params.h"

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

// Which homomorphic scheme a stage uses is a function of instance size alone:
// single (1 image) uses the Halevi-Shoup layer scheme below; small/medium/large
// (100, 1000, 10000) use the PCMM (GEMM-based) scheme in mlp_pcmm.hpp --
// pcmm's per-image cost keeps falling as the batch grows, while HS's fixed
// 128-images-per-ciphertext packing does not. Every stage binary dispatches on
// this at its own entry point; see mlp_pcmm.hpp for the PCMM-side parameters.
//
// small sits on PCMM because pcmm::numBlocks() is 1 for every batch up to
// pcmm::slotsPerMsg() (2048 under the profile small and medium share): 100
// images and 1000 images do *identical* work, so small inherits medium's cost
// outright -- against HS it drops the rotation keys entirely (175 MB -> 110 KB
// of public key material) and skips the diagonal encoding. single stays on HS:
// it is the one size where the
// key-less fold and public-key encryption are exercised, and PCMM's matrix
// encrypt is necessarily symmetric-key (see runPcmm in
// client_key_generation.cpp).
inline bool usePcmm(InstanceSize size) { return size >= InstanceSize::SMALL; }

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
// discussion in DESIGN.md.
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
// The secret key is sampled at 2^SMALL_LOG_DEGREE. The switching key budget is
// sized from that degree, halved again by CI (SKGenerator samples only half the
// coefficients there) -- see swkMaxBits().
//
// SMALL_LOG_DEGREE == LOG_DEGREE here, so the key is sampled directly in the
// ring it is used in and SKGenerator::genHighDegreeKey is a no-op: there is NO
// lifting, and therefore no separate lifting argument for a review to make.
// An earlier configuration sampled at 2^15 and lifted to 2^17 to buy fc1's
// key-less fold (a key lifted from 2^l is invariant under rotations whose step
// is a multiple of 2^(l-1)). Dropping the lifting costs that fold -- fc1 now
// folds with keys, see makeLayer -- but lets the whole scheme run at 2^15
// instead of 2^17, which is a net win: the evaluation is ~25% faster and the
// rotation keys shrink from 174.8 MB to 44.9 MB. The sampled degree, and hence
// the LWE problem and the 430-bit budget, are unchanged from that earlier
// configuration.
//
// The >=128-bit claim for this configuration has NOT been signed off yet; the
// numbers in swkMaxBits() are provisional and HW may change. See section 5,
// "Security", of DESIGN.md. Everything a review would need to change is in
// this block.
//---------------------------------------------------------------------------

constexpr u32 SMALL_LOG_DEGREE = 15; // degree the secret key is sampled at
constexpr u32 HW = 0;                // 0 = uniform ternary
constexpr double SWK_MARGIN = 5.0;
constexpr double NOISE_STDDEV = 3.2;

// Provisional 128-bit modulus budget for a uniform-ternary secret (hw == 0),
// keyed by RLWE dimension. 2^12..2^15 from HEaven's maxBitsPolicy128(); 2^16
// and 2^17 from ePrint 2024/463. The entries are hw-specific -- do not
// substitute a value from an hw > 0 table. Shared by both the HS layer scheme
// (which uses 13..17) and the PCMM scheme in mlp_pcmm.hpp (which needs 12 for
// the small/medium profile and 14 for large): one table, so a review only
// edits it once.
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
// fc1 is rectangular 128x512, so it folds 4:1 -- with keys, since the secret
// key is no longer lifted (see the security block above). fc2 is deliberately
// *squared* to 128x128 rather than the natural 16x128, which makes q/p == 1 so
// it needs no fold at all: its cosets become giant steps instead, which the
// double-hoisted BSGS accumulates behind a single mod-down. That was worth
// doing when folds were key-less and is worth more now that they cost a
// mod-down per coset.
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
// fc1's fold keys. Written only when the key-less fold is unavailable, which
// is the case whenever the secret key is not lifted -- see makeLayer.
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
// feature = ciphertext row (pcmm's contraction dim), image = slot -- the
// opposite packing from the HS scheme above. One message holds
// ringDim() images; batches larger than that split into further "blocks"
// within a single ICtMatrix, handled by mlp_pcmm.hpp. See its header comment
// for the algorithm (pcmm GEMM, the section-6.3 coeff/slot relabeling trick,
// the bias fold into an extra weight column).
//===========================================================================

namespace pcmm {

//---------------------------------------------------------------------------
// Tuning profile.
//
// The batch size decides how many blocks a matrix row splits into, and that in
// turn decides which ring, which degree and which modulus chain come out
// cheapest -- so one setting cannot be right for every instance size. What
// differs lives in this struct; everything below it is shared.
//
// Two profiles, not three: small (100) and medium (1000) both fit a single
// block, so they do identical work and share one setting, exactly as the
// usePcmm() comment above describes. Large (10000) is tuned on its own.
//---------------------------------------------------------------------------

struct Profile {
    heaan::NTTAlgorithm ntt_alg;
    heaan::PolyType poly_type;
    u32 log_degree;
    u32 base_bits;
    u32 rescale_bits;
};

// SIMPLE32 (32-bit RNS primes) on both: it halves the word size pcmm's GEMM
// backend operates on, at the cost of more primes to reach the same modulus
// budget. Both chains below are sized against that trade, so neither is valid
// read back against a GRAFTED budget.

// --- small (100) and medium (1000) -----------------------------------------
// One NORMAL message at N=2^12 holds 2^11 = 2048 images, so either batch fits
// a single block already. CI's doubled slot count would remove no block here
// and only costs constant factors, which is why these sizes use the plain ring
// while large does not. Under NORMAL the RLWE dimension is the degree itself
// -- 2^12, the same 128-bit budget entry (106 bits) the CI N=2^13 setting this
// replaces was keyed by, so the security question is unchanged.
constexpr Profile MEDIUM_PROFILE{heaan::NTTAlgorithm::NORMAL,
                                 heaan::PolyType::SIMPLE32,
                                 /*log_degree=*/12,
                                 /*base_bits=*/34,
                                 /*rescale_bits=*/24};

// --- large (10000) ---------------------------------------------------------
// CI packs twice the images per stored coefficient, and at this batch that is
// what removes blocks: 2^14 real slots per message take all 10000 images in
// one. Under CI the RLWE dimension is half the degree -- 2^14, a 430-bit
// budget, which is what lets this chain be wider than medium's.
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

// IN_DIM padded to a rows x cols GEMM contraction; +1 row folds b1 in as an
// extra weight column against an appended "ones" row (see mlp_pcmm.hpp).
constexpr u32 IN_P = INPUT_DIM + 1; // 485

//---------------------------------------------------------------------------
// SECURITY-RELEVANT PARAMETERS -- ANALYSIS PENDING, same status as the HS
// scheme's block above and the same open item in DESIGN.md section 5. Unlike
// HS, PCMM's secret key is sampled directly at its working degree (no
// lifting), so its security rests on maxBits128(logRlweDim(profile)) alone
// with no separate lifting argument to review.
//
// The budget each profile's chain is sized against:
//   small/medium  NORMAL N=2^12 -> RLWE dim 2^12 -> 106 bits
//   large         CI     N=2^15 -> RLWE dim 2^14 -> 430 bits
//
// What the two ends of base_bits/rescale_bits are:
//   lower -- the bottom modulus (base_bits - rescale_bits) has to clear the
//     iDFT coefficient range of the fc2 output (not the logit range itself,
//     but its inverse DFT, whose max coefficient magnitude is materially
//     larger), or one wrapped coefficient smears error across a whole message
//     and collapses accuracy batch-wide.
//   upper -- the relin key. x^2 runs tensor -> rescale -> relin, so the key's
//     modulus is a single prime of base_bits + rescale_bits bits (not
//     base + 2*rescale), and the gadget builder refuses unless the budget
//     above admits it with margin.
// Both settings were chosen by measurement under SIMPLE32; see HEaaN2 PR #222
// for the accuracy sweeps behind them.
//---------------------------------------------------------------------------

constexpr u32 HW = 0; // uniform ternary, same as the HS scheme
constexpr double SWK_MARGIN = 5.0;
constexpr u32 NUM_MULTS = 3; // fc1, x^2, fc2 -- one rescale each

// Level schedule, one rescale per operation, offsets below the top level:
//   L(top)   encrypt X, encode U1 -> fc1 pcmm+rescale     -> L(top-1)
//   L(top-1) x^2: tensor (still L(top-1)) -> rescale        -> L(top-2), relin
//   L(top-2) encode U2 -> fc2 pcmm+rescale                 -> L(top-3)
//   L(top-3) +b2 (plaintext add, no level cost), decrypt
// The relin key's modulus is levels.mods[top-SQUARE_OUT_DROP]: rescale runs
// BEFORE relin here (tensor -> rescale -> relin, not the more usual
// tensor -> relin -> rescale), so the key is built at x^2's post-rescale
// level, not its input level.
constexpr u32 FC1_IN_DROP = 0, FC1_OUT_DROP = 1;
constexpr u32 SQUARE_OUT_DROP = 2; // relin key lives at this level
constexpr u32 FC2_IN_DROP = 2, FC2_OUT_DROP = 3;
constexpr u32 B2_DROP = 3; // b2 is added at fc2's output level, no change

// The RLWE dimension the 128-bit budget is keyed by: the ring degree itself in
// NORMAL, half of it in the conjugate-invariant subring (CI samples only its
// free half). Numerically this is also the count below, but the two are
// different quantities and only this one belongs in maxBits128.
constexpr u32 logRlweDim(const Profile &p) {
    return p.ntt_alg == heaan::NTTAlgorithm::CYC_FOR_CI ? p.log_degree - 1
                                                        : p.log_degree;
}

// Coefficients one coeff-encoded block actually stores: the whole block in the
// NORMAL ring, half of it in CI. Matches ringDim() in HEaaN2's
// src/HomEvalMatrix.cpp, which the server-side column count has to agree with.
constexpr u32 ringDim(const Profile &p) { return 1U << logRlweDim(p); }

// Images one message holds. This is degree/2 in BOTH rings -- NORMAL has N/2
// complex slots, CI has N/2 real ones -- so it is NOT ringDim(p), which they
// differ in. The two coincide under CI only, which is why a CI-only version of
// this file could get away with a single constant for both.
constexpr u32 slotsPerMsg(const Profile &p) { return 1U << (p.log_degree - 1); }

// A batch bigger than one message splits into further blocks within one
// ICtMatrix. Blocks are counted in images, hence slotsPerMsg, not ringDim.
constexpr u32 numBlocks(const Profile &p, u32 num_images) {
    return (num_images + slotsPerMsg(p) - 1) / slotsPerMsg(p);
}

// Server-side ICtMatrix/IPtMatrix column count: stored coefficients, i.e.
// blocks * ringDim() -- NOT the client-side Matrix<Real> width, which is
// blocks * (1 << log_degree). Under CI the two genuinely differ; under NORMAL
// they happen to coincide. See the packing note in mlp_pcmm.hpp.
constexpr u32 numCols(const Profile &p, u32 num_images) {
    return numBlocks(p, num_images) * ringDim(p);
}

} // namespace pcmm

} // namespace mlp

#endif // MLP_PARAMS_HPP_

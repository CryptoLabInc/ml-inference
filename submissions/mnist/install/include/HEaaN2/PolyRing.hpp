////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2025-2026 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is a part of HEaaN2 homomorphic encryption library.            //
// - This header is provided for use with the HEaaN2 library and may be       //
//   included in software that links against HEaaN2.                          //
// - Redistribution or modification of this file, in whole or in part,        //
//   is not permitted without prior written consent from Crypto Lab Inc.      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "HEaaN2/General.hpp"

#include <vector>

namespace heaan {

/// @brief The type of Polynomial which determines RNS system
enum class HEAAN2_API PolyType {
    /// The usual RNS system, where we rescale in the unit of words.
    SIMPLE,
    /// The RNS system which utilizes inverse-rescale, exploiting the special
    /// part of modulus, sprout, for the flexible construction of moduli chain.
    /// Please refer to https://eprint.iacr.org/2024/1014 for the details.
    GRAFTED,
    /// SIMPLE, but with 32-bit (rather than 64-bit) RNS primes. Halves the
    /// word size pcmm's GEMM backend operates on, at the cost of needing more
    /// primes to reach the same modulus bit budget.
    SIMPLE32
};

/// @brief A struct representing the modulus of the polynomial ring in RNS form.
struct HEAAN2_API PolyMod : public std::vector<u64> {
    /// @brief Constructs a PolyMod with the specified size.
    /// @param size The size of the PolyMod.
    PolyMod(size_t size = 0);
    /// @brief Constructs a PolyMod from a vector of u64.
    /// @param vec The vector of u64.
    PolyMod(const std::vector<u64> &vec);

    bool operator==(const PolyMod &other) const;
    bool operator!=(const PolyMod &other) const;
    Real128 operator/(const PolyMod &other) const;
};

/// @brief NTT algorithm type
enum class HEAAN2_API NTTAlgorithm {
    /// NTT algorithm for the usual cyclotomic polynomial ring.
    NORMAL,
    /// NTT algorithm for Conjugate-Invariant subring.
    /// In coefficent state, the polynomial holds half of the coefficients.
    /// In slot state, the polynomial holds cyclic-NTT-ed data.
    /// Please refer to https://eprint.iacr.org/2018/952 for the details.
    CYC_FOR_CI
};

/// @brief Polynomial ring parameters
/// @details
/// - mod: The RNS representation of the modulus of the polynomial ring.
/// - log_degree: log of the degree of the polynomial ring.
/// - ntt_alg: NTT algorithm type of the polynomial ring.
struct HEAAN2_API PolyRing {
    PolyMod mod;
    u32 log_degree;
    NTTAlgorithm ntt_alg;
};

} // namespace heaan

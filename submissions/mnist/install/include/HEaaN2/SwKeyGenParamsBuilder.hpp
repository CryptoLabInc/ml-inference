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

#include "HEaaN2/Distribution.hpp"
#include "HEaaN2/Export.hpp"
#include "HEaaN2/SwKeyGenerator.hpp"

#include <vector>

namespace heaan::paramsUtils {

/// @brief Builder class for constructing customized SwKeyGenParams.
class HEAAN2_API SwKeyGenParamsBuilder {
public:
    /// @brief Constructs SwKeyGenParamsBuilder.
    SwKeyGenParamsBuilder();

    /// @brief Sets the noise distribution.
    /// @param dist The noise distribution
    void setNoiseDistribution(const Distribution &dist);

    /// @brief Sets the polynomial ring.
    /// @param log_degree Log degree of the ring.
    /// @param poly_type The type of the polynomial (SIMPLE/GRAFTED).
    /// @param support_conj_inv If true, primes are chosen for degree
    /// 2^(log_degree+1), allowing conversion to conjugate-invariant subring
    /// representation. If false, primes are chosen for degree 2^log_degree.
    void setRing(u32 log_degree, PolyType poly_type,
                 bool support_conj_inv = true);

    /// @brief Sets the modulus for mod-up.
    /// @param max_bits The total bit-size budget of the QP modulus.
    /// @param margin Minimum bit-size gap between P and Q moduli such that
    /// (P modulus bits) >= (Q modulus bits) + margin.
    /// @details @p margin is a knob to control the trade-off between precision
    /// and performance of key-switching.
    void setModUpPrimes(u32 max_bits, double margin);

    /// @brief Builds SwKeyGenParams for a given modulus.
    /// @param mod Modulus at the maximum (top) level.
    /// @param conj_inv If true, the switch key is generated in
    /// conjugate-invariant subring (NTTAlgorithm::CYC_FOR_CI).
    /// If false, uses the usual cyclotomic ring (NTTAlgorithm::NORMAL).
    /// @return SwKeyGenParams configured for the given modulus @p mod.
    SwKeyGenParams build(const PolyMod &mod, bool conj_inv = false) const;

    /// @brief Builds SwKeyGenParams to be compatible with given moduli.
    /// @param mods Vector of moduli. Must not be empty.
    /// @param conj_inv If true, the switch key is generated in
    /// conjugate-invariant subring (NTTAlgorithm::CYC_FOR_CI).
    /// If false, uses the usual cyclotomic ring (NTTAlgorithm::NORMAL).
    /// @return SwKeyGenParams configured to support all moduli @p mods.
    SwKeyGenParams build(const std::vector<PolyMod> &mods,
                         bool conj_inv = false) const;

    /// @brief Builds SwKeyGenParams for each modulus in the input vector.
    /// @param mods Vector of moduli. Must not be empty.
    /// @param conj_inv If true, the switch key is generated in
    /// conjugate-invariant subring (NTTAlgorithm::CYC_FOR_CI).
    /// If false, uses the usual cyclotomic ring (NTTAlgorithm::NORMAL).
    /// @return Vector of SwKeyGenParams in the same order as @p mods.
    std::vector<SwKeyGenParams> buildForEach(const std::vector<PolyMod> &mods,
                                             bool conj_inv = false) const;

private:
    Pimpl impl;
    u32 max_bits_;
    double margin_;
};

} // namespace heaan::paramsUtils

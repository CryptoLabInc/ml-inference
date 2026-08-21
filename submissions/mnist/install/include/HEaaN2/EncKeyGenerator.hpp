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
#include "HEaaN2/General.hpp"
#include "HEaaN2/IEncKey.hpp"
#include "HEaaN2/ISecretKey.hpp"
#include "HEaaN2/PresetParams.hpp"

namespace heaan {

/// @brief A struct for public key generation parameters
/// @details
/// - noise_dist: The noise distribution used for key generation.
/// - poly_type: The type of the polynomial (SIMPLE/GRAFTED).
/// - mod: The modulus of the polynomial ring.
/// - ntt_alg: The NTT algorithm (NORMAL/CYC_FOR_CI).
struct HEAAN2_API EncKeyGenParams {
public:
    /// @brief Constructs EncKeyGenParams with the specified parameters.
    /// @param noise_dist The noise distribution.
    /// @param poly_type The type of the polynomial (SIMPLE/GRAFTED).
    /// @param mod The modulus of the polynomial ring.
    /// @param ntt_alg The NTT algorithm (NORMAL/CYC_FOR_CI).
    EncKeyGenParams(const Distribution &noise_dist, PolyType poly_type,
                    const PolyMod &mod,
                    NTTAlgorithm ntt_alg = NTTAlgorithm::NORMAL)
        : noise_dist_(noise_dist), poly_type_(poly_type), mod_(mod),
          ntt_alg_(ntt_alg) {}

    /// @brief Gets the noise distribution.
    /// @return The noise distribution.
    Distribution getDistribution() const { return noise_dist_; }
    /// @brief Gets the polynomial type.
    /// @return The polynomial type.
    PolyType getPolyType() const { return poly_type_; };
    /// @brief Gets the modulus of the polynomial ring.
    /// @return The modulus of the polynomial ring.
    const PolyMod &getMod() const { return mod_; };
    /// @brief Gets the NTT algorithm.
    /// @return The NTT algorithm.
    NTTAlgorithm getNTTAlgorithm() const { return ntt_alg_; };

private:
    Distribution noise_dist_;
    PolyType poly_type_;
    PolyMod mod_;
    NTTAlgorithm ntt_alg_;
};

/// @brief Encryption key generator
class HEAAN2_API EncKeyGenerator {
public:
    /// @brief Constructs an EncKeyGenerator with the specified preset
    /// parameters.
    /// @param id Preset parameters identifier.
    EncKeyGenerator(PresetParamsId id);

    /// @brief Constructs an EncKeyGenerator with the specified parameters.
    /// @param params The encryption key generation parameters.
    EncKeyGenerator(const EncKeyGenParams &params);

    /// @brief Generates an encryption key from a secret key
    /// @param sk The secret key.
    /// @return The pointer to the encryption key.
    /// @details The device of the output follows from the input.
    Ptr<IEncKey> genKey(const ISecretKey &sk) const;

    /// @brief Gets the encryption key generation parameters.
    /// @return The encryption key generation parameters.
    const EncKeyGenParams &getParams() const { return params_; }

private:
    Pimpl impl;
    EncKeyGenParams params_;
};

} // namespace heaan

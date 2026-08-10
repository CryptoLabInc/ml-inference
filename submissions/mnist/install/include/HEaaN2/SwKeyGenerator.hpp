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
#include "HEaaN2/GadgetDecomp.hpp"
#include "HEaaN2/ISecretKey.hpp"
#include "HEaaN2/ISwKey.hpp"
#include "HEaaN2/KeyUtils.hpp"
#include "HEaaN2/Levels.hpp"
#include "HEaaN2/PresetParams.hpp"

#include <set>

namespace heaan {

/// @brief A struct for switching key generation parameters
/// @details
/// - noise_dist: The noise distribution used for key generation.
/// - poly_type: The type of the polynomial (SIMPLE/GRAFTED).
/// - ntt_alg: The NTT algorithm (NORMAL/CYC_FOR_CI).
/// - gadget_decomp: The gadget decomposition parameters for key switching.
struct HEAAN2_API SwKeyGenParams {
public:
    /// @brief Constructs SwKeyGenParams with the specified parameters.
    /// @param noise_dist The noise distribution.
    /// @param poly_type The type of the polynomial (SIMPLE/GRAFTED).
    /// @param ntt_alg The NTT algorithm (NORMAL/CYC_FOR_CI).
    /// @param gadget_decomp The gadget decomposition.
    SwKeyGenParams(const Distribution &noise_dist, PolyType poly_type,
                   NTTAlgorithm ntt_alg, const GadgetDecomp &gadget_decomp)
        : noise_dist_(noise_dist), poly_type_(poly_type), ntt_alg_(ntt_alg),
          gadget_decomp_(gadget_decomp) {}

    /// @brief Gets the noise distribution.
    /// @return The noise distribution.
    Distribution getDistribution() const { return noise_dist_; }
    /// @brief Gets the polynomial type.
    /// @return The polynomial type.
    PolyType getPolyType() const { return poly_type_; };
    /// @brief Gets the NTT algorithm.
    /// @return The NTT algorithm.
    NTTAlgorithm getNTTAlgorithm() const { return ntt_alg_; };
    /// @brief Gets the gadget decomposition.
    /// @return The gadget decomposition.
    const GadgetDecomp &getGadgetDecomp() const { return gadget_decomp_; }

private:
    Distribution noise_dist_;
    PolyType poly_type_;
    NTTAlgorithm ntt_alg_;
    GadgetDecomp gadget_decomp_;
};

/// @brief Switching key generator
class HEAAN2_API SwKeyGenerator {
public:
    /// @brief Constructs a SwKeyGenerator with the specified preset
    /// parameters.
    /// @param id Preset parameters identifier.
    SwKeyGenerator(PresetParamsId id);

    /// @brief Constructs a SwKeyGenerator with the specified parameters.
    /// @param params The switching key generation parameters.
    SwKeyGenerator(const SwKeyGenParams &params);

    /// @brief Generates a switching key from one secret key to another
    /// @param sk_from The source secret key.
    /// @param sk_to The target secret key.
    /// @return The pointer to the switching key.
    /// @details The device of the output follows from the inputs.
    /// @throws if the devices of sk_from and sk_to differ.
    Ptr<ISwKey> genKey(const ISecretKey &sk_from,
                       const ISecretKey &sk_to) const;
    /// @brief Generates a compose key, which switches from the high-degree
    /// embedding of the small-ring secret key to the large-ring secret key
    /// (see RingPacker::compose).
    /// @param sk_from The secret key of the small ring.
    /// @param sk_to The secret key of the large ring.
    /// @return The pointer to the compose key.
    /// @details The device of the output follows from the inputs.
    /// @throws if the devices of sk_from and sk_to differ, or if the log
    /// degree of sk_from is not smaller than that of sk_to.
    Ptr<ISwKey> genComposeKey(const ISecretKey &sk_from,
                              const ISecretKey &sk_to) const;
    /// @brief Generates a decompose key, which switches from the large-ring
    /// secret key to the high-degree embedding of the small-ring secret key
    /// (see RingPacker::decompose).
    /// @param sk_from The secret key of the large ring.
    /// @param sk_to The secret key of the small ring.
    /// @return The pointer to the decompose key.
    /// @details The device of the output follows from the inputs.
    /// @details The decompose key encrypts sk_from under the high-degree
    /// embedding of sk_to, so its security is determined by the SMALL ring:
    /// the total modulus (QP) of this generator must satisfy the security
    /// bound of the ring of sk_to, which is smaller than that of the
    /// large ring (see the Compose example).
    /// @throws if the devices of sk_from and sk_to differ, or if the log
    /// degree of sk_to is not smaller than that of sk_from.
    Ptr<ISwKey> genDecomposeKey(const ISecretKey &sk_from,
                                const ISecretKey &sk_to) const;
    /// @brief Generates a relinearization key
    /// @param sk The secret key.
    /// @return The pointer to the relinearization key.
    /// @details The device of the output follows from the input.
    Ptr<ISwKey> genRelinKey(const ISecretKey &sk) const;
    /// @brief Generates a conjugation key
    /// @param sk The secret key.
    /// @return The pointer to the conjugation key.
    /// @details The device of the output follows from the input.
    /// @throws if the NTT algorithm is CYC_FOR_CI.
    Ptr<ISwKey> genConjKey(const ISecretKey &sk) const;
    /// @brief Generates a rotation key
    /// @param sk The secret key.
    /// @param step The rotation step.
    /// @return The pointer to the rotation key.
    /// @details The device of the output follows from the input.
    /// @throws if the step is zero.
    Ptr<ISwKey> genRotKey(const ISecretKey &sk, i32 step) const;
    /// @brief Generates a relinearization cube key, which is used for the
    /// relinCube function.
    /// @param sk The secret key.
    /// @return The pointer to the relinearization cube key.
    /// @details The device of the output follows from the input.
    Ptr<ISwKey> genRelinCubeKey(const ISecretKey &sk) const;
    /// @brief Generates rotation keys for multiple rotation steps.
    /// @param sk The secret key.
    /// @param steps The rotation steps.
    /// @return The pointers to the rotation keys for the specified steps.
    /// @details The device of the output follows from the input.
    RotKeyPtrs genRotKeys(const ISecretKey &sk,
                          const std::set<i32> &steps) const;

    /// @brief Generates leveled relinearization keys for the specified levels.
    /// @param sk The secret key.
    /// @param max_bits The maximum bit size of the moduli in a level.
    /// @param margin The margin for the temporary prime size.
    /// @param levels The levels for which the relinearization keys are
    /// generated.
    /// @return The bundle of leveled relinearization keys.
    /// @details The device of the output follows from the input.
    /// @details The temporary primes are determined dynamically based on the
    /// input secret key, max_bits, and margin. The size of the temporary
    /// primes is guaranteed to be larger than the modulus of each
    /// level by margin in bits.
    /// The upper bound of the log2 of the temporary primes + the moduli of a
    /// level is max_bits.
    KeyPtrBundle genLeveledRelinKeys(const ISecretKey &sk, u32 max_bits,
                                     double margin, const Levels &levels) const;

    /// @brief Gets the switching key generation parameters.
    /// @return The switching key generation parameters.
    const SwKeyGenParams &getParams() const { return params_; }

private:
    Pimpl impl;
    SwKeyGenParams params_;
};

} // namespace heaan

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
#include "HEaaN2/ICiphertext.hpp"
#include "HEaaN2/ICtMatrix.hpp"
#include "HEaaN2/IEncKey.hpp"
#include "HEaaN2/IPlaintext.hpp"
#include "HEaaN2/IPtMatrix.hpp"
#include "HEaaN2/ISecretKey.hpp"
#include "HEaaN2/PresetParams.hpp"

#include <optional>

namespace heaan {

/// @brief Default standard deviation for discrete Gaussian noise distribution.
const Real DEFAULT_NOISE_DG = 3.2;

/// @brief A struct for encryption parameters.
/// @details
/// - noise_dist: The noise distribution used for encryption.
struct HEAAN2_API EncryptParams {

    /// @brief Constructs EncryptParams with the specified noise distribution.
    /// @param noise_dist The noise distribution (default: DiscreteGaussian
    /// with standard deviation 3.2).
    EncryptParams(
        const Distribution &noise_dist = DiscreteGaussian(DEFAULT_NOISE_DG))
        : noise_dist(noise_dist) {}

    /// @brief Gets the noise distribution.
    /// @return The noise distribution.
    Distribution getDistribution() const { return noise_dist; }

private:
    Distribution noise_dist;
};

/// @brief A class for encrypting plaintexts into ciphertexts and
/// decrypting ciphertexts to plaintexts
class HEAAN2_API EnDecryptor {
public:
    /// @brief Constructs an EnDecryptor without specifying parameters.
    /// @details The EnDecryptor is capable of decrypting any ciphertexts when
    /// provided with a valid secret key
    EnDecryptor();
    /// @brief Constructs an EnDecryptor with the specified preset parameters.
    /// @param id Preset parameters identifier.
    EnDecryptor(PresetParamsId id);
    /// @brief Constructs an EnDecryptor with the specified encryption
    /// parameters.
    /// @param params The encryption parameters.
    EnDecryptor(const EncryptParams &params);

    /// @brief Encrypts a plaintext into a ciphertext using a symmetric secret
    /// key.
    /// @param[in] ptxt The input plaintext to encrypt
    /// @param[in] sk The secret key used for encryption
    /// @param[out] ctxt The output ciphertext.
    /// @details The device of the output follows from the input.
    /// @throws if the devices of ptxt and sk differ.
    void encrypt(const IPlaintext &ptxt, const ISecretKey &sk,
                 ICiphertext &ctxt) const;

    /// @brief Encrypts a plaintext into a ciphertext using a public encryption
    /// key.
    /// @param[in] ptxt The input plaintext to encrypt
    /// @param[in] enckey The encryption key used for encryption
    /// @param[out] ctxt The output ciphertext.
    /// @details The device of the output follows from the input.
    /// @throws if the devices of ptxt and enckey differ.
    void encrypt(const IPlaintext &ptxt, const IEncKey &enckey,
                 ICiphertext &ctxt) const;

    /// @brief Decrypts a ciphertext into a plaintext using a secret key
    /// @param[in] ctxt The input ciphertext to decrypt
    /// @param[in] sk The secret key used for decryption
    /// @param[out] ptxt The output plaintext.
    /// @details The device of the output follows from the input.
    /// @throws if the devices of ctxt and sk differ.
    void decrypt(const ICiphertext &ctxt, const ISecretKey &sk,
                 IPlaintext &ptxt) const;

    /// @brief Encrypts a plaintext matrix into a ciphertext matrix using a
    /// symmetric secret key.
    /// @param[in] ptxt The input plaintext matrix to encrypt.
    /// @param[in] sk The secret key used for encryption.
    /// @param[out] ctxt The output ciphertext matrix.
    /// @param[in] r_ntt Whether the output ciphertext matrix should be left in
    /// the NTT (evaluation) domain.
    /// @details The device of the output follows from the input.
    /// @throws if the EnDecryptor has no parameters.
    /// @throws if the devices of ptxt and sk differ.
    /// @throws if the degrees of ptxt and sk differ.
    void encrypt(const IPtMatrix &ptxt, const ISecretKey &sk, ICtMatrix &ctxt,
                 bool r_ntt = false) const;

    /// @brief Decrypts a ciphertext matrix into a plaintext matrix using a
    /// secret key.
    /// @param[in] ctxt The input ciphertext matrix to decrypt.
    /// @param[in] sk The secret key used for decryption.
    /// @param[out] ptxt The output plaintext matrix.
    /// @details The device of the output follows from the input.
    /// @throws if the devices of ctxt and sk differ.
    /// @throws if the degrees of ctxt and sk differ.
    void decrypt(const ICtMatrix &ctxt, const ISecretKey &sk,
                 IPtMatrix &ptxt) const;

    /// @brief Gets the encryption parameters.
    /// @return The encryption parameters if set, otherwise std::nullopt.
    const std::optional<EncryptParams> &getParams() const { return params_; }

private:
    Pimpl impl;
    std::optional<EncryptParams> params_;
};

} // namespace heaan

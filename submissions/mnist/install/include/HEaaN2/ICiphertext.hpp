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

#include "HEaaN2/Encoding.hpp"
#include "HEaaN2/Encryption.hpp"
#include "HEaaN2/PolyRing.hpp"

namespace heaan {

/// @brief Interface for ciphertexts of various encryption types
struct HEAAN2_API ICiphertext : public DeviceSpecific {
    /// @brief Creates a new ciphertext instance
    /// @return The pointer to the ciphertext instance.
    static Ptr<ICiphertext> make(EncType enc_type = EncType::RLWE);

    virtual ~ICiphertext() = default;
    /// @brief Checks if the ciphertext is empty
    /// @return true if the associated polynomials are empty, false otherwise.
    virtual bool isEmpty() const = 0;
    /// @brief Creates a copy of the ciphertext
    /// @param dst The destination ciphertext to copy to.
    virtual void copyTo(ICiphertext &dst) const = 0;

    /// @brief Moves the ciphertext to another ciphertext
    /// @param dst The destination ciphertext to move to.
    virtual void moveTo(ICiphertext &dst) = 0;

    /// @brief Gets the EncType of the ciphertext.
    /// @return EncType of the ciphertext.
    virtual EncType type() const = 0;
    /// @brief Gets the PolyRing of the ciphertext.
    /// @return PolyRing of the ciphertext.
    /// @details PolyRing contains information about the modulus,
    /// polynomial degree, and NTT algorithm of the ciphertext.
    virtual PolyRing ring() const = 0;
    /// @brief Gets the Encoding of the ciphertext.
    /// @return Encoding of the ciphertext.
    /// @details Encoding contains information about the number of slots in
    /// logarithm, scale of the ciphertext, and whether the encoding is slot or
    /// coefficient.
    virtual Encoding encoding() const = 0;
    /// @brief Gets the Encryption of the ciphertext.
    /// @return Encryption of the ciphertext.
    /// @details Encryption contains information about the power, rank, and
    /// number of secrets used in the encryption.
    virtual Encryption encryption() const = 0;

    /// @brief Checks whether the ciphertext is in NTT (evaluation) form.
    /// @return true if the ciphertext polynomials are in NTT form, false
    /// otherwise.
    virtual bool isNTT() const = 0;
    /// @brief Gets the batch size of the ciphertext.
    /// @return The number of ciphertexts packed into this ciphertext.
    /// @details The batch size is 1 for RLWE ciphertexts and the number of
    /// packed ciphertexts for BatchRLWE ciphertexts.
    virtual u32 batchSize() const = 0;
};

} // namespace heaan

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
#include "HEaaN2/General.hpp"
#include "HEaaN2/PolyRing.hpp"

namespace heaan {

/// @brief Interface for plaintexts of various encoding types
struct HEAAN2_API IPlaintext : public DeviceSpecific {
    /// @brief Creates a new plaintext instance
    /// @return The pointer to the plaintext instance.
    static Ptr<IPlaintext> make(PtxtType type = PtxtType::NORMAL);

    virtual ~IPlaintext() = default;
    /// @brief Checks if the plaintext is empty
    /// @return true if the associated polynomial is empty, false otherwise.
    virtual bool isEmpty() const = 0;
    /// @brief Creates a copy of the plaintext
    /// @param dst The destination plaintext to copy to.
    virtual void copyTo(IPlaintext &dst) const = 0;
    /// @brief Moves the plaintext to another plaintext
    /// @param dst The destination plaintext to move to.
    virtual void moveTo(IPlaintext &dst) = 0;

    /// @brief Gets the type of the plaintext.
    /// @return The type of the plaintext.
    virtual PtxtType type() const = 0;
    /// @brief Gets the PolyRing of the plaintext.
    /// @return PolyRing of the plaintext.
    /// @details PolyRing contains information about the modulus,
    /// polynomial degree, and NTT algorithm of the plaintext.
    virtual PolyRing ring() const = 0;
    /// @brief Gets the Encoding of the plaintext.
    /// @return Encoding of the plaintext.
    /// @details Encoding contains information about the number of slots in
    /// logarithm, scale of the plaintext, and whether the encoding is slot or
    /// coefficient.
    virtual Encoding encoding() const = 0;

    /// @brief Checks whether the plaintext is in NTT (evaluation) form.
    /// @return true if the plaintext polynomial is in NTT form, false
    /// otherwise.
    virtual bool isNTT() const = 0;
    /// @brief Gets the batch size of the plaintext.
    /// @return The number of plaintexts packed into this plaintext.
    /// @details The batch size is 1 for NORMAL plaintexts and the number of
    /// packed plaintexts for BATCH plaintexts.
    virtual u32 batchSize() const = 0;
};

} // namespace heaan

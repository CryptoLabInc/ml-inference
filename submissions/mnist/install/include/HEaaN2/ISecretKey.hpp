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

namespace heaan {

/// @brief Interface for secret key
struct HEAAN2_API ISecretKey : public DeviceSpecific {
    ~ISecretKey() = default;
    /// @brief Checks if the secret key is empty.
    /// @return true if the associated coefficient vector is empty, false
    /// otherwise.
    virtual bool isEmpty() const = 0;
    /// @brief Creates a copy of the secret key.
    /// @param dst The destination secret key to copy to.
    virtual void copyTo(ISecretKey &dst) const = 0;
    /// @brief Moves the secret key to another secret key.
    /// @param dst The destination secret key to move to.
    virtual void moveTo(ISecretKey &dst) = 0;

    /// @brief Gets the log of the degree of the secret key.
    /// @return log of the degree of the secret key.
    virtual u32 logDegree() const = 0;
    /// @brief Gets the rank of the secret key.
    /// @return rank of the secret key.
    virtual u32 rank() const = 0;
    /// @brief Gets the number of secrets in the secret key.
    /// @return num_secrets of the secret key.
    virtual u32 numSecrets() const = 0;
};

} // namespace heaan

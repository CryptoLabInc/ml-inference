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

#include "HEaaN2/PolyRing.hpp"

namespace heaan {

/// @brief Interface for encryption key
struct HEAAN2_API IEncKey : public DeviceSpecific {
    ~IEncKey() = default;
    /// @brief Checks if the encryption key is empty.
    /// @return true if the associated polynomials are empty, false otherwise.
    virtual bool isEmpty() const = 0;
    /// @brief Creates a copy of the encryption key.
    /// @param dst The destination encryption key to copy to.
    virtual void copyTo(IEncKey &dst) const = 0;
    /// @brief Moves the encryption key to another encryption key.
    /// @param dst The destination encryption key to move to.
    virtual void moveTo(IEncKey &dst) = 0;

    /// @brief Gets the PolyRing of the encryption key.
    /// @return {mod, log_degree, ntt_alg} of the encryption key
    virtual PolyRing ring() const = 0;
};

} // namespace heaan

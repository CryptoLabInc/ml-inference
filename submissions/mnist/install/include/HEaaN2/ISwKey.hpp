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

#include "HEaaN2/Encryption.hpp"
#include "HEaaN2/GadgetDecomp.hpp"
#include "HEaaN2/General.hpp"
#include "HEaaN2/PolyRing.hpp"

namespace heaan {

/// @brief Interface for switching key
struct HEAAN2_API ISwKey : public DeviceSpecific {
    virtual ~ISwKey() = default;
    /// @brief Checks if the switching key is empty.
    /// @return true if the associated polynomials are empty, false otherwise.
    virtual bool isEmpty() const = 0;
    /// @brief Creates a copy of the switching key.
    /// @param dst The destination switching key to copy to.
    virtual void copyTo(ISwKey &dst) const = 0;
    /// @brief Moves the switching key to another switching key.
    /// @param dst The destination switching key to move to.
    virtual void moveTo(ISwKey &dst) = 0;

    /// @brief Gets the PolyRing of the switching key.
    /// @return {mod, log_degree, ntt_alg} of the switching key.
    virtual PolyRing ring() const = 0;
    /// @brief Gets the GadgetDecomp of the switching key.
    /// @return GadgetDecomp of the switching key.
    virtual GadgetDecomp gadgetDecomp() const = 0;
    /// @brief Gets the EncType of the switching key.
    /// @return EncType of the switching key.
    virtual EncType type() const = 0;
};
} // namespace heaan

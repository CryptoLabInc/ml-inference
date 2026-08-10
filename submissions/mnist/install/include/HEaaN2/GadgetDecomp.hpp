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

#include <vector>

namespace heaan {

/// @brief A struct representing the gadget decomposition parameters used for
/// key switching.
/// @details The gadget decomposition parameters consist of the maximum modulus
/// Qmax, the temporary modulus P and the list of size of each gadget block.
struct HEAAN2_API GadgetDecomp {
    /// @brief Constructs an empty GadgetDecomp.
    GadgetDecomp() = default;
    /// @brief Constructs a GadgetDecomp with the specified parameters.
    /// @param Qmax_ The maximum modulus.
    /// @param P_ The temporary modulus.
    /// @param decomp_ The list of size of each gadget block.
    GadgetDecomp(const PolyMod &Qmax_, const PolyMod &P_,
                 const std::vector<u32> &decomp_)
        : Qmax(Qmax_), P(P_), decomp(decomp_) {
        checkValidity();
    }

    /// @brief Returns true if two GadgetDecomp objects are equal.
    bool operator==(const GadgetDecomp &other) const;

    /// @brief Returns true if two GadgetDecomp objects are not equal.
    bool operator!=(const GadgetDecomp &other) const;

    /// @brief Checks whether the parameters are valid.
    /// @throws if any validity condition is violated.
    /// @details This function checks that:
    /// - The sum of decomp equals Qmax.size().
    /// - Qmax contains distinct moduli.
    /// - P contains distinc moduli.
    /// - Qmax and P do not share any modulus.
    void checkValidity() const;

    PolyMod Qmax, P;
    std::vector<u32> decomp;
};

} // namespace heaan

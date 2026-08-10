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

#include "HEaaN2/Export.hpp"
#include "HEaaN2/Levels.hpp"

namespace heaan::paramsUtils {

/// @brief Builder class for constructing customized Levels.
class HEAAN2_API LevelsBuilder {
public:
    /// @brief Constructs LevelsBuilder.
    LevelsBuilder();

    /// @brief Sets the polynomial ring.
    /// @param log_degree Log degree of the ring.
    /// @param poly_type The type of the polynomial (SIMPLE/GRAFTED).
    /// @param support_conj_inv If true, primes are chosen for degree
    /// 2^(log_degree+1), allowing conversion to conjugate-invariant subring
    /// representation. If false, primes are chosen for degree 2^log_degree.
    void setRing(u32 log_degree, PolyType poly_type,
                 bool support_conj_inv = true);

    /// @brief Initializes the current modulus using its bit-size.
    /// @param mod_bits Bit-size of the current modulus.
    /// @details Sets the current scale to match the current modulus size.
    void initMod(u32 mod_bits);

    /// @brief Sets the current modulus and scale.
    /// @param mod The modulus to set as the current modulus.
    /// @param scale The scale to set as the current scale.
    void setMod(const PolyMod &mod, Real128 scale);

    /// @brief Builds Levels above the current modulus.
    /// @param num_mults Number of additional levels to generate.
    /// @param bits Rescaling bit-size applied at each level. If bits == 0,
    /// the current scale is used as the rescaling size.
    /// @return Constructed Levels.
    /// @details The previous current modulus becomes the modulus of bottom
    /// level, and the current modulus is updated to the modulus of the top
    /// level.
    Levels buildAbove(u32 num_mults, u32 bits = 0);

    /// @brief Builds Levels below the current modulus.
    /// @param num_mults Number of additional levels to generate.
    /// @param bits Rescaling bit-size applied at each level. If bits == 0,
    /// the current scale is used as the rescaling size.
    /// @return Constructed Levels.
    /// @details The previous current modulus becomes the modulus of top level,
    /// and the current modulus is updated to the modulus of the bottom level.
    Levels buildBelow(u32 num_mults, u32 bits = 0);

    /// @brief Gets the current modulus.
    /// @return the current modulus.
    PolyMod currMod() const;

    /// @brief Gets the current scale.
    /// @return the current scale.
    Real128 currScale() const;

private:
    Pimpl impl;
};

} // namespace heaan::paramsUtils

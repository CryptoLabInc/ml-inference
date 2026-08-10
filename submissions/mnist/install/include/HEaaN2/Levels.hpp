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

/// @brief A struct representing the levels of the modulus chain
/// @details A levels consists of a chain of PolyMod and the corresponding chain
/// of scales.
struct HEAAN2_API Levels {
    /// @brief Creates an empty Levels.
    Levels() = default;
    /// @brief Constructs Levels with the specified modulus chain and top
    /// scale.
    /// @param mods_ The modulus chain.
    /// @param top_scale The top scale.
    /// @details The next scale is given by
    /// next_scale = current_scale ** 2 * (next_mod / current_mod).
    Levels(const std::vector<PolyMod> &mods_, Real128 top_scale);

    /// @brief Gets the size of the levels.
    /// @return The size of the levels.
    size_t size() const;
    /// @brief Gets the top level.
    /// @return The top level.
    u32 top() const;

    /// @brief Finds the level corresponding to the specified modulus.
    /// @param mod The modulus to find.
    /// @return The level corresponding to the specified modulus.
    /// @throws if the modulus is not found in the levels.
    u32 find(const PolyMod &mod) const;

    std::vector<PolyMod> mods;
    std::vector<Real128> scales;
};

} // namespace heaan

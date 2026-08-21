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

#include <variant>

namespace heaan {

/// @brief A struct representing a discrete Gaussian distribution for noise
/// generation.
/// @details The discrete Gaussian distribution is used to sample noise in
/// encryption and key generation.
struct HEAAN2_API DiscreteGaussian {
    /// @brief The standard deviation of the Gaussian distribution.
    double stdev = 0.0;
    /// @brief Default constructor.
    DiscreteGaussian() = default;
    /// @brief Constructs a zero-mean discrete Gaussian distribution with the
    /// given standard deviation.
    /// @param stdev The standard deviation.
    DiscreteGaussian(double stdev) : stdev(stdev) {}
};

/// @brief A variant type representing different noise distributions.
/// @details Currently supports discrete Gaussian distribution.
using Distribution = std::variant<DiscreteGaussian>;

} // namespace heaan

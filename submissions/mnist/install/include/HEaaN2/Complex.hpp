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

#include "HEaaN2/DataType.hpp"

namespace heaan::basic {
// Operations for 64-bits
Real HEAAN2_API pow(Real base, Real exponent);
Complex HEAAN2_API exp(Complex exponent);
Real HEAAN2_API abs(Real x);
Real HEAAN2_API abs(Complex x);
Real HEAAN2_API sqrt(Real x);
Complex HEAAN2_API conj(Complex x);
Real HEAAN2_API sin(Real x);
Real HEAAN2_API cos(Real x);
Real HEAAN2_API log2(Real x);
Real HEAAN2_API round(Real x);

// Operations for 128-bits
Real128 HEAAN2_API pow(Real128 base, Real128 exponent);
Complex128 HEAAN2_API exp(Complex128 exponent);
Real128 HEAAN2_API abs(Real128 x);
Real128 HEAAN2_API abs(Complex128 x);
Real128 HEAAN2_API sqrt(Real128 x);
Complex128 HEAAN2_API conj(Complex128 x);
Real128 HEAAN2_API sin(Real128 x);
Real128 HEAAN2_API cos(Real128 x);
Real128 HEAAN2_API log2(Real128 x);
Real128 HEAAN2_API round(Real128 x);

} // namespace heaan::basic

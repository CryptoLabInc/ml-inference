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

enum class HEAAN2_API PtxtType { NORMAL, BATCH };

enum class HEAAN2_API PtMatrixType { DEFAULT };

/// @brief Metadata regarding the encoding of a plaintext or a ciphertext.
/// @details
/// - log_slots: log of the number of slots
/// - scale: scaling factor
/// - dft: whether the encoding is slot or coefficient.
/// - If true, the corresponding object is slot-encoded
struct HEAAN2_API Encoding {
    u32 log_slots;
    Real128 scale;
    bool dft;
};

} // namespace heaan

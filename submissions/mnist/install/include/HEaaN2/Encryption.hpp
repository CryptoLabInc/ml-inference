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

enum class HEAAN2_API EncType { RLWE, BatchRLWE };

enum class HEAAN2_API CtMatrixType { BatchRLWE };

/// @brief Metadata regarding the encryption type of a ciphertext.
/// @details
/// - power: the power of the encryption. A freshly encrypted ciphertext has
/// power 1. The result of a tensor product has power 2.
/// - rank: The rank of the encryption. An RLWE instance has rank 1, whereas a
/// (M)LWE instance may have rank greater than 1.
/// - num_secrets: The number of secret keys involved in the encryption. An
/// RLWE/LWE/MLWE instance has num_secrets 1, whereas an MSRLWE instance has
/// num_secrets greater than 1.
struct HEAAN2_API Encryption {
    u32 power;
    u32 rank;
    u32 num_secrets;
};

} // namespace heaan

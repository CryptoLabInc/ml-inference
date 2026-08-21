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
#include "HEaaN2/Levels.hpp"

#include <string>

namespace heaan {

/// @brief Preset parameter identifiers
/// @details
/// 'S' stands for Somewhat Homomorphic, 'F' stands for Fully Homomorphic,
/// 'FR' stands for Fully Homomorphic for Real-numbers,
/// 'Gr' stands for Grafting, and 'Opt' stands for Optimized bootstrap.
/// All preset parameters are of log_degree 16.
/// All preset parameters are of 128-bit security level.
enum class HEAAN2_API PresetParamsId {
    S16 = 1, // enum begins from 1 to prevent empty value to be implicitly
             // converted to S16
    S16_Gr,
    F16,
    F16_Gr,
    F16Opt,
    F16Opt_Gr,
    FR16Opt_Gr,
    FR16_SOFTMAX,
    M12F16,
};

inline std::string to_string(PresetParamsId id) {
    switch (id) {
    case PresetParamsId::S16:
        return "S16";
    case PresetParamsId::S16_Gr:
        return "S16_Gr";
    case PresetParamsId::F16:
        return "F16";
    case PresetParamsId::F16_Gr:
        return "F16_Gr";
    case PresetParamsId::F16Opt:
        return "F16Opt";
    case PresetParamsId::F16Opt_Gr:
        return "F16Opt_Gr";
    case PresetParamsId::FR16Opt_Gr:
        return "FR16Opt_Gr";
    case PresetParamsId::FR16_SOFTMAX:
        return "FR16_SOFTMAX";
    case PresetParamsId::M12F16:
        return "M12F16";
    default:
        return "Unknown PresetParamsId";
    }
}

/// @brief Checks if the preset parameters support bootstrapping
/// @param id Preset parameter identifier
/// @return true if bootstrapping is supported, false otherwise.
bool HEAAN2_API isBootstrappable(PresetParamsId id);

/// @brief Gets the maximum level supported by the preset parameters
/// @param id Preset parameter identifier
/// @return Maximum level supported by the preset parameters
u32 HEAAN2_API getMaxLevel(PresetParamsId id);

/// @brief Gets the bottom modulus and bottom scale of the preset parameters
/// @param id Preset parameter identifier
/// @return A pair containing the bottom modulus and bottom scale of the preset
/// parameters
std::pair<PolyMod, Real128> HEAAN2_API getBottomLevels(PresetParamsId id);

/// @brief Gets the maximum log(PQ) of the preset parameters
/// @param id Preset parameter identifier
/// @return Maximum log(PQ) of the preset parameters
double HEAAN2_API getMaxLogPQ(PresetParamsId id);

/// @brief Gets the Hamming weight for the secret key generation of the preset
/// parameters
/// @param id Preset parameter identifier
/// @return Hamming weight for the secret key
u32 HEAAN2_API getHW(PresetParamsId id);

/// @brief Gets the Hamming weight used for the SSE optimization of the preset
/// parameters
/// @param id Preset parameter identifier
/// @return Hamming weight used for the SSE optimization of the preset
/// parameters
/// @details Regarding SSE (Sparse secret encapsulation), please
/// refer to https://eprint.iacr.org/2022/024.
u32 HEAAN2_API getSSEHW(PresetParamsId id);

/// @brief Prints information about the preset parameters
/// @param id Preset parameter identifier
/// @details Prints information about the preset parameters, including their
///         maximum level, maximum log(PQ), Hamming weight for secret key
///         generation, and Hamming weight for SSE optimization.
void HEAAN2_API printPresetParamsInfo(PresetParamsId id);

} // namespace heaan

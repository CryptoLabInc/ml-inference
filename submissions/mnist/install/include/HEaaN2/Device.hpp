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

#include <string>

namespace heaan {

/// @brief Enumeration of supported computation devices.
enum class HEAAN2_API Device {
    CPU,
    GPU_CUDA,
};

/// @brief Converts a Device enum value to its string representation.
/// @param device The Device enum value to convert.
/// @return A string representing the device type.
inline HEAAN2_API std::string to_string(Device device) {
    switch (device) {
    case Device::CPU:
        return "CPU";
    case Device::GPU_CUDA:
        return "GPU_CUDA";
    default:
        return "UNDEFINED";
    }
}

} // namespace heaan

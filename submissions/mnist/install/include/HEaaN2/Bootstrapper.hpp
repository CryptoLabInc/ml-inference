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

#include "HEaaN2/ICiphertext.hpp"
#include "HEaaN2/ISecretKey.hpp"
#include "HEaaN2/PresetParams.hpp"

namespace heaan {

/// @brief A struct holding pointers to bootstrapping keys.
struct HEAAN2_API BootKeyPtrs {
    /// @brief Constructs an empty BootKeyPtrs (used by deserialization).
    BootKeyPtrs();
    /// @brief Constructs a BootKeyPtrs with the specified preset
    /// parameters and secret key.
    /// @param id The preset parameters identifier.
    /// @param sk The secret key.
    BootKeyPtrs(PresetParamsId id, const ISecretKey &sk);

    /// @brief Gets the device of the BootKeyPtrs.
    Device device() const;

    Pimpl impl;
};

/// @brief Options for bootstrapping.
struct HEAAN2_API BootOptions {
    /// @brief Whether the target message is real-valued.
    /// @details If true, the bootstrapping will target real-valued messages.
    /// This results in a better latency for bootstrapping.
    /// Note that this option is not compatible with F16Opt and F16Opt_Gr.
    bool target_msg_real = false;
    /// @brief The range of input messages.
    /// @details The default message range is (-1.0, 1.0) for real and imaginary
    /// parts. If the input messages are known to be in a smaller range, setting
    /// this value accordingly may improve the bootstrapping precision.
    /// If the input messages are known to be in a larger range, you should
    /// enlarge this value to get correct results. Expect some precision loss
    /// when enlarging this value.
    Real input_msg_range = 1.0;
};

/// @brief A class for bootstrapping ciphertexts.
class HEAAN2_API Bootstrapper {
public:
    /// @brief Constructs a Bootstrapper with the given preset parameters and
    /// bootstrapping keys with the optional bootstrapping options.
    /// @param id The preset parameters identifier.
    /// @param bootkeys The bootstrapping keys.
    /// @param options The bootstrapping options.
    Bootstrapper(PresetParamsId id, const BootKeyPtrs &bootkeys,
                 const BootOptions &options = {});

    /// @brief Warms up the bootstrapper.
    /// @details This function precomputes necessary data for bootstrapping.
    void warmup() const;
    /// @brief Bootstraps the given ciphertext.
    /// @param op The ciphertext to be bootstrapped.
    /// @throws if op is not forwardNTTed.
    /// @details The level of the input ciphertext must be zero.
    /// If one wants to bootstrap a ciphertext with non-zero level,
    /// first reduce its level to zero using HomEval::levelDownTo.
    /// @details The device of the input ciphertext must match the device of the
    /// bootstrapper.
    void bootstrap(ICiphertext &op) const;

    /// @brief Gets the device of the associated BootKeyPtrs.
    /// @return The device of the associated BootKeyPtrs.
    Device device() const;

private:
    Pimpl impl;
    Device device_;
    PresetParamsId id_;
};

} // namespace heaan

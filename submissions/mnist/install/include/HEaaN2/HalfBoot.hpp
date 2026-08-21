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

#include "HEaaN2/Bootstrapper.hpp"
#include "HEaaN2/ICiphertext.hpp"
#include "HEaaN2/Levels.hpp"
#include "HEaaN2/PresetParams.hpp"

namespace heaan {

/// @brief A struct holding pointers to half-bootstrapping keys.
struct HEAAN2_API HalfBootKeyPtrs {
    /// @brief Constructs an empty HalfBootKeyPtrs (used by deserialization).
    HalfBootKeyPtrs();
    /// @brief Constructs a HalfBootKeyPtrs with the specified preset
    /// parameters and secret key.
    /// @param id The preset parameters identifier.
    /// @param sk The secret key.
    HalfBootKeyPtrs(PresetParamsId id, const ISecretKey &sk);

    /// @brief Gets the device of the HalfBootKeyPtrs.
    Device device() const;

    Pimpl impl;
};

/// @brief A class evaluating the HalfBoot operation on ciphertexts.
/// @details Half-bootstrapping performs the ModRaise + CoeffToSlot + EvalMod.
class HEAAN2_API HalfBoot {
public:
    /// @brief Constructs a HalfBoot with the given preset parameters
    /// @param id The preset parameters identifier.
    /// @param bootkeys The bootstrapping keys
    /// @param options The bootstrapping options. The target_msg_real option is
    /// not supported by half-bootstrapping and must be false.
    /// @throws if the preset does not support bootstrapping, or if
    /// options.target_msg_real is true.
    HalfBoot(PresetParamsId id, const HalfBootKeyPtrs &bootkeys,
             const BootOptions &options = {});

    /// @brief Evaluates the HalfBoot operation on the given ciphertext.
    /// @param op The ciphertext to be half-bootstrapped, in place.
    /// @details The device of the input ciphertext must match the device of the
    /// associated bootstrapping keys.
    void eval(ICiphertext &op) const;

    /// @brief Gets the device of the associated bootstrapping keys.
    /// @return The device of the associated bootstrapping keys.
    Device device() const;

private:
    Pimpl impl;
    Device device_;
    PresetParamsId id_;
};

} // namespace heaan

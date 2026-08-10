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
#include "HEaaN2/Levels.hpp"

namespace heaan {

/// @brief Parameters for the CoeffToSlot operation.
/// @details
/// - log_slots: log of the number of slots of the input ciphertext.
/// - decomp_steps: The steps of the decomposition.
/// - decomp_num_bs: The number of baby steps used in each decomposition step.
/// - poly_type: The type of the polynomial (SIMPLE/GRAFTED), which must match
/// the representation of the ciphertexts to be evaluated on.
/// - levels: The levels for the evaluation. The levels should be sufficient
/// for the decomposition, i.e. at least decomp_steps.size() + 1 levels.
/// - use_min_keys: Whether to use a minimal set of rotation keys.
struct HEAAN2_API CoeffToSlotParams {
    u32 log_slots;
    std::vector<u32> decomp_steps;
    std::vector<u32> decomp_num_bs;
    PolyType poly_type = PolyType::SIMPLE;
    Levels levels;
    bool use_min_keys = false;

    /// @brief Sets log of the number of slots.
    CoeffToSlotParams &setLogSlots(u32 log_slots);
    /// @brief Sets the type of the polynomial (SIMPLE/GRAFTED).
    CoeffToSlotParams &setPolyType(PolyType poly_type);
    /// @brief Sets the decomposition.
    /// @param decomp_steps The steps of the decomposition.
    /// @param decomp_num_bs The number of baby steps for each step.
    /// @details The two vectors must have the same size.
    CoeffToSlotParams &setDecomp(const std::vector<u32> &decomp_steps,
                                 const std::vector<u32> &decomp_num_bs);
    /// @brief Sets the levels for the evaluation.
    CoeffToSlotParams &setLevels(const Levels &levels);
    /// @brief Sets whether to use a minimal set of rotation keys.
    CoeffToSlotParams &setKeyStrategy(bool use_min_keys = false);

    /// @brief Checks the validity of the CoeffToSlotParams.
    /// @throws if the decomposition vectors have different sizes, or if the
    /// levels are not sufficient for the decomposition.
    void checkValidity() const;
};

/// @brief A struct holding pointers to the rotation keys required for the
/// CoeffToSlot operation.
struct HEAAN2_API CoeffToSlotKeyPtrs {
    /// @brief Constructs an empty CoeffToSlotKeyPtrs (used by
    /// deserialization).
    CoeffToSlotKeyPtrs();
    /// @brief Constructs a CoeffToSlotKeyPtrs with the specified parameters
    /// and secret key.
    /// @param params The CoeffToSlot parameters.
    /// @param sk The secret key.
    /// @details The device of the generated keys follows from the input
    /// secret key.
    /// @throws if there is no way to generate the keys with 128-bit security
    /// for the given secret key.
    CoeffToSlotKeyPtrs(const CoeffToSlotParams &params, const ISecretKey &sk);

    /// @brief Gets the device of the CoeffToSlotKeyPtrs.
    Device device() const;

    Pimpl impl;
};

/// @brief A class evaluating the CoeffToSlot operation on ciphertexts using
/// the standard decomposition parameterized by CoeffToSlotParams.
class HEAAN2_API CoeffToSlot {
public:
    /// @brief Constructs a CoeffToSlot with the given parameters and rotation
    /// keys.
    /// @param params The CoeffToSlot parameters.
    /// @param cts_keys The rotation keys for the evaluation.
    CoeffToSlot(const CoeffToSlotParams &params,
                const CoeffToSlotKeyPtrs &cts_keys);

    /// @brief Evaluates the CoeffToSlot operation on the given ciphertext.
    /// @param op The ciphertext to be transformed, in place.
    /// @param multiplier The multiplier to be applied to the output
    /// ciphertext.
    /// @details The device of the input ciphertext must match the device of
    /// the associated CoeffToSlotKeyPtrs.
    void eval(ICiphertext &op, Real128 multiplier = 1.0) const;

    /// @brief Gets the device of the associated CoeffToSlotKeyPtrs.
    /// @return The device of the associated CoeffToSlotKeyPtrs.
    Device device() const;

private:
    Pimpl impl;
    Device device_;
};

} // namespace heaan

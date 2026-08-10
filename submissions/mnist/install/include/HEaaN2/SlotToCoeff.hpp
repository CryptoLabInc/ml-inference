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
#include "HEaaN2/PresetParams.hpp"

namespace heaan {

/// @brief Parameters for the SlotToCoeff operation.
/// @details
/// - log_slots: log of the number of slots.
/// - decomp_steps: The steps of the decomposition.
/// - decomp_num_bs: The number of baby steps used in each decomposition step.
/// - poly_type: The type of the polynomial (SIMPLE/GRAFTED), which must match
/// the representation of the ciphertexts to be evaluated on.
/// - levels: The levels for the evaluation. The levels should be sufficient
/// for the decomposition, i.e. at least decomp_steps.size() + 1 levels.
/// - use_min_keys: Whether to use a minimal set of rotation keys.
struct HEAAN2_API SlotToCoeffParams {
    u32 log_slots;
    std::vector<u32> decomp_steps;
    std::vector<u32> decomp_num_bs;
    PolyType poly_type = PolyType::SIMPLE;
    Levels levels;
    bool use_min_keys = false;

    /// @brief Sets log of the number of slots.
    SlotToCoeffParams &setLogSlots(u32 log_slots);
    /// @brief Sets the type of the polynomial (SIMPLE/GRAFTED).
    SlotToCoeffParams &setPolyType(PolyType poly_type);
    /// @brief Sets the decomposition.
    /// @param decomp_steps The steps of the decomposition.
    /// @param decomp_num_bs The number of baby steps for each step.
    /// @details The two vectors must have the same size.
    SlotToCoeffParams &setDecomp(const std::vector<u32> &decomp_steps,
                                 const std::vector<u32> &decomp_num_bs);
    /// @brief Sets the levels for the evaluation.
    SlotToCoeffParams &setLevels(const Levels &levels);
    /// @brief Sets whether to use a minimal set of rotation keys.
    SlotToCoeffParams &setKeyStrategy(bool use_min_keys = false);

    /// @brief Checks the validity of the SlotToCoeffParams.
    /// @throws if the decomposition vectors have different sizes, or if the
    /// levels are not sufficient for the decomposition.
    void checkValidity() const;
};

/// @brief A struct holding pointers to the rotation keys required for the
/// SlotToCoeff operation.
struct HEAAN2_API SlotToCoeffKeyPtrs {
    /// @brief Constructs an empty SlotToCoeffKeyPtrs (used by
    /// deserialization).
    SlotToCoeffKeyPtrs();
    /// @brief Constructs a SlotToCoeffKeyPtrs with the specified parameters
    /// and secret key.
    /// @param params The SlotToCoeff parameters.
    /// @param sk The secret key.
    /// @details The device of the generated keys follows from the input
    /// secret key.
    /// @throws if there is no way to generate the keys with 128-bit security
    /// for the given secret key.
    SlotToCoeffKeyPtrs(const SlotToCoeffParams &params, const ISecretKey &sk);

    /// @brief Gets the device of the SlotToCoeffKeyPtrs.
    Device device() const;

    Pimpl impl;
};

/// @brief A class evaluating the SlotToCoeff operation on ciphertexts using
/// the standard decomposition parameterized by SlotToCoeffParams.
class HEAAN2_API SlotToCoeff {
public:
    /// @brief Constructs a SlotToCoeff with the given parameters and rotation
    /// keys.
    /// @param params The SlotToCoeff parameters.
    /// @param stc_keys The rotation keys for the evaluation.
    SlotToCoeff(const SlotToCoeffParams &params,
                const SlotToCoeffKeyPtrs &stc_keys);

    /// @brief Evaluates the SlotToCoeff operation on the given ciphertext.
    /// @param op The ciphertext to be transformed, in place.
    /// @param multiplier The multiplier to be applied to the output
    /// ciphertext.
    /// @details The device of the input ciphertext must match the device of
    /// the associated SlotToCoeffKeyPtrs.
    void eval(ICiphertext &op, Real128 multiplier = 1.0) const;

    /// @brief Evaluates the SlotToCoeff operation and lands the output on the
    /// given modulus and scale.
    /// @param op The ciphertext to be transformed, in place.
    /// @param mod_to The modulus the output lands on, at or below the output
    /// level of the levels.
    /// @param scale_to The scale the output lands on.
    /// @param r_ntt Whether the output ciphertext should be left in the NTT
    /// (evaluation) domain.
    /// @param multiplier The multiplier to be applied to the output
    /// ciphertext.
    /// @details The device of the input ciphertext must match the device of
    /// the associated SlotToCoeffKeyPtrs.
    void eval(ICiphertext &op, const PolyMod &mod_to, Real128 scale_to,
              bool r_ntt, Real128 multiplier = 1.0) const;

    /// @brief Gets the device of the associated SlotToCoeffKeyPtrs.
    /// @return The device of the associated SlotToCoeffKeyPtrs.
    Device device() const;

private:
    Pimpl impl;
    Device device_;
};

/// @brief Returns the SlotToCoeff parameters used by the bootstrapping of the
/// given preset.
/// @param id The preset parameters identifier.
/// @return The SlotToCoeffParams of the preset's bootstrapping.
/// @throws if the preset does not support bootstrapping.
SlotToCoeffParams HEAAN2_API getSlotToCoeffParams(PresetParamsId id);

/// @brief Returns the multiplier to pass to SlotToCoeff::eval when composing
/// SlotToCoeff with HalfBoot, so the result reaches full-bootstrap precision.
/// @param id The preset parameters identifier.
/// @param input_scale The scale of the SlotToCoeff input ciphertext, which must
/// sit at the top of getSlotToCoeffParams(id).levels (use the input
/// ciphertext's encoding().scale).
/// @param input_msg_range The range of the input messages (matches
/// BootOptions::input_msg_range); defaults to 1.0.
/// @return The SlotToCoeff multiplier computed by the preset's bootstrapping.
/// @throws if the preset does not support bootstrapping.
Real128 HEAAN2_API getSlotToCoeffMultiplier(PresetParamsId id,
                                            Real128 input_scale,
                                            Real input_msg_range = 1.0);

} // namespace heaan

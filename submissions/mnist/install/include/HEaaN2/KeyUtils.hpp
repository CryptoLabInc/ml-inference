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

#include "HEaaN2/ISwKey.hpp"
#include "HEaaN2/Levels.hpp"

#include <map>

namespace heaan {

// holds & owns a key with exclusive ownership
using KeyPtr = Ptr<ISwKey>;

/// @brief A struct holding multiple rotation keys sharing the same GadgetDecomp
/// and PolyRing
/// @details The keys are indexed by the rotation step. Indices are equivalent
/// modulo the number of slots.
struct HEAAN2_API RotKeyPtrs : public std::map<i32, KeyPtr> {
    RotKeyPtrs() = default;
    ~RotKeyPtrs() = default;
    RotKeyPtrs(const RotKeyPtrs &) = delete;
    RotKeyPtrs &operator=(const RotKeyPtrs &) = delete;
    RotKeyPtrs(RotKeyPtrs &&) = default;
    RotKeyPtrs &operator=(RotKeyPtrs &&) = default;

    /// @brief Checks if the RotKeyPtrs contains a key for the specified
    /// rotation step.
    /// @param step The rotation step to check.
    /// @param log_slots log of the number of slots, used to determine the
    /// equivalence of rotation steps.
    bool has(i32 step, u32 log_slots) const;
    /// @brief Gets the key for the specified rotation step.
    /// @param step The rotation step to get the key for.
    /// @param log_slots log of the number of slots, used to determine the
    /// equivalence of rotation steps.
    /// @return The key for the specified rotation step.
    /// @throws if the key for the specified rotation step does not exist.
    const KeyPtr &get(i32 step, u32 log_slots) const;
    /// @brief Gets the encryption type of the keys.
    /// @return The encryption type of the keys.
    /// @throws if the RotKeyPtrs is empty.
    /// @throws if the keys in the RotKeyPtrs have different encryption types.
    EncType type() const;
    /// @brief Gets the polynomial ring of the keys.
    /// @return The polynomial ring of the keys.
    /// @throws if the RotKeyPtrs is empty.
    /// @throws if the keys in the RotKeyPtrs have different polynomial rings.
    PolyRing ring() const;
    /// @brief Gets the gadget decomposition of the keys.
    /// @return The gadget decomposition parameters of the keys.
    /// @throws if the RotKeyPtrs is empty.
    /// @throws if the keys in the RotKeyPtrs have different gadget
    /// decompositions.
    GadgetDecomp gadgetDecomp() const;

    /// @brief Gets the device of the keys.
    /// @return Device of the keys.
    /// @throws if the RotKeyPtrs is empty.
    /// @throws if the keys in the RotKeyPtrs are on different devices.
    Device device() const;
    /// @brief Copies the keys to the specified device.
    /// @param device Device where the keys are copied to.
    /// @throws if the RotKeyPtrs is empty.
    void to(Device device);
};

/// @brief A struct holding multiple relinearization keys with varying
/// GadgetDecomp and PolyRing
struct HEAAN2_API KeyPtrBundle {
    KeyPtrBundle();
    ~KeyPtrBundle() = default;
    KeyPtrBundle(const KeyPtrBundle &) = delete;
    KeyPtrBundle &operator=(const KeyPtrBundle &) = delete;
    KeyPtrBundle(KeyPtrBundle &&) = default;
    KeyPtrBundle &operator=(KeyPtrBundle &&) = default;

    /// @brief Checks if the KeyPtrBundle is empty.
    /// @return true if the KeyPtrBundle is empty, false otherwise.
    bool isEmpty() const;

    /// @brief Gets the number of keys in the KeyPtrBundle.
    /// @return The number of keys in the KeyPtrBundle.
    size_t size() const;
    /// @brief Checks if the KeyPtrBundle contains keys for the specified
    /// polynomial ring.
    /// @param ring The polynomial ring to check.
    /// @return true if the KeyPtrBundle contains keys for the specified
    /// polynomial ring, false otherwise.
    bool isCompatibleWith(const PolyRing &ring) const;
    /// @brief Checks if the KeyPtrBundle contains keys compatible with the
    /// specified levels.
    /// @param levels The levels to check compatibility with.
    /// @return true if the KeyPtrBundle contains keys compatible with the
    /// specified levels, false otherwise.
    /// @details if the KeyPtrBundle contains a single key, it is compatible if
    /// its modulus is a prefix of the modulus of each level. If the
    /// KeyPtrBundle contains multiple keys, then the number of keys must match
    /// the number of levels, and each key must be
    /// compatible with the corresponding level (excluding the base level).
    bool isCompatibleWith(const Levels &levels) const;

    /// @brief Gets the device of the keys in the KeyPtrBundle.
    /// @return Device of the keys in the KeyPtrBundle.
    /// @throws if the KeyPtrBundle is empty.
    /// @throws if the keys in the KeyPtrBundle are on different devices.
    Device device() const;
    /// @brief Copies the keys in the KeyPtrBundle to the specified device.
    /// @param device Device where the keys are copied to.
    /// @throws if the KeyPtrBundle is empty.
    void to(Device device);

    Pimpl impl;
};

} // namespace heaan

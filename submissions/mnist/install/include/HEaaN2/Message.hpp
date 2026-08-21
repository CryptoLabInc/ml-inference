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

/// @brief A struct for complex vectors
/// @tparam T Number type (Complex or Complex128)
template <typename T> struct HEAAN2_API MessageT : public DeviceSpecific {
    static_assert(std::is_same_v<T, Complex> || std::is_same_v<T, Complex128>,
                  "MessageT<T> only supports Complex and Complex128");
    /// @brief Creates an empty Message.
    MessageT();
    /// @brief Constructs a Message with the specified log_slots and device
    /// @param log_slots log of the number of slots
    /// @param device Device where the message is stored
    /// @details Creates a Message with (1 << log_slots) slots reserved on the
    /// specified device.
    MessageT(u32 log_slots, Device device = Device::CPU);
    /// @brief Checks if the message is empty
    /// @return true if the underlying data is empty, false otherwise.
    bool isEmpty() const;

    /// @brief Creates a copy of the message
    /// @return The copied message.
    MessageT<T> copy() const;

    /// @brief Gets log of the number of slots
    /// @return log_slots of the message.
    u32 logSlots() const;
    /// @brief Accesses the element at the specified index
    /// @param idx Index of the element to access
    /// @return a reference to the element at the specified index.
    /// @throws if the message is empty or on a device other than CPU.
    /// @details There is no bounds checking for the index. Prefer data() for
    /// bulk access.
    T &operator[](size_t idx);
    /// @brief Accesses the element at the specified index
    /// @param idx Index of the element to access
    /// @return a constant reference to the element at the specified index.
    /// @throws if the message is empty or on a device other than CPU.
    /// @details There is no bounds checking for the index. Prefer data() for
    /// bulk access.
    const T &operator[](size_t idx) const;

    /// @brief Returns a pointer to the underlying contiguous slot data.
    /// @return a pointer to the first element.
    /// @throws if the message is empty.
    /// @details Prefer this over operator[] for bulk read/write.
    T *data();
    /// @brief Returns a pointer to the underlying contiguous slot data.
    /// @return a constant pointer to the first element.
    /// @throws if the message is empty.
    /// @details Prefer this over operator[] for bulk read/write.
    const T *data() const;

    /// @brief Gets the device of the message
    /// @return Device of the message.
    Device device() const override;
    /// @brief Copies the message to the specified device.
    /// @param device Device where the message is copied to.
    void to(Device device) override;

    Pimpl impl;
};

using Message = MessageT<Complex>;
using Message128 = MessageT<Complex128>;

} // namespace heaan

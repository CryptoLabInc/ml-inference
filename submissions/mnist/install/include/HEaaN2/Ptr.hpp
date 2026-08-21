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

#include <memory>
#include <utility>

namespace heaan {

template <typename T> using WeakPtr = std::weak_ptr<T>;

// A custom smart pointer that acts like unique_ptr
// but allows for weak references.
template <class T> class HEAAN2_API Ptr {
public:
    // can be made with args
    template <class... U> static Ptr make(U &&...args) {
        Ptr p;
        p.ptr_ = std::make_shared<T>(std::forward<U>(args)...);
        return p;
    }

    Ptr() = default;
    ~Ptr() = default;

    // cannot be copy-constructed/assigned.
    Ptr(Ptr const &other) = delete;
    Ptr &operator=(const Ptr &other) = delete;

    // can be move-constructed/assigned with optional upcasting
    template <class U, class = std::enable_if_t<std::is_base_of_v<T, U>>>
    Ptr(Ptr<U> &&other) noexcept : ptr_{std::move(other.ptr_)} {}
    template <class U, class = std::enable_if_t<std::is_base_of_v<T, U>>>
    Ptr &operator=(Ptr<U> &&other) noexcept {
        ptr_ = std::move(other.ptr_);
        return *this;
    }

    // propagates constness
    T &operator*() { return *ptr_; }
    T const &operator*() const { return *ptr_; }

    T *operator->() { return ptr_.get(); }
    T const *operator->() const { return ptr_.get(); }

    // check if the pointer is valid
    explicit operator bool() const { return static_cast<bool>(ptr_); }

    // get weak reference of the content
    WeakPtr<const T> weak() const { return ptr_; }
    WeakPtr<T> weak() { return ptr_; }

private:
    template <class U> friend class Ptr;

    std::shared_ptr<T> ptr_;
};

} // namespace heaan

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

#include <memory>

namespace heaan {

struct HEAAN2_API Pimpl : public std::unique_ptr<void, void (*)(void *)> {
    using std::unique_ptr<void, void (*)(void *)>::unique_ptr;

    template <typename T> const T &cast() const {
        return *static_cast<const T *>(this->get());
    }

    template <typename T> T &cast() { return *static_cast<T *>(this->get()); }
};

} // namespace heaan

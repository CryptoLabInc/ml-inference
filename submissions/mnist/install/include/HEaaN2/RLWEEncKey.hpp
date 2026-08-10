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

#include "HEaaN2/IEncKey.hpp"

namespace heaan {

struct HEAAN2_API RLWEEncKey : public IEncKey {
    RLWEEncKey();
    ~RLWEEncKey() = default;
    bool isEmpty() const override;
    void copyTo(IEncKey &dst) const override;
    void moveTo(IEncKey &dst) override;

    PolyRing ring() const override;

    Device device() const override;
    void to(Device device) override;

    Pimpl impl;
};

} // namespace heaan

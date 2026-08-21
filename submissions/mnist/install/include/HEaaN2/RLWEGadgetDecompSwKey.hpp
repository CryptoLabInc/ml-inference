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

namespace heaan {

struct HEAAN2_API RLWEGadgetDecompSwKey : public ISwKey {
    RLWEGadgetDecompSwKey();
    bool isEmpty() const override;
    ~RLWEGadgetDecompSwKey() override = default;

    void copyTo(ISwKey &dst) const override;
    void moveTo(ISwKey &dst) override;

    PolyRing ring() const override;
    GadgetDecomp gadgetDecomp() const override;
    EncType type() const override;

    Device device() const override;
    void to(Device device) override;

    Pimpl impl;
};

} // namespace heaan

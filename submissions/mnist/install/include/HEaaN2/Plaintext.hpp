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

#include "HEaaN2/IPlaintext.hpp"

namespace heaan {

struct HEAAN2_API Plaintext : public IPlaintext {
    Plaintext();
    ~Plaintext() = default;
    bool isEmpty() const override;
    void copyTo(IPlaintext &dst) const override;
    void moveTo(IPlaintext &dst) override;

    PtxtType type() const override;
    PolyRing ring() const override;
    Encoding encoding() const override;
    bool isNTT() const override;
    u32 batchSize() const override;

    Device device() const override;
    void to(Device device) override;

    Pimpl impl;
};

} // namespace heaan

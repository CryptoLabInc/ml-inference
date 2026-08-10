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

namespace heaan {

struct HEAAN2_API RLWECiphertext : public ICiphertext {
    RLWECiphertext();
    ~RLWECiphertext() = default;
    bool isEmpty() const override;
    void copyTo(ICiphertext &dst) const override;
    void moveTo(ICiphertext &dst) override;

    EncType type() const override;
    PolyRing ring() const override;
    Encoding encoding() const override;
    Encryption encryption() const override;
    bool isNTT() const override;
    u32 batchSize() const override;

    Device device() const override;
    void to(Device device) override;

    Pimpl impl;
};

} // namespace heaan

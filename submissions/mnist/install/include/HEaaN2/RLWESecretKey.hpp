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

#include "HEaaN2/ISecretKey.hpp"

namespace heaan {

struct HEAAN2_API RLWESecretKey : public ISecretKey {
    RLWESecretKey();
    ~RLWESecretKey() override = default;
    bool isEmpty() const override;
    void copyTo(ISecretKey &dst) const override;
    void moveTo(ISecretKey &dst) override;

    u32 logDegree() const override;
    u32 rank() const override;
    u32 numSecrets() const override;

    Device device() const override;
    void to(Device device) override;

    Pimpl impl;
};

} // namespace heaan

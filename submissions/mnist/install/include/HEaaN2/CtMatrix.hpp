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

#include "HEaaN2/ICtMatrix.hpp"

namespace heaan {

struct HEAAN2_API CtMatrix : public ICtMatrix {
    CtMatrix();
    ~CtMatrix() = default;
    bool isEmpty() const override;
    void copyTo(ICtMatrix &dst) const override;
    void moveTo(ICtMatrix &dst) override;

    void copyTo(ICiphertext &dst) const override;
    void moveTo(ICiphertext &dst) override;
    void copyFrom(const ICiphertext &src, u32 rows, u32 cols) override;
    void moveFrom(ICiphertext &src, u32 rows, u32 cols) override;

    CtMatrixType type() const override;
    PolyRing ring() const override;
    Encoding encoding() const override;
    u32 rows() const override;
    u32 cols() const override;
    u32 numSecrets() const override;
    bool isNTT() const override;

    Device device() const override;
    void to(Device device) override;

    Pimpl impl;
};

} // namespace heaan

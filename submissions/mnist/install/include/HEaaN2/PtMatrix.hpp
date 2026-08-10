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

#include "HEaaN2/IPtMatrix.hpp"

namespace heaan {

struct HEAAN2_API PtMatrix : public IPtMatrix {
    PtMatrix();
    ~PtMatrix() = default;
    bool isEmpty() const override;
    void copyTo(IPtMatrix &dst) const override;
    void moveTo(IPtMatrix &dst) override;

    void copyTo(IPlaintext &dst) const override;
    void moveTo(IPlaintext &dst) override;
    void copyFrom(const IPlaintext &src, u32 rows, u32 cols) override;
    void moveFrom(IPlaintext &src, u32 rows, u32 cols) override;

    PtMatrixType type() const override;
    PolyRing ring() const override;
    Encoding encoding() const override;
    u32 rows() const override;
    u32 cols() const override;
    bool isNTT() const override;

    Device device() const override;
    void to(Device device) override;

    Pimpl impl;
};

} // namespace heaan

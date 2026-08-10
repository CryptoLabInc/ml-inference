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

#include "HEaaN2/Encoding.hpp"
#include "HEaaN2/General.hpp"
#include "HEaaN2/PolyRing.hpp"

namespace heaan {

struct IPlaintext;

/// @brief Interface for plaintext matrices.
struct HEAAN2_API IPtMatrix : public DeviceSpecific {
    /// @brief Creates a new plaintext matrix instance.
    /// @return The pointer to the plaintext matrix instance.
    static Ptr<IPtMatrix> make();

    virtual ~IPtMatrix() = default;
    /// @brief Checks if the plaintext matrix is empty.
    /// @return true if no encoded data is stored, false otherwise.
    virtual bool isEmpty() const = 0;
    /// @brief Creates a copy of the plaintext matrix.
    /// @param dst The destination plaintext matrix to copy to.
    virtual void copyTo(IPtMatrix &dst) const = 0;
    /// @brief Moves the plaintext matrix to another plaintext matrix.
    /// @param dst The destination plaintext matrix to move to.
    virtual void moveTo(IPtMatrix &dst) = 0;

    /// @brief Copies the data of the matrix into a batched plaintext.
    /// @param dst The destination plaintext; must be of PtxtType::BATCH.
    /// @throws if the matrix is empty or dst is not a BATCH plaintext.
    virtual void copyTo(IPlaintext &dst) const = 0;
    /// @brief Moves the data of the matrix into a batched plaintext.
    /// @param dst The destination plaintext; must be of PtxtType::BATCH.
    /// @throws if the matrix is empty or dst is not a BATCH plaintext.
    virtual void moveTo(IPlaintext &dst) = 0;
    /// @brief Fills the matrix from a batched plaintext.
    /// @param src  The source plaintext; must be of PtxtType::BATCH.
    /// @param rows The number of rows of the resulting matrix.
    /// @param cols The number of columns of the resulting matrix.
    /// @details Any previous content of the matrix is replaced.
    virtual void copyFrom(const IPlaintext &src, u32 rows, u32 cols) = 0;
    /// @brief Fills the matrix from a batched plaintext without copying.
    /// @param src  The source plaintext; must be of PtxtType::BATCH. Its
    /// polynomial data is moved into the matrix, leaving it empty.
    /// @param rows The number of rows of the resulting matrix.
    /// @param cols The number of columns of the resulting matrix.
    /// @details Any previous content of the matrix is replaced.
    virtual void moveFrom(IPlaintext &src, u32 rows, u32 cols) = 0;

    /// @brief Gets the type of the plaintext matrix.
    /// @return The type of the plaintext matrix.
    virtual PtMatrixType type() const = 0;
    /// @brief Gets the PolyRing of the plaintext matrix.
    /// @return PolyRing of the plaintext matrix.
    /// @details PolyRing contains information about the modulus,
    /// polynomial degree, and NTT algorithm of the plaintext matrix.
    virtual PolyRing ring() const = 0;
    /// @brief Gets the Encoding of the plaintext matrix.
    /// @return Encoding of the plaintext matrix.
    /// @details Encoding contains information about the number of slots in
    /// logarithm, scale of the plaintext matrix, and whether the encoding is
    /// slot or coefficient.
    virtual Encoding encoding() const = 0;
    /// @brief Gets the number of logical rows.
    /// @return The row count.
    virtual u32 rows() const = 0;
    /// @brief Gets the number of logical columns.
    /// @return The column count.
    virtual u32 cols() const = 0;

    /// @brief Checks whether the plaintext matrix is in NTT (evaluation) form.
    /// @return true if the plaintext matrix polynomials are in NTT form, false
    /// otherwise.
    virtual bool isNTT() const = 0;
};

} // namespace heaan

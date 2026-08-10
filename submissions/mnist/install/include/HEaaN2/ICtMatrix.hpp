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
#include "HEaaN2/Encryption.hpp"
#include "HEaaN2/PolyRing.hpp"

namespace heaan {

struct ICiphertext;

/// @brief Interface for ciphertext matrices.
struct HEAAN2_API ICtMatrix : public DeviceSpecific {
    /// @brief Creates a new ciphertext matrix instance.
    /// @return The pointer to the ciphertext matrix instance.
    static Ptr<ICtMatrix> make(CtMatrixType type = CtMatrixType::BatchRLWE);

    virtual ~ICtMatrix() = default;
    /// @brief Checks if the ciphertext matrix is empty.
    /// @return true if no encrypted data is stored, false otherwise.
    virtual bool isEmpty() const = 0;
    /// @brief Creates a copy of the ciphertext matrix.
    /// @param dst The destination ciphertext matrix to copy to.
    virtual void copyTo(ICtMatrix &dst) const = 0;
    /// @brief Moves the ciphertext matrix to another ciphertext matrix.
    /// @param dst The destination ciphertext matrix to move to.
    virtual void moveTo(ICtMatrix &dst) = 0;

    /// @brief Copies the data of the matrix into a batched ciphertext.
    /// @param dst The destination ciphertext; must be of EncType::BatchRLWE.
    /// @throws if the matrix is empty or dst is not a BatchRLWE ciphertext.
    virtual void copyTo(ICiphertext &dst) const = 0;
    /// @brief Moves the data of the matrix into a batched ciphertext.
    /// @param dst The destination ciphertext; must be of EncType::BatchRLWE.
    /// @throws if the matrix is empty or dst is not a BatchRLWE ciphertext.
    virtual void moveTo(ICiphertext &dst) = 0;
    /// @brief Fills the matrix from a batched ciphertext.
    /// @param src  The source ciphertext; must be of EncType::BatchRLWE.
    /// @param rows The number of rows of the resulting matrix.
    /// @param cols The number of columns of the resulting matrix.
    /// @details Any previous content of the matrix is replaced.
    virtual void copyFrom(const ICiphertext &src, u32 rows, u32 cols) = 0;
    /// @brief Fills the matrix from a batched ciphertext without copying.
    /// @param src  The source ciphertext; must be of EncType::BatchRLWE. Its
    /// polynomial data is moved into the matrix, leaving it empty.
    /// @param rows The number of rows of the resulting matrix.
    /// @param cols The number of columns of the resulting matrix.
    /// @details Any previous content of the matrix is replaced.
    virtual void moveFrom(ICiphertext &src, u32 rows, u32 cols) = 0;

    /// @brief Gets the type of the ciphertext matrix.
    /// @return The type of the ciphertext matrix.
    virtual CtMatrixType type() const = 0;
    /// @brief Gets the PolyRing of the ciphertext matrix.
    /// @return PolyRing of the ciphertext matrix.
    /// @details PolyRing contains information about the modulus,
    /// polynomial degree, and NTT algorithm of the ciphertext matrix.
    virtual PolyRing ring() const = 0;
    /// @brief Gets the Encoding of the ciphertext matrix.
    /// @return Encoding of the ciphertext matrix.
    /// @details Encoding contains information about the number of slots in
    /// logarithm, scale of the ciphertext matrix, and whether the encoding is
    /// slot or coefficient.
    virtual Encoding encoding() const = 0;
    /// @brief Gets the number of logical rows.
    /// @return The row count.
    virtual u32 rows() const = 0;
    /// @brief Gets the number of logical columns.
    /// @return The column count.
    virtual u32 cols() const = 0;
    /// @brief Gets the number of secrets used in the encryption.
    /// @return The number of secrets.
    virtual u32 numSecrets() const = 0;

    /// @brief Checks whether the ciphertext matrix is in NTT (evaluation) form.
    /// @return true if the ciphertext matrix polynomials are in NTT form, false
    /// otherwise.
    virtual bool isNTT() const = 0;
};

} // namespace heaan

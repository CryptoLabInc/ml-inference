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

#include "HEaaN2/General.hpp"

#include <type_traits>

namespace heaan {

/// @brief A row-major matrix of data, packed into CKKS messages.
/// @tparam T Element type. Only Real is currently supported.
/// @details Logical rows / columns: the matrix as the caller sees it, i.e. what
/// is passed to the constructor and returned by rows() / cols().
/// @details Physical (stored) rows: each is one CKKS message holding degree =
/// 2^logDegree real coefficients. A logical row wider than a ring is split
/// along its columns into num_blocks = ceil(cols / degree) blocks, one message
/// per block, so the matrix stores rows * num_blocks physical rows in
/// block-major order: stored index = block * rows + row.
template <typename T> class HEAAN2_API Matrix : public DeviceSpecific {
    static_assert(std::is_same_v<T, Real>,
                  "Currently, Matrix supports only T = Real.");

public:
    /// @brief Creates an empty matrix for EnDecoder::decode.
    Matrix();
    /// @brief Creates a (rows x cols) matrix with the given packing width.
    /// @param log_degree log of the polynomial degree, i.e. the number of real
    /// coefficients packed per stored row.
    /// @param rows number of logical rows.
    /// @param cols number of logical columns.
    Matrix(u32 log_degree, u32 rows, u32 cols);

    /// @brief Checks if the matrix holds no rows.
    /// @return true if no rows are stored, false otherwise.
    bool isEmpty() const;

    /// @brief Creates a deep copy of the matrix.
    /// @return The copied matrix (same dims, packing width and device).
    Matrix<T> copy() const;

    /// @brief Gets the number of logical rows.
    /// @return the row count.
    u32 rows() const;
    /// @brief Gets the number of logical columns.
    /// @return the column count.
    u32 cols() const;
    /// @brief Gets log of the polynomial degree, i.e. the number of real
    /// coefficients packed per stored row.
    /// @return logDegree of the matrix.
    /// @details A stored row occupies a consecutive block of coefficients. The
    /// row is packed into a CKKS message, then encoded and encrypted into a
    /// single CKKS ciphertext, forming one part of the whole encrypted matrix.
    u32 logDegree() const;

    /// @brief Accesses the packed slot buffer of a stored row.
    /// @param row_idx Index of the stored row to access.
    /// @return a pointer to the first element of the row.
    /// @throws if row_idx is out of range.
    /// @details Cache the row pointer and index it for bulk read/write to avoid
    /// the per-call range/device check.
    T *operator[](size_t row_idx);
    /// @brief Accesses the packed slot buffer of a stored row.
    /// @param row_idx Index of the stored row to access.
    /// @return a constant pointer to the first element of the row.
    /// @throws if row_idx is out of range.
    /// @details Cache the row pointer and index it for bulk read/write to avoid
    /// the per-call range/device check.
    const T *operator[](size_t row_idx) const;

    /// @brief Accesses the element at logical (row, col).
    /// @param row Logical row index, in [0, rows()).
    /// @param col Element index within the logical row, spanning its blocks:
    /// 2^logDegree values per block.
    /// @return a reference to the element at (row, col).
    /// @throws if row or col is out of range, or the matrix is empty or off
    /// CPU.
    /// @details For bulk access index a cached operator[] row pointer instead;
    /// at() re-checks and re-resolves the stored index on every call.
    T &at(u32 row, u32 col);
    /// @brief Accesses the element at logical (row, col).
    /// @param row Logical row index, in [0, rows()).
    /// @param col Element index within the logical row, spanning its blocks:
    /// 2^logDegree values per block.
    /// @return a constant reference to the element at (row, col).
    /// @throws if row or col is out of range, or the matrix is empty or off
    /// CPU.
    /// @details For bulk access index a cached operator[] row pointer instead;
    /// at() re-checks and re-resolves the stored index on every call.
    const T &at(u32 row, u32 col) const;

    /// @brief Gets the device of the matrix.
    /// @return Device of the matrix.
    Device device() const override;
    /// @brief Copies the matrix to the specified device.
    /// @param device Device where the matrix is copied to.
    void to(Device device) override;

    Pimpl impl;
};

} // namespace heaan

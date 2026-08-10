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
#include "HEaaN2/HomEval.hpp"
#include "HEaaN2/ICtMatrix.hpp"
#include "HEaaN2/IPtMatrix.hpp"
#include "HEaaN2/PresetParams.hpp"

#include <optional>

namespace heaan {

/// @brief A class for performing homomorphic evaluations on plaintext and
/// ciphertext matrices.
class HEAAN2_API HomEvalMatrix {
public:
    /// @brief Constructs a HomEvalMatrix without a modulus chain.
    HomEvalMatrix();

    /// @brief Constructs a HomEvalMatrix with the specified preset parameters.
    /// @param id The preset parameters identifier.
    HomEvalMatrix(PresetParamsId id);

    /// @brief Constructs a HomEvalMatrix with the specified parameters.
    /// @param params The HomEval parameters.
    HomEvalMatrix(const HomEvalParams &params);

    /// @brief Rescales a plaintext matrix to reduce its scale.
    /// @param[in] op The plaintext matrix to be rescaled.
    /// @param[out] res The resulting rescaled plaintext matrix.
    /// @param[in] r_ntt The NTT domain of the result. If not given, the domain
    /// of op is kept.
    /// @details Rescaling consumes one level.
    /// @throws if op is empty or op and res do not share the same layout.
    /// @throws if the evaluator was constructed without levels.
    /// @throws if the modulus of op is not in the levels, or its level is
    /// zero.
    /// @throws if the scale of op does not match the modulus chain.
    void rescale(const IPtMatrix &op, IPtMatrix &res,
                 const std::optional<bool> r_ntt = std::nullopt) const;

    /// @brief Rescales a ciphertext matrix to reduce its scale.
    /// @param[in] op The ciphertext matrix to be rescaled.
    /// @param[out] res The resulting rescaled ciphertext matrix.
    /// @param[in] r_ntt The NTT domain of the result. If not given, the domain
    /// of op is kept.
    /// @details Rescaling consumes one level.
    /// @throws if op is empty or op and res do not share the same layout.
    /// @throws if the evaluator was constructed without levels.
    /// @throws if the modulus of op is not in the levels, or its level is
    /// zero.
    /// @throws if the scale of op does not match the modulus chain.
    void rescale(const ICtMatrix &op, ICtMatrix &res,
                 const std::optional<bool> r_ntt = std::nullopt) const;

    /// @brief Multiplies a plaintext matrix by a ciphertext matrix, producing a
    /// ciphertext matrix (W = U * V).
    /// @param[in] u The left, plaintext operand of shape m x k.
    /// @param[in] v The right, ciphertext operand of shape k x n.
    /// @param[out] w The resulting ciphertext matrix of shape m x n.
    /// @throws if u or v is empty.
    /// @throws if v and w do not share the same layout.
    /// @throws if either operand is slot-encoded (dft).
    /// @throws if u and v have incompatible rings or devices.
    /// @throws if u.cols() does not equal v.rows(), or if v.rows() exceeds the
    /// ring degree.
    /// @details Both operands must be coefficient-encoded (non-dft) and
    /// contract over the inner dimension k = v.rows(). The GEMM backend is
    /// selected by device: native on CPU, int8-quantized on GPU. The scale of w
    /// will be the product of the scales of u and v.
    void pcmm(const IPtMatrix &u, const ICtMatrix &v, ICtMatrix &w) const;

    /// @brief Gets the homomorphic evaluation parameters.
    /// @return The homomorphic evaluation parameters.
    const HomEvalParams &getParams() const { return params_; }

private:
    Pimpl impl;
    HomEvalParams params_;
};

} // namespace heaan

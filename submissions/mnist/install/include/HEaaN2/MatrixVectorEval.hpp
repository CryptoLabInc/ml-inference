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
#include "HEaaN2/ICiphertext.hpp"
#include "HEaaN2/KeyUtils.hpp"
#include "HEaaN2/Levels.hpp"
#include "HEaaN2/Message.hpp"
#include "HEaaN2/PolyRing.hpp"

#include <map>
#include <set>
#include <vector>

namespace heaan {

/// @brief Parameters for the MatrixVectorEval operation.
/// @param bs_indices, gs_indices The baby-step and giant-step splitting of the
///        diagonal offsets. The evaluated diagonals are the sums gs + bs, which
///        must be pairwise distinct.
/// @param ring, encoding: the ring and the encoding of the input ciphertext,
/// @param levels: the modulus chain the input ciphertext lives in.
/// @param poly_type: the type of the polynomial (SIMPLE/GRAFTED) of the input
///        ciphertext
struct HEAAN2_API MatrixVectorEvalParams {
    std::vector<i32> bs_indices;
    std::vector<i32> gs_indices;
    PolyRing ring;
    Encoding encoding;
    Levels levels;
    PolyType poly_type = PolyType::SIMPLE;

    /// @brief Sets the baby-step giant-step splitting of the diagonal offsets.
    /// @param bs_indices The baby steps.
    /// @param gs_indices The giant steps.
    /// @details The evaluated diagonals are the sums gs + bs, which must be
    /// pairwise distinct.
    MatrixVectorEvalParams &setSteps(const std::vector<i32> &bs_indices,
                                     const std::vector<i32> &gs_indices);
    /// @brief Sets the ring and the encoding of the input ciphertext.
    /// @param ring The ring of the input ciphertext.
    /// @param encoding The encoding of the input ciphertext.
    /// @details Pass op.ring() and op.encoding() of the ciphertext eval() will
    /// be called on: the diagonals are encoded to match them.
    MatrixVectorEvalParams &setInput(const PolyRing &ring,
                                     const Encoding &encoding);
    /// @brief Sets the modulus chain the input ciphertext lives in.
    /// @details The evaluation consumes one level, so the output lands on the
    /// modulus and the scale of the level right below the one of the input.
    MatrixVectorEvalParams &setLevels(const Levels &levels);
    /// @brief Sets the type of the polynomial (SIMPLE/GRAFTED).
    /// @details Must match the one of the input ciphertext.
    MatrixVectorEvalParams &setPolyType(PolyType poly_type);

    /// @brief Checks the validity of the MatrixVectorEvalParams.
    /// @throws if either index vector is empty or holds duplicates, or if two
    /// (giant step, baby step) pairs share a diagonal offset.
    /// @throws if encoding is not slot-encoding, or if encoding holds more
    /// slots than ring does.
    /// @throws if the modulus of ring is not one of levels, or if it is the
    /// bottom one, leaving no level for the evaluation to consume.
    /// @details The scale of encoding is not required to be the one levels
    /// holds for that modulus: the diagonals are encoded to the scale of the
    /// input, so the output simply keeps the square of it, mod-downed.
    void checkValidity() const;
};

/// @brief A class evaluating a matrix-vector product on ciphertexts
/// @details With n slots and the diagonal of offset d given as diags[d], the
/// evaluated map is res[i] = sum_d diags[d][i] * op[(i + d) % n]. The
/// evaluation uses the baby-step giant-step algorithm with double-hoisted
/// rotations, and consumes one level.
class HEAAN2_API MatrixVectorEval {
public:
    /// @brief Gets the rotation steps whose keys the evaluation requires.
    /// @param params The MatrixVectorEval parameters.
    /// @return The rotation steps
    static std::set<i32> rotKeyIndices(const MatrixVectorEvalParams &params);

    /// @brief Constructs a MatrixVectorEval for the matrix given by @p diags.
    /// @param params The MatrixVectorEval parameters.
    /// @param rot_keys The rotation keys for the evaluation, which must
    /// contain a key for every step of rotKeyIndices(params) and must outlive
    /// this object.
    /// @param diags The diagonals of the matrix, keyed by their offset.
    /// @details The diagonals are encoded on the initialization, on the device
    /// of rot_keys. The diagonals must be on the CPU.
    /// @throws if diags is empty, holds a diagonal that no (giant step, baby
    /// step) pair reaches, or leaves a giant step without any diagonal.
    /// @throws if a rotation key is missing.
    MatrixVectorEval(const MatrixVectorEvalParams &params,
                     const RotKeyPtrs &rot_keys,
                     const std::map<i32, Message> &diags);

    /// @brief Evaluates the matrix-vector product.
    /// @param[in] op The ciphertext holding the vector in its slots.
    /// @param[out] res The resulting ciphertext, one level below op.
    /// @details res may be the same object as op, in which case the result
    /// replaces the input.
    /// @throws if op is empty, is not in NTT form, or does not have the ring,
    /// the encoding or the PolyType the evaluator was constructed for.
    /// @throws if op and res have different encryption types, or if the device
    /// of op does not match the device of the rotation keys.
    void eval(const ICiphertext &op, ICiphertext &res) const;

    /// @brief Gets the device of the rotation keys.
    /// @return The device of the rotation keys.
    Device device() const;

private:
    Pimpl impl;
    Device device_;
};

} // namespace heaan

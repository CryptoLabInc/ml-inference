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
#include "HEaaN2/KeyUtils.hpp"
#include "HEaaN2/Levels.hpp"

namespace heaan {

/// @brief Computes the required depth (i.e. the level consumption) of the
/// Chebyshev polynomial evaluation for the given coefficients.
/// @param coeffs The coefficients of the Chebyshev polynomial.
/// @return The required depth of the Chebyshev polynomial evaluation.
/// @details The required depth is equal to ceil(log2(coeffs.size()))
u32 computeChebyDepth(const std::vector<Real> &coeffs);

/// @brief Parameters for Chebyshev polynomial evaluation
/// @details
/// - levels: The levels for the Chebyshev polynomial evaluation. The levels
/// should be sufficient for the evaluation, which can be checked by
/// computeChebyDepth.
/// - coeffs: The coefficients of the Chebyshev polynomial.
/// - log_degree_bs: log of the degree of baby step used in the evaluation.
struct HEAAN2_API ChebyshevParams {
    ChebyshevParams() = default;
    ChebyshevParams(const Levels &levels_, const std::vector<Real> &coeffs_,
                    u32 log_degree_bs_)
        : levels(levels_), coeffs(coeffs_), log_degree_bs(log_degree_bs_) {
        checkValidity();
    }

    Levels levels;
    std::vector<Real> coeffs;
    u32 log_degree_bs;

    /// @brief Checks the validity of the ChebyshevParams.
    /// @throws if log_degree_bs is not 2 or 3.
    ///          - coeffs.size() must be at least 8.
    ///          - coeffs.size() must be at least 2^log_degree_bs.
    ///          - levels.size() should be at least depth + 1.
    void checkValidity() const;
};

/// @brief Evaluates a Chebyshev polynomial on a ciphertext.
class HEAAN2_API ChebyshevPolynomialEval {
public:
    /// @brief Constructs a ChebyshevPolynomialEval with the specified
    /// parameters.
    /// @param params_ The parameters for the Chebyshev polynomial evaluation.
    ChebyshevPolynomialEval(const ChebyshevParams &params_);
    /// @brief Evaluates the Chebyshev polynomial on the input ciphertext and
    /// stores the result in the output ciphertext.
    /// @param op The input ciphertext.
    /// @param res The output ciphertext where the result is stored.
    /// @param keys The keys required for the evaluation.
    /// @throws if op is not forwardNTTed.
    /// @details Input 1-size key to use one key for all levels, and
    /// mult-depth-many keys to use corresponding keys for each of
    /// multiplication modulus.
    void eval(const ICiphertext &op, ICiphertext &res,
              const KeyPtrBundle &keys) const;

private:
    ChebyshevParams params;
};

} // namespace heaan

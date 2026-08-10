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
#include "HEaaN2/ISwKey.hpp"

namespace heaan {

/// @brief A class evaluating the ring-dimension switching operations,
/// packing low degree ciphertexts into high degree ciphertexts (compose) and
/// splitting high degree ciphertexts into low degree ciphertexts (decompose).
class HEAAN2_API RingPacker {
public:
    /// @brief Compose low degree ciphertexts into one high degree ciphertext.
    /// @param op The coefficient-encoded low degree ciphertexts of a common
    /// ring. The number of ciphertexts must not exceed 2^(log_degree - input
    /// log degree).
    /// @param res The output ciphertext of high degree, whose
    /// coefficients interleave the coefficients of the inputs with a stride
    /// of 2^(log_degree - input log degree).
    /// @param key The compose key (see SwKeyGenerator::genComposeKey).
    /// @param log_degree log degree of the output ring.
    /// @details The devices of the input ciphertexts must match the device
    /// of the key.
    /// @details The output is always in iNTT form.
    void compose(const std::vector<const ICiphertext *> &op, ICiphertext &res,
                 const ISwKey &key, u32 log_degree) const;

    /// @brief Packs many groups of small-ring ciphertexts into large-ring
    /// ciphertexts in a single batched call.
    /// @param op A coefficient-encoded BatchRLWE ciphertext whose batch holds
    /// the small-ring inputs of a common ring, grouped by output: with
    /// log_gap = log_degree - input log degree, batch elements in
    /// [g * 2^log_gap, (g + 1) * 2^log_gap) are composed into res[g]. The batch
    /// size must be a multiple of 2^log_gap.
    /// @param res The output ciphertexts of degree 2^log_degree. res[g]
    /// interleaves its group's coefficients with a stride of 2^log_gap, exactly
    /// as the non-batched compose. The number of ciphertexts must equal the
    /// batch size divided by 2^log_gap.
    /// @param key The switching key from the high-degree embedding of the
    /// small-ring secret key to the large-ring secret key (see
    /// SwKeyGenerator::genComposeKey).
    /// @param log_degree log degree of the output ring.
    /// @details The device of the input ciphertext must match the device of
    /// the key.
    /// @details The output is always in iNTT form.
    void compose(const ICiphertext &op, const std::vector<ICiphertext *> &res,
                 const ISwKey &key, u32 log_degree) const;

    /// @brief Decompose a high degree ciphertext into low degree ciphertexts.
    /// @param op The coefficient-encoded high degree ciphertext.
    /// @param res The output ciphertexts of low degree, receiving
    /// the coefficients of the input with a stride of 2^(input log degree -
    /// log_degree). The number of ciphertexts must be exactly 2^(input log
    /// degree - log_degree).
    /// @param key The decompose key (see
    /// SwKeyGenerator::genDecomposeKey).
    /// @param log_degree log degree of the output ring.
    /// @details The device of the input ciphertext must match the device of
    /// the key.
    /// @details The outputs are always in iNTT form.
    void decompose(const ICiphertext &op, const std::vector<ICiphertext *> &res,
                   const ISwKey &key, u32 log_degree) const;

    /// @brief Splits many large-ring ciphertexts into small-ring ciphertexts
    /// packed into one BatchRLWE ciphertext, in a single batched call.
    /// @param op The coefficient-encoded large-ring ciphertexts of a common
    /// ring.
    /// @param res A BatchRLWE ciphertext receiving 2^log_gap * op.size()
    /// small-ring ciphertexts of degree 2^log_degree, where
    /// log_gap = input log degree - log_degree. op[g] is split with a stride of
    /// 2^log_gap, exactly as the non-batched decompose, into batch elements
    /// [g * 2^log_gap, (g + 1) * 2^log_gap).
    /// @param key The switching key from the large-ring secret key to the
    /// high-degree embedding of the small-ring secret key (see
    /// SwKeyGenerator::genDecomposeKey).
    /// @param log_degree log degree of the output ring.
    /// @details The devices of the input ciphertexts must match the device of
    /// the key.
    /// @details The outputs are always in iNTT form.
    void decompose(const std::vector<const ICiphertext *> &op, ICiphertext &res,
                   const ISwKey &key, u32 log_degree) const;
};

} // namespace heaan

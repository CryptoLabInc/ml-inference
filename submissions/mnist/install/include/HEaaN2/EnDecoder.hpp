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
#include "HEaaN2/IPlaintext.hpp"
#include "HEaaN2/IPtMatrix.hpp"
#include "HEaaN2/Levels.hpp"
#include "HEaaN2/Matrix.hpp"
#include "HEaaN2/Message.hpp"
#include "HEaaN2/PresetParams.hpp"

#include <optional>

namespace heaan {

/// @brief A struct for encoding parameters
/// @details
/// - poly_type: The type of the polynomial (SIMPLE/GRAFTED).
/// - log_degree: log degree of the ring.
/// - levels: The levels for encoding.
/// - ntt_alg: The NTT algorithm (NORMAL/CYC_FOR_CI).
/// - coeff_encoding: Whether to use coefficient encoding.
struct HEAAN2_API EncodeParams {
public:
    /// @brief Constructs EncodeParams with the specified parameters.
    /// @param poly_type The type of the polynomial (SIMPLE/GRAFTED).
    /// @param log_degree log degree of the ring.
    /// @param levels The levels.
    /// @param ntt_alg The NTT algorithm (NORMAL/CYC_FOR_CI).
    /// @param coeff_encoding Whether to use coefficient encoding.
    EncodeParams(PolyType poly_type, u32 log_degree, const Levels &levels = {},
                 NTTAlgorithm ntt_alg = NTTAlgorithm::NORMAL,
                 bool coeff_encoding = false)
        : poly_type_(poly_type), log_degree_(log_degree), levels_(levels),
          ntt_alg_(ntt_alg), coeff_encoding_(coeff_encoding){};

    /// @brief Gets the polynomial type.
    /// @return The polynomial type.
    PolyType getPolyType() const { return poly_type_; };
    /// @brief Gets the log degree.
    /// @return The log degree of the ring.
    u32 getLogDegree() const { return log_degree_; };
    /// @brief Gets the view of levels.
    /// @return The levels.
    const Levels &getLevels() const { return levels_; };
    /// @brief Gets the NTT algorithm.
    /// @return The NTT algorithm.
    NTTAlgorithm getNTTAlgorithm() const { return ntt_alg_; };
    /// @brief Checks whether coefficient encoding is used.
    /// @return True if coefficient encoding is used, false otherwise.
    bool isCoeffEncoding() const { return coeff_encoding_; };

private:
    PolyType poly_type_;
    u32 log_degree_;
    Levels levels_;
    NTTAlgorithm ntt_alg_;
    bool coeff_encoding_;
};

/// @brief A class for encoding messages into plaintexts and decoding
/// plaintexts to messages
class HEAAN2_API EnDecoder {
public:
    /// @brief Constructs an EnDecoder without specifying parameters.
    /// @details The EnDecoder is capable of decoding any plaintexts.
    EnDecoder();
    /// @brief Constructs an EnDecoder with the specified preset
    /// parameters.
    /// @param id The preset parameters identifier.
    EnDecoder(PresetParamsId id);

    /// @brief Constructs an EnDecoder with the specified encoding
    /// parameters.
    /// @param params The encoding parameters.
    EnDecoder(const EncodeParams &params);

    /// @brief Encodes a message into a plaintext with the specified modulus and
    /// scale.
    /// @tparam TComplex Complex number type (Complex or Complex128).
    /// @param[in] msg The input message to encode.
    /// @param[out] ptxt The output plaintext.
    /// @param[in] mod The modulus of the plaintext.
    /// @param[in] scale The scale of the plaintext.
    /// @details The device of the output follows from the input.
    /// @throws if the number of slots in the message exceeds half of the
    /// polynomial degree determined by the preset parameters.
    template <typename TComplex>
    void encode(const MessageT<TComplex> &msg, IPlaintext &ptxt,
                const PolyMod &mod, Real128 scale) const;

    /// @brief Encodes a message into a plaintext with the specified level.
    /// @tparam TComplex Complex number type (Complex or Complex128).
    /// @param[in] msg The input message to encode.
    /// @param[out] ptxt The output plaintext.
    /// @param[in] level The specified level.
    /// @details The device of the output follows from the input.
    /// @throws if the number of slots in the message exceeds half of the
    /// polynomial degree determined by the preset parameters.
    /// @throws if the level exceeds the maximum level determined by the
    /// preset parameters.
    template <typename TComplex>
    void encode(const MessageT<TComplex> &msg, IPlaintext &ptxt,
                u32 level) const;

    /// @brief Encodes a message into a plaintext at the maximum level.
    /// @tparam TComplex Complex number type (Complex or Complex128).
    /// @param[in] msg The input message to encode.
    /// @param[out] ptxt The output plaintext.
    /// @details The device of the output follows from the input.
    /// @throws if the number of slots in the message exceeds half of the
    /// polynomial degree determined by the preset parameters.
    template <typename TComplex>
    void encode(const MessageT<TComplex> &msg, IPlaintext &ptxt) const;

    /// @brief Decodes a plaintext into a message.
    /// @tparam TComplex Complex number type (Complex or Complex128)
    /// @param[in] ptxt The input plaintext to decode.
    /// @param[out] msg The output message.
    /// @details The device of the output follows from the input.
    template <typename TComplex>
    void decode(const IPlaintext &ptxt, MessageT<TComplex> &msg) const;

    /// @brief Encodes multiple messages into a batched plaintext with the
    /// specified modulus and scale.
    /// @tparam TComplex Complex number type (Complex or Complex128).
    /// @param[in] msgs The input messages to encode.
    /// @param[out] ptxt The output plaintext.
    /// @param[in] mod The modulus of the plaintext.
    /// @param[in] scale The scale of the plaintext.
    /// @details The device of the output follows from the input.
    template <typename TComplex>
    void encode(const std::vector<MessageT<TComplex>> &msgs, IPlaintext &ptxt,
                const PolyMod &mod, Real128 scale) const;

    /// @brief Encodes multiple messages into a batched plaintext with the
    /// specified level.
    /// @tparam TComplex Complex number type (Complex or Complex128).
    /// @param[in] msgs The input messages to encode.
    /// @param[out] ptxt The output plaintext.
    /// @param[in] level The specified level.
    /// @details The device of the output follows from the input.
    template <typename TComplex>
    void encode(const std::vector<MessageT<TComplex>> &msgs, IPlaintext &ptxt,
                u32 level) const;

    /// @brief Encodes multiple messages into a batched plaintext at the
    /// maximum level.
    /// @tparam TComplex Complex number type (Complex or Complex128).
    /// @param[in] msgs The input messages to encode.
    /// @param[out] ptxt The output plaintext.
    /// @details The device of the output follows from the input.
    template <typename TComplex>
    void encode(const std::vector<MessageT<TComplex>> &msgs,
                IPlaintext &ptxt) const;

    /// @brief Decodes a batched plaintext into multiple messages.
    /// @tparam TComplex Complex number type (Complex or Complex128)
    /// @param[in] ptxt The input plaintext to decode.
    /// @param[out] msgs The output messages.
    /// @details The device of the output follows from the input.
    template <typename TComplex>
    void decode(const IPlaintext &ptxt,
                std::vector<MessageT<TComplex>> &msgs) const;

    /// @brief Encodes a matrix into a plaintext matrix with the specified
    /// modulus and scale.
    /// @tparam T Element type of the matrix. Only Real is supported.
    /// @param[in] matrix The input matrix to encode. Each logical row is
    /// encoded into one CKKS message per block: a row wider than the ring is
    /// split into ceil(cols / degree) messages.
    /// @param[out] ptxt The output plaintext matrix.
    /// @param[in] mod The modulus of the plaintext matrix.
    /// @param[in] scale The scale of the plaintext matrix.
    /// @details The device of the output follows from the input.
    /// @throws if the encoding parameters use NTTAlgorithm::CYC_FOR_CI.
    /// @throws if matrix.logDegree() does not match the encoding parameters'
    /// log degree.
    template <typename T>
    void encode(const Matrix<T> &matrix, IPtMatrix &ptxt, const PolyMod &mod,
                Real128 scale) const;

    /// @brief Encodes a matrix into a plaintext matrix with the specified
    /// level.
    /// @tparam T Element type of the matrix. Only Real is supported.
    /// @param[in] matrix The input matrix to encode.
    /// @param[out] ptxt The output plaintext matrix.
    /// @param[in] level The specified level.
    /// @details The device of the output follows from the input.
    /// @throws if the level exceeds the maximum level determined by the
    /// encoding parameters.
    template <typename T>
    void encode(const Matrix<T> &matrix, IPtMatrix &ptxt, u32 level) const;

    /// @brief Encodes a matrix into a plaintext matrix at the maximum level.
    /// @tparam T Element type of the matrix. Only Real is supported.
    /// @param[in] matrix The input matrix to encode.
    /// @param[out] ptxt The output plaintext matrix.
    /// @details The device of the output follows from the input.
    template <typename T>
    void encode(const Matrix<T> &matrix, IPtMatrix &ptxt) const;

    /// @brief Decodes a plaintext matrix into a matrix.
    /// @tparam T Element type of the matrix. Only Real is supported.
    /// @param[in] ptxt The input plaintext matrix to decode.
    /// @param[out] matrix The output matrix.
    /// @details The device of the output follows from the input.
    template <typename T>
    void decode(const IPtMatrix &ptxt, Matrix<T> &matrix) const;

    /// @brief Gets the encoding parameters.
    /// @return The encoding parameters.
    const std::optional<EncodeParams> &getParams() const { return params_; }

private:
    Pimpl impl;
    std::optional<EncodeParams> params_;
};

} // namespace heaan

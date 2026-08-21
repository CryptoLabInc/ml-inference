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

#include "HEaaN2/Export.hpp"
#include "HEaaN2/ICiphertext.hpp"
#include "HEaaN2/IPlaintext.hpp"
#include "HEaaN2/ISwKey.hpp"
#include "HEaaN2/KeyUtils.hpp"
#include "HEaaN2/Levels.hpp"
#include "HEaaN2/PresetParams.hpp"

namespace heaan {

/// @brief A struct for HomEval parameters
/// @details
/// - levels: The levels that are used for scalar multiplication, rescaling and
/// levelDown.
struct HEAAN2_API HomEvalParams {
public:
    /// @brief Constructs HomEvalParams with the specified levels.
    /// @param levels The levels for HomEval.
    HomEvalParams(const Levels &levels) : levels_(levels) {}

    /// @brief Gets the levels.
    /// @return The levels.
    const Levels &getLevels() const;

private:
    Levels levels_;
};

/// @brief A class for performing homomorphic evaluations on ciphertexts and
/// plaintexts.
class HEAAN2_API HomEval {
public:
    /// @brief Constructs a HomEval with the specified preset parameters.
    /// @param id The preset parameters identifier.
    HomEval(PresetParamsId id);

    /// @brief Constructs a HomEval with the specified parameters.
    /// @param params The HomEval parameters.
    HomEval(const HomEvalParams &params);

    // Utilities
    /// @brief Gets the number of available multiplications on a plaintext.
    /// @param[in] op The plaintext.
    /// @return The number of available multiplications on the plaintext.
    /// @throws on incompatible modulus chain of the plaintext.
    u32 getLevel(const IPlaintext &op) const;
    /// @brief Gets the number of available multiplications of a ciphertext.
    /// @param[in] op The ciphertext.
    /// @return The number of available multiplications of the ciphertext.
    /// @throws on incompatible modulus chain of the ciphertext.
    u32 getLevel(const ICiphertext &op) const;

    // Basic Homomorphic Operations
    /// @brief Negates a plaintext.
    /// @param[in] op The plaintext to be negated.
    /// @param[out] res The resulting negated plaintext.
    void neg(const IPlaintext &op, IPlaintext &res) const;
    /// @brief Negates a ciphertext.
    /// @param[in] op The ciphertext to be negated.
    /// @param[out] res The resulting negated ciphertext.
    void neg(const ICiphertext &op, ICiphertext &res) const;
    /// @brief Adds two plaintexts.
    /// @param[in] op1 The first plaintext.
    /// @param[in] op2 The second plaintext.
    /// @param[out] res The resulting plaintext after addition.
    /// @throws if op1 and op2 have different levels, scales, or devices.
    /// @throws if op1 and op2 are not both forwardNTTed or both backwardNTTed.
    /// @details if the number of slots of op1 and op2 differ, the smaller
    /// one will be broadcasted to match the larger one.
    void add(const IPlaintext &op1, const IPlaintext &op2,
             IPlaintext &res) const;
    /// @brief Adds a ciphertext and a plaintext.
    /// @param[in] op1 The ciphertext.
    /// @param[in] op2 The plaintext.
    /// @param[out] res The resulting ciphertext after addition.
    /// @throws if op1 and op2 have different levels, scales, or devices.
    /// @throws if op1 and op2 are not both forwardNTTed or both backwardNTTed.
    /// @details if the number of slots of op1 and op2 differ, the smaller
    /// one will be broadcasted to match the larger one.
    void add(const ICiphertext &op1, const IPlaintext &op2,
             ICiphertext &res) const;
    /// @brief Adds two ciphertexts.
    /// @param[in] op1 The first ciphertext.
    /// @param[in] op2 The second ciphertext.
    /// @param[out] res The resulting ciphertext after addition.
    /// @throws if op1 and op2 have different levels, scales, or devices.
    /// @throws if op1 and op2 are not both forwardNTTed or both backwardNTTed.
    /// @details if the number of slots of op1 and op2 differ, the smaller
    /// one will be broadcasted to match the larger one.
    void add(const ICiphertext &op1, const ICiphertext &op2,
             ICiphertext &res) const;
    /// @brief Subtracts two plaintexts.
    /// @param[in] op1 The first plaintext.
    /// @param[in] op2 The second plaintext.
    /// @param[out] res The resulting plaintext (op1 - op2) after
    /// subtraction.
    /// @throws if op1 and op2 have different levels, scales, or devices.
    /// @throws if op1 and op2 are not both forwardNTTed or both backwardNTTed.
    /// @details if the number of slots of op1 and op2 differ, the smaller
    /// one will be broadcasted to match the larger one.
    void sub(const IPlaintext &op1, const IPlaintext &op2,
             IPlaintext &res) const;
    /// @brief Subtracts a plaintext from a ciphertext.
    /// @param[in] op1 The ciphertext.
    /// @param[in] op2 The plaintext.
    /// @param[out] res The resulting ciphertext (op1 - op2) after
    /// subtraction.
    /// @throws if op1 and op2 have different levels, scales, or devices.
    /// @throws if op1 and op2 are not both forwardNTTed or both backwardNTTed.
    /// @details if the number of slots of op1 and op2 differ, the smaller
    /// one will be broadcasted to match the larger one.
    void sub(const ICiphertext &op1, const IPlaintext &op2,
             ICiphertext &res) const;
    /// @brief Subtracts two ciphertexts.
    /// @param[in] op1 The first ciphertext.
    /// @param[in] op2 The second ciphertext.
    /// @param[out] res The resulting ciphertext (op1 - op2) after
    /// subtraction.
    /// @throws if op1 and op2 have different levels, scales, or devices.
    /// @throws if op1 and op2 are not both forwardNTTed or both backwardNTTed.
    /// @details if the number of slots of op1 and op2 differ, the smaller
    /// one will be broadcasted to match the larger one.
    void sub(const ICiphertext &op1, const ICiphertext &op2,
             ICiphertext &res) const;
    /// @brief Multiplies two plaintexts.
    /// @param[in] op1 The first plaintext.
    /// @param[in] op2 The second plaintext.
    /// @param[out] res The resulting plaintext after multiplication.
    /// @throws if op1 and op2 have different levels or devices.
    /// @throws if op1 or op2 is not forwardNTTed.
    /// @details The scale of res will be the product of the scales of op1
    /// and op2.
    /// @details if the number of slots of op1 and op2 differ, the smaller
    /// one will be broadcasted to match the larger one.
    void mul(const IPlaintext &op1, const IPlaintext &op2,
             IPlaintext &res) const;
    /// @brief Multiplies a ciphertext and a plaintext.
    /// @param[in] op1 The ciphertext.
    /// @param[in] op2 The plaintext.
    /// @param[out] res The resulting ciphertext after multiplication.
    /// @throws if op1 and op2 have different levels or devices.
    /// @throws if op1 or op2 is not forwardNTTed.
    /// @details The scale of res will be the product of the scales of op1
    /// and op2.
    /// @details if the number of slots of op1 and op2 differ, the smaller
    /// one will be broadcasted to match the larger one.
    void mul(const ICiphertext &op1, const IPlaintext &op2,
             ICiphertext &res) const;
    /// @brief Computes the tensor product of two ciphertexts.
    /// @param[in] op1 The first ciphertext.
    /// @param[in] op2 The second ciphertext.
    /// @param[out] res The resulting ciphertext after tensor product, which
    /// encrypts the multiplication of the messages of the input
    /// ciphertexts.
    /// @throws if op1 and op2 have different levels or devices.
    /// @throws if powers of op1 and op2 are not both 1.
    /// @throws if op1 or op2 is not forwardNTTed.
    /// @details The scale of res will be the product of the scales of op1
    /// and op2. The resulting ciphertext has power two. To reduce it back
    /// to a power-one ciphertext, use the relin function with an
    /// appropriate relinearization key.
    /// @details if the number of slots of op1 and op2 differ, the smaller
    /// one will be broadcasted to match the larger one.
    void tensor(const ICiphertext &op1, const ICiphertext &op2,
                ICiphertext &res) const;

    // Operations with Complex Numbers
    /// @brief Adds a complex number to a plaintext.
    /// @param[in] op1 The plaintext.
    /// @param[in] op2 The complex number.
    /// @param[out] res The resulting plaintext after addition.
    /// @throws if op1 is not forwardNTTed.
    /// @details op2 will be encoded on the fly so that it has the same
    /// encoding as op1.
    void add(const IPlaintext &op1, Complex128 op2, IPlaintext &res) const;
    /// @brief Adds a complex number to a ciphertext.
    /// @param[in] op1 The ciphertext.
    /// @param[in] op2 The complex number.
    /// @param[out] res The resulting ciphertext after addition.
    /// @throws if op1 is not forwardNTTed.
    /// @details op2 will be encoded on the fly so that it has the same
    /// encoding as op1.
    void add(const ICiphertext &op1, Complex128 op2, ICiphertext &res) const;
    /// @brief Subtracts a complex number from a plaintext.
    /// @param[in] op1 The plaintext.
    /// @param[in] op2 The complex number.
    /// @param[out] res The resulting plaintext (op1 - op2) after
    /// subtraction.
    /// @throws if op1 is not forwardNTTed.
    /// @details op2 will be encoded on the fly so that it has the same
    /// encoding as op1.
    void sub(const IPlaintext &op1, Complex128 op2, IPlaintext &res) const;
    /// @brief Subtracts a complex number from a ciphertext.
    /// @param[in] op1 The ciphertext.
    /// @param[in] op2 The complex number.
    /// @param[out] res The resulting ciphertext (op1 - op2) after
    /// subtraction.
    /// @throws if op1 is not forwardNTTed.
    /// @details op2 will be encoded on the fly so that it has the same
    /// encoding as op1.
    void sub(const ICiphertext &op1, Complex128 op2, ICiphertext &res) const;
    /// @brief Multiplies a plaintext by a complex number.
    /// @param[in] op1 The plaintext.
    /// @param[in] op2 The complex number.
    /// @param[out] res The resulting plaintext after multiplication.
    /// @throws if op2 has a nonzero imaginary part and op1 is not
    /// forwardNTTed.
    /// @details op2 will be encoded on the fly so that it has the same
    /// encoding as op1. The resulting plaintext will have the scale =
    /// (scale of op1) ** 2.
    void mul(const IPlaintext &op1, Complex128 op2, IPlaintext &res) const;
    /// @brief Multiplies a ciphertext by a complex number.
    /// @param[in] op1 The ciphertext.
    /// @param[in] op2 The complex number.
    /// @param[out] res The resulting ciphertext after multiplication.
    /// @throws if op2 has a nonzero imaginary part and op1 is not
    /// forwardNTTed.
    /// @details op2 will be encoded on the fly so that it has the same
    /// encoding as op1. The resulting ciphertext will have the scale =
    /// (scale of op1)
    /// ** 2.
    void mul(const ICiphertext &op1, Complex128 op2, ICiphertext &res) const;

    // Automorphisms
    /// @brief Rotates a plaintext by a specified step.
    /// @param[in] op The plaintext to be rotated.
    /// @param[in] step The rotation step.
    /// @param[out] res The resulting rotated plaintext.
    /// @details Positive step indicates left rotation, while negative step
    /// indicates right rotation.
    void rot(const IPlaintext &op, i32 step, IPlaintext &res) const;
    /// @brief Conjugates a plaintext.
    /// @param[in] op The plaintext to be conjugated.
    /// @param[out] res The resulting conjugated plaintext.
    void conj(const IPlaintext &op, IPlaintext &res) const;
    /// @brief Applies the Galois automorphism X -> X^pow to a ciphertext,
    /// without a key switch.
    /// @param[in] op The ciphertext to be mapped.
    /// @param[in] pow The Galois exponent, taken directly rather than as a slot
    /// step: the exponent rotating left by `step` is
    /// 5^(step mod 2^(log_degree - 1)) mod 2^(log_degree + 1), the conversion
    /// the keyed rot() applies internally.
    /// @param[out] res The resulting ciphertext. May alias `op`.
    /// @details This is the automorphism alone: it consumes no level, changes
    /// neither the scale nor the NTT state, adds no noise beyond a permutation
    /// of the existing noise, and needs no key material. RLWE and BatchRLWE
    /// ciphertexts of any power are supported; for BatchRLWE the same
    /// automorphism applies element-wise across every lane, which is what makes
    /// a lane-shared key fold in one call.
    ///
    /// PRECONDITION, WHICH THIS FUNCTION CANNOT CHECK: the secret key must be
    /// invariant under the automorphism. Applying X -> X^pow to a ciphertext
    /// encrypted under s yields a valid ciphertext under the mapped key
    /// s(X^pow), so the result is only meaningful when s(X^pow) = s. That holds
    /// for a key built by SKGenerator::genHighDegreeKey, whose invariance
    /// period is documented there, when `pow` is the exponent of a rotation by
    /// a multiple of that period. Under any other key this call still succeeds
    /// and the result silently decrypts to noise.
    /// @throws if op is empty, if the encryption types of op and res do not
    /// match, if op is neither RLWE nor BatchRLWE, if pow is even, or if pow is
    /// not 1 mod 4 for a conjugate-invariant ciphertext.
    void frobMap(const ICiphertext &op, i32 pow, ICiphertext &res) const;

    // Key-switching Operations
    /// @brief Relinearizes a ciphertext using a relinearization key.
    /// @param[inout] op The ciphertext to be relinearized.
    /// @param[in] key The relinearization key.
    /// @details relin is an in-place operation that modifies the input
    /// ciphertext to reduce its power back to one. If the rescaling is
    /// desired after relinearization, the rescale function should be called
    /// separately.
    /// @throws if op and key are on different devices.
    /// @throws if the power of the input ciphertext is not 2.
    void relin(ICiphertext &op, const ISwKey &key) const;

    /// @brief Relinearizes a ciphertext using a bundle of leveled
    /// relinearization keys.
    /// @param[inout] op The ciphertext to be relinearized.
    /// @param[in] leveled_relin_keys The bundle of leveled relinearization
    /// keys.
    /// @details relin is an in-place operation that modifies the input
    /// ciphertext to reduce its power back to one. If the rescaling is
    /// desired after relinearization, the rescale function should be called
    /// separately.
    /// @throws if op and leveled_relin_keys are on different devices.
    /// @throws if the power of the input ciphertext is not 2.
    /// @throws if the modulus of the input ciphertext is not compatible with
    /// the modulus of any of the relinearization keys in the bundle.
    void relin(ICiphertext &op, const KeyPtrBundle &leveled_relin_keys);

    /// @brief Rotates a ciphertext by a specified step using a rotation
    /// key.
    /// @param[in] op The ciphertext to be rotated.
    /// @param[in] step The rotation step.
    /// @param[out] res The resulting rotated ciphertext.
    /// @param[in] key The rotation key.
    /// @details Positive step indicates left rotation, while negative step
    /// indicates right rotation.
    /// @throws if op and key are on different devices.
    /// @throws if power of input ciphertext is not 1.
    void rot(const ICiphertext &op, i32 step, ICiphertext &res,
             const ISwKey &key) const;

    /// @brief Rotates a ciphertext by specified steps using a bundle of leveled
    /// rotation keys.
    /// @param[in] op The ciphertext to be rotated.
    /// @param[in] steps The rotation steps.
    /// @param[out] res The resulting rotated ciphertexts. The size of res
    /// should be the same as the size of steps.
    /// @param[in] keys The bundle of leveled rotation keys. The size of keys
    /// should be the same as the size of steps.
    /// @details Positive steps indicate left rotations, while negative steps
    /// indicate right rotations.
    /// @throws if op and keys are on different devices.
    /// @throws if power of input ciphertext is not 1.
    void rot(const ICiphertext &op, const std::vector<i32> &steps,
             const std::vector<ICiphertext *> &res,
             const RotKeyPtrs &keys) const;

    /// @brief Conjugates a ciphertext using a conjugation key.
    /// @param[in] op The ciphertext to be conjugated.
    /// @param[out] res The resulting conjugated ciphertext.
    /// @param[in] key The conjugation key.
    /// @throws if op and key are on different devices.
    /// @throws if power of input ciphertext is not 1.
    void conj(const ICiphertext &op, ICiphertext &res, const ISwKey &key) const;

    /// @brief Switches the key of a ciphertext using a switching key.
    /// @param[in] op The ciphertext encrypted under s_from.
    /// @param[out] res The resulting ciphertext encrypted under s_to.
    /// @param[in] key The switching key from s_from to s_to.
    /// @throws if op and key are on different devices.
    /// @throws if power of input ciphertext is not 1.
    void switchKey(const ICiphertext &op, ICiphertext &res,
                   const ISwKey &key) const;

    // Representation Modifiers
    /// @brief Rescales a plaintext to reduce its scale.
    /// @param[in] op The plaintext to be rescaled.
    /// @param[out] res The resulting rescaled plaintext.
    /// @details Rescaling consumes one level.
    /// @throws if the level of the plaintext is zero.
    void rescale(const IPlaintext &op, IPlaintext &res) const;
    /// @brief Rescales a ciphertext to reduce its scale.
    /// @param[in] op The ciphertext to be rescaled.
    /// @param[out] res The resulting rescaled ciphertext.
    /// @details Rescaling consumes one level.
    /// @throws if the level of the ciphertext is zero.
    void rescale(const ICiphertext &op, ICiphertext &res) const;
    /// @brief Lowers the level of a plaintext by one.
    /// @param[in] op The plaintext to be leveled down.
    /// @param[out] res The resulting leveled down plaintext.
    /// @details Leveling down consumes one level.
    /// @throws if the level of the plaintext is zero.
    void levelDown(const IPlaintext &op, IPlaintext &res) const;
    /// @brief Lowers the level of a ciphertext by one.
    /// @param[in] op The ciphertext to be leveled down.
    /// @param[out] res The resulting leveled down ciphertext.
    /// @details Leveling down consumes one level.
    /// @throws if the level of the ciphertext is zero.
    void levelDown(const ICiphertext &op, ICiphertext &res) const;
    /// @brief Lowers the level of a plaintext to a specified
    /// level.
    /// @param[in] op The plaintext to be leveled down.
    /// @param[out] res The resulting leveled down plaintext.
    /// @param[in] to The target level.
    /// @details The resulting plaintext will have the level equal to to.
    /// @throws if the level of the plaintext is less than to.
    void levelDownTo(const IPlaintext &op, IPlaintext &res, u32 to) const;
    /// @brief Lowers the level of a ciphertext to a specified
    /// level.
    /// @param[in] op The ciphertext to be leveled down.
    /// @param[out] res The resulting leveled down ciphertext.
    /// @param[in] to The target level.
    /// @details The resulting ciphertext will have the level equal to to.
    /// @throws if the level of the ciphertext is less than to.
    void levelDownTo(const ICiphertext &op, ICiphertext &res, u32 to) const;

    // Optimizations
    /// @brief Multiplies a plaintext by a Gaussian integer.
    /// @param[in] op1 The plaintext.
    /// @param[in] op2 The Gaussian integer.
    /// @param[out] res The resulting plaintext after multiplication.
    /// @throws if op2 has a nonzero imaginary part and op1 is not
    /// forwardNTTed.
    /// @details The resulting plaintext will have the scale = scale of op1.
    /// Thus, multiplication by Gaussian integer does not consume level.
    void mul(const IPlaintext &op1, GaussianInt op2, IPlaintext &res) const;
    /// @brief Multiplies a ciphertext by a Gaussian integer.
    /// @param[in] op1 The ciphertext.
    /// @param[in] op2 The Gaussian integer.
    /// @param[out] res The resulting ciphertext after multiplication.
    /// @throws if op2 has a nonzero imaginary part and op1 is not
    /// forwardNTTed.
    /// @details The resulting ciphertext will have the scale = scale of
    /// op1. Thus, multiplication by Gaussian integer does not consume
    /// level.
    void mul(const ICiphertext &op1, GaussianInt op2, ICiphertext &res) const;

    /// @brief Performs a fused tensor, relin, and rescale operation.
    /// @details This function is equivalent to calling tensor, relin, and
    /// rescale sequentially, but executes them as a single fused operation for
    /// improved performance.
    /// @details The fused implementation is slightly faster but incurs
    /// approximately one bit of precision loss compared to performing the three
    /// operations separately.
    /// @param[in] op1 The first ciphertext.
    /// @param[in] op2 The second ciphertext.
    /// @param[out] res The resulting ciphertext.
    /// @param[in] key The relinearization key.
    /// @throws if op1 and op2 have different moduli.
    /// @throws if op1, op2 and key are on different devices.
    /// @throws if power of op1 and op2 are not both 1.
    /// @throws if op1 or op2 is not forwardNTTed.
    void mulRescale(const ICiphertext &op1, const ICiphertext &op2,
                    ICiphertext &res, const ISwKey &key) const;

    /// @brief Performs NTT on the input plaintext.
    /// @param[in] op The input plaintext
    /// @param[out] res The resulting plaintext in NTT form.
    void forwardNTT(const IPlaintext &op, IPlaintext &res) const;
    /// @brief Performs NTT on the input ciphertext.
    /// @param[in] op The input ciphertext.
    /// @param[out] res The resulting ciphertext in NTT form.
    void forwardNTT(const ICiphertext &op, ICiphertext &res) const;
    /// @brief Performs iNTT on the input plaintext.
    /// @param[in] op The input plaintext.
    /// @param[out] res The resulting plaintext in iNTT form.
    void backwardNTT(const IPlaintext &op, IPlaintext &res) const;
    /// @brief Performs iNTT on the input ciphertext.
    /// @param[in] op The input ciphertext.
    /// @param[out] res The resulting ciphertext in iNTT form.
    void backwardNTT(const ICiphertext &op, ICiphertext &res) const;

    /// @brief Gets the homomorphic evaluation parameters.
    /// @return The Homomorphic Evaluation parameters.
    const HomEvalParams &getParams() const { return params_; }

private:
    Pimpl impl;
    HomEvalParams params_;
};

/// @brief Homomorphic evaluator supporting levelless operations
class HEAAN2_API HomEvalFlexible : public HomEval {
public:
    HomEvalFlexible();

    /// @brief Computes the tensor product of three ciphertexts.
    /// @param[in] op1 The first ciphertext.
    /// @param[in] op2 The second ciphertext.
    /// @param[in] op3 The third ciphertext.
    /// @param[out] res The resulting ciphertext after tensor product, which
    /// encrypts the multiplication of the messages of the input ciphertexts.
    /// @throws if op1, op2 and op3 have different moduli or different devices.
    /// @throws if power of op1, op2 and op3 are not all 1.
    /// @throws if op1, op2 or op3 is not forwardNTTed.
    /// @details The scale of res will be the product of scales of op1, op2 and
    /// op3. The resulting ciphertext has power 3. To reduce it back to a
    /// power-one ciphertext, use the relinCube function with appropriate
    /// relinearization key and relinearization cube key.
    /// @details if the number of slots of op1, op2 and op3 differ, the smaller
    /// one will be broadcasted to match the larger one.
    void tensor(const ICiphertext &op1, const ICiphertext &op2,
                const ICiphertext &op3, ICiphertext &res) const;

    /// @brief Relinearizes a ciphertext using a relinearization key and a
    /// relinearization cube key.
    /// @param[inout] op The ciphertext to be relinearized.
    /// @param[in] relin_key The relinearization key.
    /// @param[in] relin_cube_key The relinearization cube key.
    /// @details relinCube is an in-place operation that modifies the input
    /// ciphertext to reduce its power back to one. If the rescaling is desired
    /// after relinearization, the rescale function should be called separately.
    /// @throws if op, relin_key and relin_cube_key are on different devices.
    /// @throws if the power of the input ciphertext is not 3.
    void relinCube(ICiphertext &op, const ISwKey &relin_key,
                   const ISwKey &relin_cube_key) const;

    /// @brief Rescales a ciphertext to reduce its scale
    /// @param[in] op The ciphertext to be rescaled.
    /// @param[out] res The resulting rescaled ciphertext.
    /// @param[in] mod_to The target modulus for rescaled ciphertext
    /// @details Rescaling consumes modulus
    /// @throws if the target modulus is not compatible with the modulus of the
    /// input ciphertext
    void rescale(const ICiphertext &op, ICiphertext &res,
                 const PolyMod &mod_to) const;

    /// @brief Adjusts a ciphertext to a target modulus and scale.
    /// @param[in] op The ciphertext to be adjusted.
    /// @param[out] res The resulting adjusted ciphertext.
    /// @param[in] mod_to The target modulus for the adjusted ciphertext.
    /// @param[in] scale_to The target scale for the adjusted ciphertext.
    /// @details adjust lowers the modulus of the ciphertext to mod_to and sets
    /// its scale to scale_to by multiplying the scale difference.
    /// @throws if the target modulus is not compatible with the modulus of the
    /// input ciphertext.
    void adjust(const ICiphertext &op, ICiphertext &res, const PolyMod &mod_to,
                Real128 scale_to) const;

    /// @brief Performs a fused tensor, relinCube, and rescale operation.
    /// @details This function is equivalent to calling tensor, relinCube, and
    /// rescale sequentially, but executes them as a single fused operation for
    /// improved performance.
    /// @details The fused implementation is slightly faster but incurs
    /// approximately one bit of precision loss compared to performing the three
    /// operations separately.
    /// @param[in] op1 The first ciphertext.
    /// @param[in] op2 The second ciphertext.
    /// @param[in] op3 The third ciphertext.
    /// @param[out] res The resulting ciphertext.
    /// @param[in] relin_key The relinearization key.
    /// @param[in] relin_cube_key The relinearization cube key.
    /// @throws if op1, op2 and op3 have different moduli.
    /// @throws if op1, op2, op3, relin_key and relin_cube_key are on different
    /// devices.
    /// @throws if power of op1, op2 and op3 are not all 1.
    /// @throws if op1, op2 or op3 is not forwardNTTed.
    void mulRescale(const ICiphertext &op1, const ICiphertext &op2,
                    const ICiphertext &op3, ICiphertext &res,
                    const ISwKey &relin_key, const ISwKey &relin_cube_key,
                    const PolyMod &mod_to) const;

    /// @brief Modifies the scale of a ciphertext.
    /// @param[inout] op The ciphertext.
    /// @param[in] scale The target scale.
    /// @details setScale modifies only metadata scale. It does not change the
    /// polynomial data of the ciphertext.
    void setScale(ICiphertext &op, Real128 scale) const;

    /// @brief Set the DFT flag of a plaintext.
    /// @param[inout] op The plaintext.
    /// @param[in] dft The target DFT flag.
    /// @details setDFT modifies only metadata DFT flag. It does not change the
    /// polynomial data of the plaintext.
    void setDFT(IPlaintext &op, bool dft) const;

    /// @brief Set the DFT flag of a ciphertext.
    /// @param[inout] op The ciphertext.
    /// @param[in] dft The target DFT flag.
    /// @details setDFT modifies only metadata DFT flag. It does not change the
    /// polynomial data of the ciphertext.
    void setDFT(ICiphertext &op, bool dft) const;

    /// @brief Copies the ciphertexts into one BatchRLWE ciphertext
    /// @param[in] op The vector of raw pointers to the ciphertexts to be
    /// copied.
    /// @param[out] res The resulting batched ciphertext.
    /// @throws if the encryption type of res is not BatchRLWE.
    /// @throws if the encryption types or batch sizes of op are different.
    /// @throws if the ciphertexts in op are not all forwardNTTed or all
    /// backwardNTTed.
    /// @details The encryption types of op can be RLWE or BatchRLWE.
    void batch(const std::vector<const ICiphertext *> &op,
               ICiphertext &res) const;

    /// @brief Copies the BatchRLWE ciphertext into the ciphertexts
    /// @param[in] op The ciphertext to be copied.
    /// @param[out] res The vector of raw pointers to the resulting ciphertexts
    /// @throws if the encryption type of op is not BatchRLWE.
    /// @throws if the encryption types or batch sizes of res are different.
    /// @throws if the size of res does not divide the batch size of op and
    /// the outputs are BatchRLWE.
    /// @throws if the size of res is not equal to the batch size of op and
    /// the outputs are RLWE.
    /// @details The encryption types of res can be RLWE or BatchRLWE.
    void unbatch(const ICiphertext &op,
                 const std::vector<ICiphertext *> &res) const;
};

} // namespace heaan

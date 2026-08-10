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

#include "Bootstrapper.hpp"
#include "ICiphertext.hpp"
#include "IEncKey.hpp"
#include "IPlaintext.hpp"
#include "ISecretKey.hpp"
#include "ISwKey.hpp"
#include "KeyUtils.hpp"
#include "Message.hpp"

#include "Device.hpp"
#include "Export.hpp"
#include "Ptr.hpp"

#include <fstream>
#include <iostream>

namespace heaan::serial {

template <typename T> void save(std::ostream &os, const T &op) = delete;

template <typename T>
HEAAN2_API void save(const std::string &filename, const T &op);

template <typename T>
T load(std::istream &is, const Device &device = {Device::CPU}) = delete;

template <typename T>
Ptr<T> loadAsPtr(std::istream &is,
                 const Device &device = {Device::CPU}) = delete;

template <typename T>
HEAAN2_API T load(const std::string &filename,
                  const Device &device = {Device::CPU});

template <typename T>
HEAAN2_API Ptr<T> loadAsPtr(const std::string &filename,
                            const Device &device = {Device::CPU});

template <> HEAAN2_API void save(std::ostream &, const heaan::Message &);
template <> HEAAN2_API void save(std::ostream &, const heaan::Message128 &);
template <> HEAAN2_API void save(std::ostream &, const heaan::IPlaintext &);
template <> HEAAN2_API void save(std::ostream &, const heaan::ICiphertext &);
template <> HEAAN2_API void save(std::ostream &, const heaan::ISecretKey &);
template <> HEAAN2_API void save(std::ostream &, const heaan::IEncKey &);
template <> HEAAN2_API void save(std::ostream &, const heaan::ISwKey &);
template <> HEAAN2_API void save(std::ostream &, const heaan::RotKeyPtrs &);
template <> HEAAN2_API void save(std::ostream &, const heaan::KeyPtrBundle &);
template <> HEAAN2_API void save(std::ostream &, const heaan::BootKeyPtrs &);

template <> HEAAN2_API heaan::Message load(std::istream &, const Device &);
template <> HEAAN2_API heaan::Message128 load(std::istream &, const Device &);
template <> HEAAN2_API heaan::RotKeyPtrs load(std::istream &, const Device &);
template <> HEAAN2_API heaan::KeyPtrBundle load(std::istream &, const Device &);
template <> HEAAN2_API heaan::BootKeyPtrs load(std::istream &, const Device &);
template <>
HEAAN2_API Ptr<heaan::IPlaintext> loadAsPtr(std::istream &, const Device &);
template <>
HEAAN2_API Ptr<heaan::ICiphertext> loadAsPtr(std::istream &, const Device &);
template <>
HEAAN2_API Ptr<heaan::ISecretKey> loadAsPtr(std::istream &, const Device &);
template <>
HEAAN2_API Ptr<heaan::IEncKey> loadAsPtr(std::istream &, const Device &);
template <>
HEAAN2_API Ptr<heaan::ISwKey> loadAsPtr(std::istream &, const Device &);

} // namespace heaan::serial

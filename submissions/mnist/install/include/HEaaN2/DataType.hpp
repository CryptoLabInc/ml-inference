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

#include <algorithm>
#include <complex>
#include <cstdint>

// Use long double as 128-bit floating point if it has enough precision,
// otherwise use __float128.
#if defined(__LDBL_MANT_DIG__) && __LDBL_MANT_DIG__ == 113
#define HEAAN2_REAL128_LD 1
#else
#define HEAAN2_REAL128_LD 0
#endif

namespace heaan {

__extension__ using u128 = unsigned __int128;
__extension__ using i128 = __int128;
using u64 = std::uint64_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;

inline HEAAN2_API std::ostream &operator<<(std::ostream &os,
                                           __uint128_t value) {
    if (value == 0) {
        os << '0';
        return os;
    }
    std::string result;
    while (value > 0) {
        result.push_back(static_cast<char>('0' + value % 10));
        value /= 10;
    }
    std::reverse(result.begin(), result.end());
    os << result;
    return os;
}

using Real = double;
#if HEAAN2_REAL128_LD
using Real128 = long double;
#else
using Real128 = __float128;

inline HEAAN2_API std::ostream &operator<<(std::ostream &os, Real128 value) {
    os << static_cast<double>(value);
    return os;
}
#endif

using Complex = std::complex<Real>;
struct HEAAN2_API Complex128 : public std::complex<Real128> {
    using std::complex<Real128>::complex;
    Complex128() = default;
    Complex128(const std::complex<Real128> &v) : std::complex<Real128>(v) {}
};

inline HEAAN2_API std::ostream &operator<<(std::ostream &os, Complex128 value) {
    os << "(" << static_cast<double>(value.real()) << ", "
       << static_cast<double>(value.imag()) << ")";
    return os;
}

struct HEAAN2_API GaussianInt {
    explicit GaussianInt(i32 real_in) : real(real_in), imag(0) {}
    explicit GaussianInt(i32 real_in, i32 imag_in)
        : real(real_in), imag(imag_in) {}
    explicit GaussianInt(std::initializer_list<i32> list) {
        if (list.size() > 2) {
            throw std::invalid_argument(
                "[GaussianInt] Too many arguments in initializer list.");
        }
        const auto *it = list.begin();
        real = *it;
        imag = list.size() == 1 ? 0 : *(++it);
    }
    i32 real;
    i32 imag;
};

} // namespace heaan

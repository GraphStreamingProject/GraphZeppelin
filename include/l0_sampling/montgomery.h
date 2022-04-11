#pragma once
#include <boost/multiprecision/cpp_int.hpp>

namespace Montgomery {
  namespace mp = boost::multiprecision;
  using namespace mp::literals;

  /**
   * Sets x and y so that ax - by = gcd(a, b)
   * a,b should be in (0, 2^32]
   */
  inline mp::uint128_t extended_gcd(const mp::uint256_t a, const mp::uint256_t b,
      mp::uint128_t& x, mp::uint128_t& y) {
    mp::uint128_t s = 0, t = 1;
    mp::uint256_t m = a, n = b;
    x = 1;
    y = 0;
    while (n) {
      mp::uint128_t quot = static_cast<mp::uint128_t>(m / n);
      m -= quot * n;
      x += quot * s;
      y += quot * t;
      if (!m) {
        x = static_cast<mp::uint128_t>(b - s);
        y = static_cast<mp::uint128_t>(a - t);
        return static_cast<mp::uint128_t>(n);
      }
      quot = static_cast<mp::uint128_t>(n / m);
      n -= quot * m;
      s += quot * x;
      t += quot * y;
    }
    return static_cast<mp::uint128_t>(m);
  }

  /**
   * Montgomery multiplication using aux mod 2^128
   */
  class Ctx {
    mp::uint128_t rinv;
    mp::uint128_t mod;
    mp::uint128_t modinv;

    public:
    Ctx(mp::uint128_t mod) : mod(mod) {
      assert(extended_gcd(0x1_cppui256 << 128, mod, rinv, modinv) == 1);
    }

    inline mp::uint128_t toMont(mp::uint128_t a) const {
      return static_cast<mp::uint128_t>((static_cast<mp::uint256_t>(a) << 128) % mod);
    }

    inline mp::uint128_t fromMont(mp::uint128_t a) const {
      return static_cast<mp::uint128_t>((static_cast<mp::uint256_t>(a) * rinv) % mod);
    }

    inline mp::uint128_t mult(mp::uint128_t a, mp::uint128_t b) const {
      return redc(static_cast<mp::uint256_t>(a) * b);
    }

    inline mp::uint128_t redc(mp::uint256_t t) const {
      mp::uint256_t tmodr = static_cast<mp::uint128_t>(t);
      t = (static_cast<mp::uint256_t>(static_cast<mp::uint128_t>(tmodr * modinv)) * mod >> 128) + (t >> 128) + (tmodr ? 1 : 0);
      return static_cast<mp::uint128_t>(t < mod ? t : t - mod);
    }
  };
}

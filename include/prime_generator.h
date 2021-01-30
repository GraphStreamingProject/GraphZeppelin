#pragma once
#include <boost/multiprecision/cpp_int.hpp>
#include "montgomery.h"

namespace mp = boost::multiprecision;

namespace PrimeGenerator{
/*
  inline mp::uint128_t powermod(mp::uint128_t a, mp::uint128_t b, const Montgomery::Ctx& ctx) {
    mp::uint128_t res = ctx.toMont(1);
    a = ctx.toMont(a);
    while (b) {
      if (b & 1) {
        res = ctx.mult(res, a);
      }
      b >>= 1;
      a = ctx.mult(a, a);
    }
    return ctx.fromMont(res);
  }
/*/
  inline mp::uint128_t powermod(mp::uint128_t a, mp::uint128_t b, mp::uint128_t m) {
    mp::uint128_t res = 1;
    while (b) {
      if (b & 1) {
        res = static_cast<mp::uint128_t>(static_cast<mp::uint256_t>(res) * a % m);
      }
      b >>= 1;
      a = static_cast<mp::uint128_t>(static_cast<mp::uint256_t>(a) * a % m);
    }
    return res;
  }

  //Generate a^2^i for i in [0, log_2(n)]
  inline std::vector<mp::uint128_t> gen_sq_cache(mp::uint128_t a, mp::uint128_t n, mp::uint128_t m) {
    //TODO: actual log2 for uint128_t, if we're even using 128 bit vector size
    unsigned log2n = log2(static_cast<uint64_t>(n));
    std::vector<mp::uint128_t> sq_cache(log2n + 1);
    sq_cache[0] = a;
    for (unsigned i = 0; i < log2n; i++) {
      sq_cache[i + 1] = static_cast<mp::uint128_t>(static_cast<mp::uint256_t>(sq_cache[i]) * sq_cache[i] % m);
    }
    return sq_cache;
  }

  inline mp::uint128_t cached_powermod(const std::vector<mp::uint128_t>& sq_cache, mp::uint128_t b, mp::uint128_t m) {
    unsigned i = 0;
    mp::uint128_t res = 1;
    while (b) {
      if (b & 1) {
        res = static_cast<mp::uint128_t>(static_cast<mp::uint256_t>(res) * sq_cache[i] % m);
      }
      b >>= 1;
      i++;
    }
    return res;
  }
//*/

  static bool IsPrime(mp::uint128_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0) return false;

    const mp::uint128_t iMax = sqrt(n) + 1;
    for (mp::uint128_t i = 3; i <= iMax; i += 2)
      if (n % i == 0)
        return false;

    return true;
  }
  //Generates a prime number greater than or equal to n
  inline mp::uint128_t generate_prime(mp::uint128_t n){
    if (n % 2 == 0){
      n++;
    }
    while(!IsPrime(n)){
      n += 2;
    }
    return n;
  }
}

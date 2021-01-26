#pragma once
#include <boost/multiprecision/cpp_int.hpp>
#include "montgomery.h"

using boost::multiprecision::int128_t;
using boost::multiprecision::uint128_t;

namespace PrimeGenerator{
  inline uint128_t powermod(uint128_t a, uint128_t b, const Montgomery::Ctx& ctx) {
    uint128_t res = ctx.toMont(1);
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

  static bool IsPrime(uint128_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0) return false;

    const uint128_t iMax = sqrt(n) + 1;
    for (uint128_t i = 3; i <= iMax; i += 2)
      if (n % i == 0)
        return false;

    return true;
  }
  //Generates a prime number greater than or equal to n
  inline uint128_t generate_prime(uint128_t n){
    if (n % 2 == 0){
      n++;
    }
    while(!IsPrime(n)){
      n += 2;
    }
    return n;
  }
}

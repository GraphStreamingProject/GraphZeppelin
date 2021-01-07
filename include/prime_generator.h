#pragma once
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::int128_t;
using boost::multiprecision::uint128_t;

namespace PrimeGenerator{
  int128_t power(int128_t x, uint128_t y, uint128_t p){
    int128_t res = 1;
    x = x % p;
    while (y > 0)
    {
      if (y & 1)
        res = (res*x) % p;
      y = y>>1;
      x = (x*x) % p;
    }
    return res;
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
  uint128_t generate_prime(uint128_t n){
    if (n % 2 == 0){
      n++;
    }
    while(IsPrime(n)){
      n += 2;
    }
    return n;
  }
}

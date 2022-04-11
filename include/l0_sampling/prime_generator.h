#pragma once
#include <vector>
#include <cmath>
#include "../types.h"

namespace PrimeGenerator{
  inline ubucket_t powermod(ubucket_t a, ubucket_t b, ubucket_t m) {
    ubucket_t res = 1;
    while (b) {
      if (b & 1) {
        res = static_cast<ubucket_t>(static_cast<ubucket_prod_t>(res) * a % m);
      }
      b >>= 1;
      a = static_cast<ubucket_t>(static_cast<ubucket_prod_t>(a) * a % m);
    }
    return res;
  }

  //Generate a^2^i for i in [0, log_2(n)]
  inline std::vector<ubucket_t> gen_sq_cache(ubucket_t a, ubucket_t n, ubucket_t m) {
    unsigned log2n = log2(n);
    std::vector<ubucket_t> sq_cache(log2n + 1);
    sq_cache[0] = a;
    for (unsigned i = 0; i < log2n; i++) {
      sq_cache[i + 1] = static_cast<ubucket_t>(static_cast<ubucket_prod_t>(sq_cache[i]) * sq_cache[i] % m);
    }
    return sq_cache;
  }

  inline ubucket_t cached_powermod(const std::vector<ubucket_t>& sq_cache, ubucket_t b, ubucket_t m) {
    unsigned i = 0;
    ubucket_t res = 1;
    while (b) {
      if (b & 1) {
        res = static_cast<ubucket_t>(static_cast<ubucket_prod_t>(res) * sq_cache[i] % m);
      }
      b >>= 1;
      i++;
    }
    return res;
  }

  static bool IsPrime(ubucket_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0) return false;

    const ubucket_t iMax = sqrt(n) + 1;
    for (ubucket_t i = 3; i <= iMax; i += 2)
      if (n % i == 0)
        return false;

    return true;
  }
  //Generates a prime number greater than or equal to n
  inline ubucket_t generate_prime(ubucket_t n){
    if (n % 2 == 0){
      n++;
    }
    while(!IsPrime(n)){
      n += 2;
    }
    return n;
  }
}

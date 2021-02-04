#pragma once

typedef long long int ll;
typedef unsigned long long int ull;

namespace PrimeGeneratorNative {
  inline ll power(ll x, ull y, ull p){
      ll res = 1;
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
  // not an implementation of AKS
  static bool IsPrime(unsigned int n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0) return false;

    const unsigned int iMax = (int)sqrt(n) + 1;
    unsigned int i;
    for (i = 3; i <= iMax; i += 2)
      if (n % i == 0)
        return false;

    return true;
  }
  //Generates a prime number greater than or equal to n
  inline ull generate_prime(ull n){
    if (n % 2 == 0){
      n++;
    }
    while(!IsPrime(n)){
      n += 2;
    }
    return n;
  }
}

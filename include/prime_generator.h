#pragma once
#include <boost/multiprecision/cpp_int.hpp>
#include "aks.h"

typedef long long int ll;
typedef unsigned long long int ull;

using boost::multiprecision::int128_t;
using boost::multiprecision::uint128_t;

namespace PrimeGenerator{
//  ll power(ll x, ull y, ull p){
//      ll res = 1;
//      x = x % p;
//      while (y > 0)
//      {
//          if (y & 1)
//              res = (res*x) % p;
//          y = y>>1;
//          x = (x*x) % p;
//      }
//      return res;
//  }
//  //Generates a prime number greater than or equal to n
//  ull generate_prime(ull n){
//    if (n % 2 == 0){
//      n++;
//    }
//    while(!AKS::IsPrime(n)){
//      n += 2;
//    }
//    return n;
//  }

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
  //Generates a prime number greater than or equal to n
  uint128_t generate_prime(uint128_t n){
    if (n % 2 == 0){
      n++;
    }
    while(!AKS::IsPrime(n)){
      n += 2;
    }
    return n;
  }
}

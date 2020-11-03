#pragma once
#include "aks.h"

typedef long long int ll;
typedef unsigned long long int ull;

namespace PrimeGenerator{
  ll power(ll x, ull y, ull p){
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
  //Generates a prime nuumber greater than or equal to n
  ull generate_prime(ull n){
    if (n % 2 == 0){
      n++;
    }
    while(!AKS::IsPrime(n)){
      n += 2;
    }
    return n;
  }
}

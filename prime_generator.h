#pragma once
#include "aks.h"

namespace PrimeGenerator{
  long power(long  x, unsigned long  y, long  p){
      long int res = 1;
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
  unsigned long generate_prime(unsigned long n){
  	if (n % 2 == 0){
  		n++;
  	}
    while(!AKS::IsPrime(n)){
  		n += 2;
  	}
  	return n;
  }
}

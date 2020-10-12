#pragma once
#include "aks.h"

namespace PrimeGenerator{
  int power(int x, unsigned int y, int p){
      int res = 1;
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
  int generate_prime(int n){
  	if (n % 2 == 0){
  		n++;
  	}
  	while(!AKS::IsPrime(n)){
  		n += 2;
  	}
  	return n;
  }
}

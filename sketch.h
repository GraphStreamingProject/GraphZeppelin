#pragma once
#include <math.h>
#include <cmath>
#include <assert.h>
#include <vector>
#include <iostream>
#include <string>
#include "prime_generator.h"
#include "update.h"
#include "bucket.h"
using namespace std;

struct Sketch{
  const long seed;
  const int n;
  std::vector<Bucket> buckets;
  const unsigned long random_prime;
  bool already_quered = false;

  //Initialize a sketch of a vector of size n
public:
  Sketch(int n, long seed): n(n), seed(seed), random_prime(PrimeGenerator::generate_prime(n*n)) {
  	std::cout << "Prime: " << random_prime << std::endl;
    const int num_buckets = log2(n);
  	std::cout << "Number of buckets: " << num_buckets << std::endl;
  	buckets = std::vector<Bucket>(num_buckets*(num_buckets+1));
  	for (unsigned int i = 0; i < num_buckets; ++i){
  		for (unsigned int j = 0; j < num_buckets+1; ++j){
  			buckets[i*(num_buckets+1)+j].set_guess(1 << j);
        buckets[i*(num_buckets+1)+j].set_seed(i*(num_buckets+1)+j,seed,random_prime);
  		}
  	}
  }

  void update(Update update ){
    for (unsigned int j = 0; j < buckets.size(); j++){
			if (buckets[j].contains(update.index)){
				buckets[j].a += update.delta;
				buckets[j].b += update.delta*update.index;
				buckets[j].c += (update.delta*PrimeGenerator::power(buckets[j].r,update.index,random_prime))%random_prime;
			}
		}
  }

  Update query(){
    if (already_quered){
      std::cerr << "This sketch has already been sampled!\n";
      exit(1);
    }
    already_quered = true;
    for (int i = 0; i < buckets.size(); i++){
  		Bucket& b = buckets[i];
  		if ( b.a != 0 && b.b % b.a == 0 && (b.c - b.a*PrimeGenerator::power(b.r,b.b/b.a,random_prime))% random_prime == 0  ){
  			//cout << "Passed all tests: " << "b.a: " << b.a << " b.b: " << b.b << " b.c: " << b.c << " Guess: " << b.guess_nonzero << " r: " << b.r << endl;
        //cout << "String: " << b.stuff << endl;
        return {b.b/b.a,b.a};
  		}
  	}
    std::cerr << "Found no good bucket!\n";
    exit(1);
  }

  friend Sketch operator+ (const Sketch &sketch1, const Sketch &sketch2);
  friend Sketch operator* (const Sketch &sketch1, long scaling_factor );
};

Sketch operator+ (const Sketch &sketch1, const Sketch &sketch2){
  assert (sketch1.n == sketch2.n);
  assert (sketch1.seed == sketch2.seed);
  assert (sketch1.random_prime == sketch2.random_prime);
  Sketch result = Sketch(sketch1.n,sketch1.seed);
  for (int i = 0; i < result.buckets.size(); i++){
    Bucket& b = result.buckets[i];
    b.a = sketch1.buckets[i].a + sketch2.buckets[i].a;
    b.b = sketch1.buckets[i].b + sketch2.buckets[i].b;
    b.c = (sketch1.buckets[i].c + sketch2.buckets[i].c)%result.random_prime;
  }
  return result;
}

Sketch operator* (const Sketch &sketch1, long scaling_factor){
  Sketch result = Sketch(sketch1.n,sketch1.seed);
  for (int i = 0; i < result.buckets.size(); i++){
    Bucket& b = result.buckets[i];
    b.a = sketch1.buckets[i].a * scaling_factor;
    b.b = sketch1.buckets[i].b * scaling_factor;
    b.c = (sketch1.buckets[i].c * scaling_factor)% result.random_prime;
  }
  return result;
}

#pragma once
#include <xxhash.h>
#include <iostream>
#include <string>
using namespace std;

/**
 * Represents a bucket in a sketch.
 */
struct Bucket{
  long long int a = 0;
  long long int b = 0;
  long long int c = 0;
  XXH64_hash_t bucket_seed;
  long guess_nonzero = 1;
  long long int r = -1;

  /**
   * Checks whether the hash associated with the Bucket hashes the index to 0.
   * @return true if the index is NOT hashed to zero.
   */
  bool contains(unsigned long long int index){
    XXH64_hash_t hash = XXH64(&index, 8, bucket_seed);
    if (hash % guess_nonzero == 0)
      return true;
    return false;
  }

  /**
   * Takes a given guess for mu.
   * @param guess the number of nonzero elements we assume.
   */
  void set_guess(long guess) {
    guess_nonzero = guess;
    //cout << "Guess: " << guess_nonzero << endl;
  }

  void set_seed(long bucket_id, long sketch_seed, long long int random_prime){
    bucket_seed = XXH64(&bucket_id ,8, sketch_seed);
    //Do not use r in {-1, 0, 1}, they make the c test less effective
    r = 2 + bucket_seed % (random_prime - 3);
    //cout << "r : " << r << endl;
  }
};

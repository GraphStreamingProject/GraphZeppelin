#pragma once
#include <xxhash.h>
#include <iostream>
#include <string>
using namespace std;

/**
 * Represents a bucket in a sketch.
 */
struct Bucket{
	long a = 0;
  long b = 0;
  long c = 0;
	XXH64_hash_t bucket_seed;
	long guess_nonzero = 1;
	long r = -1;
  std::string stuff = "";

  /**
   * Checks whether the hash associated with the Bucket hashes the index to 0.
   * @return true if the index is NOT hashed to zero.
   */
  bool contains(long index){
    // TODO: verify independence of hash functions based on independence of seeds
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

  /**
   * 
   */
  void set_seed(long bucket_id, long sketch_seed,long random_prime){
    bucket_seed = XXH64(&bucket_id ,8, sketch_seed);
    r = bucket_seed % random_prime;
    //cout << "r : " << r << endl;
  }
};

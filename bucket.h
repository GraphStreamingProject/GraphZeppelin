#pragma once
#include <xxhash.h>

struct bucket{
	int a,b,c;
	XXH64_hash_t seed;
	long long guess_nonzero = 1;
	long long r = -1;

	bucket(long long random_prime): a(0),b(0),c(0),seed(rand()){
		r = rand() % random_prime;
	}

	bool contains(int index){
	 XXH64_hash_t hash = XXH64(&index, 64, seed);
	 if (hash % guess_nonzero == 0){
		 return true;
	 }
	 return false;;
	}

	void set_guess(int guess){
		guess_nonzero = guess;
		//cout << "Guess: " << guess_nonzero << endl;
	}
};

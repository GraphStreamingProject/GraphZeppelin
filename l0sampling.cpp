#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <stdio.h>
#include "prime_generator.h"
#include "bucket.h"
using namespace std;

int main(){
	srand (time(NULL));
	const unsigned int n = 1000;
	const unsigned int updates = 1000;
	int vect[n] = {0};
	int random_prime = PrimeGenerator::generate_prime(n*n);
	cout << "Prime: " << random_prime << endl;
	//Define the update struct
	struct update{
		int index,delta;
	};

	update stream[updates];

	//Initialize the stream, and finalize the input vector.
	for (unsigned int i = 0; i < updates; i++){
		stream[i] = {rand()%n,rand()%10 - 5};
		vect[stream[i].index] += stream[i].delta;
		//cout << "Index: " << stream[i].index << " Delta: " << stream[i].delta << endl;
	}

	cout << "Done initializing vector" << endl;


	//Compute the value of mu
	float mu = 0;
	for (unsigned int i = 0; i < n; i++){
		//cout << "i : " << i << " Value: " << vect[i] << endl;
		mu += (vect[i] != 0 ) ? 1 : 0;
	}
	mu /= n;

	cout << "Value of mu: " << mu << endl;

	const int num_buckets = log2(n);
	cout << "Number of buckets: " << num_buckets << endl;
	vector<bucket> buckets(num_buckets*(num_buckets+1),bucket(random_prime));

	for (unsigned int i = 0; i < num_buckets; ++i){
		for (unsigned int j = 0; j < num_buckets+1; ++j){
			buckets[i*(num_buckets+1)+j].set_guess(1 << j);
		}
	}

	for (unsigned int i = 0; i < updates; i++){
		for (unsigned int j = 0; j < num_buckets*(num_buckets+1); j++){
			if (buckets[j].contains(stream[i].index)){
				//cout << "Yep\n";
				//cout << "Index: " << stream[i].index << " Delta: " << stream[i].delta << endl;
				buckets[j].a += stream[i].delta;
				buckets[j].b += stream[i].delta*stream[i].index;
				buckets[j].c += stream[i].delta*PrimeGenerator::power(buckets[j].r,stream[i].index,random_prime);
				//buckets[j].c += stream[i].delta*
			}
		}
	}
	cout << "Done going through stream\n";

	for (int i = 0; i < num_buckets*(num_buckets+1); i++){
		bucket& b = buckets[i];
		//cout << "bucket number: " << i << " Guessed number of nonzero: " << buckets[i].guess_nonzero << " b.a: " << b.a << " b.b: " << b.b << " b.c: " << b.c << endl;
		/*if (b.a != 0 && b.b % b.a == 0){
			cout << "Passed a/b test: " << "Index: " << b.b/b.a << " Value: " << b.a << endl;
		}*/
		if (b.a != 0 && b.b % b.a == 0 && (b.c - b.a*PrimeGenerator::power(b.r,b.b/b.a,random_prime))% random_prime == 0  ){
			cout << "Passed all tests: " << "Index: " << b.b/b.a << " Value: " << b.a << endl;
		}
		if (b.a != 0 && b.b % b.a == 0 && b.b/b.a > 0 && b.b/b.a < n && vect[b.b/b.a] == b.a ){
			cout << "Actual answer: " << "Index: " << b.b/b.a << " Value: " << b.a << endl;
		}
	}
	/*
	bucket sp(random_prime);
	int index = 11;
	int delta = -4;
	sp.a += delta;
	sp.b += index*delta;
	sp.c += delta*PrimeGenerator::power(sp.r,index,random_prime);
	cout << " b.a: " << sp.a << " b.b: " << sp.b << " b.c: " << sp.c << endl;
	if (sp.a != 0 && sp.b % sp.a == 0 && sp.b/sp.a > 0 && sp.b/sp.a < n && (sp.c - sp.a*PrimeGenerator::power(sp.r,sp.b/sp.a,random_prime))% random_prime == 0  ){
		cout << "Actual answer: " << "Index: " << sp.b/sp.a << " Value: " << sp.a << endl;
	}
	*/
	return 0;
}


/*
TODO:
- We will have log n sets of log n hash function.
- For each has function, we will

Is the memory usage really O(polylog n)?
If phi = number of nonzero indices, our hash function should return 1 with probability 1/phi.
Size of hash function

Tasks for implementing l_0 sampling:

replace the trivial hash function with xxhash.
rewrite code to determine whether an index belongs in a bucket
add in c code: note that p must be prime and larger than poly(n) = n^2
write code to guess mu (/ phi) instead of assuming it
make the code modular

Suppose we have many buckets guessing the number of nonzero entries. Suppose n
is 1024. Do we wish to make guesses in the range 1,2,4,8...1024 or 1,2,4,8...512?
*/

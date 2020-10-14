#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <stdio.h>
#include "update.h"
#include "sketch.h"
using namespace std;

int main(){
	srand (time(NULL));
	const long n = 1000000;
	const long updates = 100;
	const long updates2 = 100;
	int vect[n] = {0};
	int vect2[n] = {0};

	Update stream[updates];
	Update stream2[updates2];

	//Initialize the stream, and finalize the input vector.
	for (unsigned int i = 0; i < updates; i++){
		stream[i] = {(long)rand()%n,rand()%10 - 5};
		stream2[i] = {(long)rand()%n,rand()%10 - 5};
		vect[stream[i].index] += stream[i].delta;
		vect2[stream2[i].index] += stream2[i].delta;
		//cout << "Index: " << stream[i].index << " Delta: " << stream[i].delta << endl;
	}

	cout << "Done initializing vector" << endl;

	//cout << "Value of mu: " << mu << endl;
	int random = rand();
	Sketch sketch = Sketch(n,random);
	Sketch sketch2 = Sketch(n, random);

	for (unsigned int i = 0; i < updates; i++){
		sketch.update(stream[i]);
	}
	for (unsigned int i = 0; i < updates2; i++){
		sketch2.update(stream2[i]);
	}
	cout << "Done going through streams\n";

	Sketch sum = sketch + sketch2;
	Update update = sum.query();
	cout << update << endl;
	cout << "The correct value at that index was: " << to_string(vect[update.index]+vect2[update.index]) << endl;
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

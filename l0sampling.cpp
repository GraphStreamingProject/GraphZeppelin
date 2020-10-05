#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <vector>
using namespace std;

bool contains(vector<int>& array,int element){
	//cout << "array size: " << array.size() << endl;
	for (int e : array){
		//cout << "Array element: " << e << endl;
		if (e == element)
			return true;
	}
	return false;
}

int main(){
	srand (time(NULL));
	const unsigned int n = 100;
	const unsigned int updates = 100;
	int vect[n] = {0};

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
		cout << "i : " << i << " Value: " << vect[i] << endl;
		mu += (vect[i] != 0 ) ? 1 : 0;	
	}
	mu /= n;
	static unsigned int indices_per_bucket = 1/mu;
	
	cout << "Value of mu: " << mu << endl;

	struct bucket{
		int a,b,c;
		vector<int> indices;
		
		bucket(): a(0),b(0),c(0){
			indices.resize(indices_per_bucket);
			for (unsigned int i = 0; i < indices_per_bucket; i++){
				indices[i] = (rand()%n);
			}
			cout << "Bucket\n";
			for (unsigned int i = 0; i < indices_per_bucket; i++){
				cout << "index: " << indices[i] << endl;
			}
			//cout << "SIZE: " << indices.size() << endl;
		}
	};

	const int num_buckets = log(n);
	cout << "Number of buckets: " << num_buckets << endl;
	bucket buckets[num_buckets];
	

	for (unsigned int i = 0; i < updates; i++){
		for (unsigned int j = 0; j < num_buckets; j++){
			if (contains(buckets[j].indices,stream[i].index)){
				cout << "Yep\n";
				cout << "Index: " << stream[i].index << " Delta: " << stream[i].delta << endl;
				cout << "j : " << j << " Array size: " << buckets[j].indices.size() << endl;
				buckets[j].a += stream[i].delta;
				buckets[j].b += stream[i].delta*stream[i].index;
			}
		}
	}
	
	cout << "Done going through stream\n";

	for (int i = 0; i < num_buckets; i++){
		bucket b = buckets[i];
		cout << "bucket number: " << i << " b.a: " << b.a << " b.b: " << b.b << endl;
		if (b.a != 0 && b.b % b.a == 0){
			cout << b.b/b.a << endl;
			exit(0);
		}
	}
	return 0;
}






/*
Is the memory usage really O(polylog n)?

*/
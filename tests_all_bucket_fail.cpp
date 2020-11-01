#include  <criterion/criterion.h>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include "sketch.h"
#include "update.h"

Test(sketch_test_suite, testing_all_bucket_fail_probability){
	srand (time(NULL));
	const unsigned long num_sketches = 1000000;
	const double max_fail_probability = .01;
	
	unsigned long all_bucket_failures = 0;
	for (int i = 0; i < num_sketches; i++){
		const unsigned long vect_size = 1000;
		const unsigned long num_updates = 1000;
		Sketch sketch = Sketch(vect_size, rand());
		for (unsigned long j = 0; j < num_updates; j++){
			sketch.update({(long)(rand() % vect_size), rand() % 11 - 5});
		}
		Update result = sketch.query();
		
		if (result.delta == 0) {
			all_bucket_failures++;
		}
	}
	cr_assert(all_bucket_failures < max_fail_probability * num_sketches, "All buckets failed %lu/%lu times (expected less than %lf)", all_bucket_failures, num_sketches, max_fail_probability);
}

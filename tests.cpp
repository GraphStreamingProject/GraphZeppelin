#include  <criterion/criterion.h>
#include <cstdlib>
#include <iostream>
#include "sketch.h"
#include "testing_vector.h"
#include "update.h"

Test(sketch_test_suite, testing_sketch_addition){
	srand (time(NULL));
	for (int i = 0; i < 1000; i++){
		std::cout << "Test number: " << i << std::endl;
		const unsigned long vect_size = 1000;
		const unsigned long num_updates = 1000;
		Sketch sketch = Sketch(vect_size,rand());
		Testing_Vector test_vector = Testing_Vector(vect_size,num_updates);
		for (unsigned long j = 0; j < num_updates; j++){
				sketch.update(test_vector.get_update(j));
		}
		Update result = sketch.query();
		if (test_vector.get_entry(result.index) == 0 || test_vector.get_entry(result.index) != result.delta ){
			cr_assert(0);
		}
	}
}

#include  <criterion/criterion.h>
<<<<<<< HEAD
#include "sketch.h"
#include "testing_vector.h"
#include "update.h"
#include <cstdlib>

Test(sketch_test_suite, testing_sketch_addition){
	srand (time(NULL));
	for (int i = 0; i < 1; i++){
		const unsigned long vect_size = 100;
		const unsigned long num_updates = 100;
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
=======

Test(misc, failing){
	cr_assert(0);
}

Test(misc, passing){
>>>>>>> d1ab12581f7e9708c5507070b367163e2d5765aa
	cr_assert(1);
}

#include <criterion/criterion.h>
#include "../include/sketch.h"
#include "util/testing_vector.h"
#include "../include/update.h"

Test(sketch_test_suite, GIVEN_only_index_zero_updated_THEN_it_works) {
  // GIVEN only the index 0 is updated
  srand(time(NULL));
  int vec_size = 1000;
  int num_updates = 100;
  int delta = 0, d;
  Sketch sketch = Sketch(vec_size, rand());
  for (int i=0;i<num_updates-1;++i) {
    d = rand()%10 - 5;
    sketch.update({0,d});
    delta+=d;
  }
  d = rand()%10 - 5;
  if (delta+d == 0) ++d;
  sketch.update({0,d});
  delta+=d;

  // THEN it works
  Update res = sketch.query();
  Update exp {0,delta};
  cr_assert(res == exp, "Res: {%d,%d}. Actual delta: %d",res.index, res.delta, delta);
  cout << "Passed: GIVEN_only_index_zero_updated_THEN_it_works" << endl;
}

Test(sketch_test_suite, WHEN_many_sketches_are_sampled_THEN_all_of_them_are_nonzero) {
  srand(time(NULL));
  int n = 100000;
  int vec_size = 1000;
  int num_updates = 100;
  int times = 2;

  while (times--) {
    bool failed_flag = 0;
    Testing_Vector test_vec = Testing_Vector(vec_size,num_updates);
    for (int i=0;i<n;++i) {
      Sketch sketch = Sketch(vec_size,rand());
      for (int j=0;j<num_updates;++j) {
        sketch.update(test_vec.get_update(j));
      }
      Update res = sketch.query();
      if (res.delta == 0 || test_vec.get_entry(res.index) != res.delta) {
        failed_flag = 1;
        break;
      }
    }
    if (!failed_flag) {
      cr_assert(true, "All sketches were nonzero");
      cout << "Passed: WHEN_many_sketches_are_sampled_THEN_all_of_them_are_nonzero" << endl;
      return;
    }
  }
  cr_assert(false, "Test failed twice in a row");
  // TODO: create probabilistic scoring system; if probability of the event we test
  // becomes below some threshhold, fail
}

Test(sketch_test_suite, testing_sketch_addition){
  srand (time(NULL));
  for (int i = 0; i < 1000; i++){
    const unsigned long vect_size = 1000;
    const unsigned long num_updates = 1000;
    Sketch sketch = Sketch(vect_size,rand());
    Testing_Vector test_vector = Testing_Vector(vect_size,num_updates);
    for (unsigned long j = 0; j < num_updates; j++){
      sketch.update(test_vector.get_update(j));
    }
    Update result = sketch.query();
    if (test_vector.get_entry(result.index) == 0 || test_vector.get_entry(result.index) != result.delta ){
      cr_assert(0, "Failed on test %d", i);
    }
  }
}

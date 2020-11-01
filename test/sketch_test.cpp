#include <criterion/criterion.h>
#include "../sketch.h"
#include "../testing_vector.h"
#include "../update.h"

Test(sketch_test_suite, IF_many_sketches_are_sampled_THEN_all_of_them_are_nonzero) {
  srand(time(NULL));
  int n = 1000000;
  int vec_size = 1000;
  int num_updates = 1000;
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
      return;
    }
  }
  cr_assert(false, "Test failed twice in a row");
  // TODO: create probabilistic scoring system; if probability of the event we test
  // becomes below some threshhold, fail
}

#include <gtest/gtest.h>
#include "../include/sketch.h"
#include "util/testing_vector.h"

TEST(SketchTestSuite, TestExceptions) {
  Sketch sketch1 = Sketch(10,rand());
  ASSERT_THROW(sketch1.query(), AllBucketsZeroException);
  ASSERT_THROW(sketch1.query(), MultipleQueryException);

  /*
   * Use deterministic sketch
   * This generates the following buckets:
   * 1111111111
   * 1001010011
   * 0100000100
   * 0000000000
   * 1111111111
   * 0010100111
   * 0111100000
   * 0000110101
   * 1111111111
   * 0101011010
   * 0000000010
   * 0000000000
   * 1111111111
   * 0000000010
   * 0000011000
   * 0000000000
   * 0100000000
   *
   * To find a set of indicies where no good buckets exist, repeatedly
   * eliminate the index of a good bucket until no good buckets exist.
   *
   * Once we have our indicies, try deltas until all buckets are not good.
   */
  Sketch sketch2 = Sketch(10, 1);
  sketch2.update({0, 1});
  sketch2.update({2, 1});
  sketch2.update({3, 1});
  sketch2.update({4, 1});
  sketch2.update({5, 1});
  sketch2.update({6, 1});
  sketch2.update({9, 1});
  ASSERT_THROW(sketch2.query(), NoGoodBucketException);
}

TEST(SketchTestSuite, GIVENonlyIndexZeroUpdatedTHENitWorks) {
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
  ASSERT_EQ(res, exp) << "Expected: " << exp << "\nActual: " << res;
}

TEST(SketchTestSuite, TestingSketchAddition){
  srand (time(NULL));
  for (int i = 0; i < 1000; i++){
    const unsigned long vect_size = 1000;
    const unsigned long num_updates = 1000;
    Sketch sketch = Sketch(vect_size,rand());
    Testing_Vector test_vector = Testing_Vector(vect_size,num_updates);
    for (unsigned long j = 0; j < num_updates; j++){
      sketch.update(test_vector.get_update(j));
    }
    Update result;
    try {
      result = sketch.query();
    } catch (exception& e) {
      ASSERT_FALSE(0) << "Failed on test " << i << "due to: " << e.what();
    }
    if (test_vector.get_entry(result.index) == 0 || test_vector.get_entry(result.index) != result.delta ){
      ASSERT_FALSE(0) << "Failed on test " << i;
    }
  }
}

TEST(SketchTestSuite, WHENmanySketchesAreSampledTHENallOfThemAreNonzero) {
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
      return;
    }
    std::cout << "Failed once" << std::endl;
  }
  ASSERT_FALSE(0) << "Test failed twice in a row";
  // TODO: create probabilistic scoring system; if probability of the event we test
  // becomes below some threshhold, fail
}

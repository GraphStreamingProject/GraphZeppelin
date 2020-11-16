#include <gtest/gtest.h>
#include "../include/sketch.h"
#include "util/testing_vector.h"
#include "../include/update.h"

TEST(SketchTestSuite, TestExceptions) {
  Sketch sketch = Sketch(10,rand());
  ASSERT_THROW(sketch.query(), NoGoodBucketException);
  ASSERT_THROW(sketch.query(), MultipleQueryException);
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
  ASSERT_EQ(res, exp) << "Expected: " << exp << std::endl << "Actual: " << res;
}

/**
 * Makes sure sketch addition works
 */
/*
TEST(SketchTestSuite, TestingSketchAddition){
  srand (time(NULL));
  for (int i = 0; i < 10000; i++){
    const unsigned long vect_size = 1000;
    const unsigned long num_updates = 100;
    const long seed = rand();
    Sketch sketch1 = Sketch(vect_size, seed);
    Sketch sketch2 = Sketch(vect_size, seed);
    Testing_Vector test_vector1 = Testing_Vector(vect_size, num_updates);
    Testing_Vector test_vector2 = Testing_Vector(vect_size, num_updates);
    for (unsigned long j = 0; j < num_updates; j++){
      sketch1.update(test_vector1.get_update(j));
      sketch2.update(test_vector2.get_update(j));
    }
    Sketch sketchsum = sketch1 + sketch2;
    Update result;
    try {
      result = sketchsum.query();
    } catch (exception& e) {
      ASSERT_FALSE(0) << "Failed on test " << i << "due to: " << e.what();
    }
    ASSERT_EQ(test_vector1.get_entry(result.index) + test_vector2.get_entry(result.index), result.delta)
      << "Failed on test " << i;
  }
}
*/

TEST(SketchTestSuite, test_sketch_fail_probability) {
  srand (time(NULL));
  const unsigned long num_sketches = 1000;
  const double max_fail_probability = .01;
  const unsigned long vec_size = 1000;
  const unsigned long num_updates = 100;
  
  unsigned long all_bucket_failures = 0;
  unsigned long sample_incorrect_failures = 0;
  for (unsigned long i = 0; i < num_sketches; i++) {
    Testing_Vector test_vec = Testing_Vector(vec_size, num_updates);
    Sketch sketch = Sketch(vec_size, rand());
    for (unsigned long j = 0; j < num_updates; j++){
      sketch.update(test_vec.get_update(j));
    }
    try {
      Update res = sketch.query();
      ASSERT_NE(res.delta, 0) << "Sample is zero";
      if (res.index < vec_size) {
        if (res.delta != test_vec.get_entry(res.index)) {
          EXPECT_EQ(res.delta, test_vec.get_entry(res.index))
            << "Sampled index " << res.index << ": got " << res.delta
            << ", expected " << test_vec.get_entry(res.index);
        }
      } else {
        EXPECT_LT(res.index, vec_size) << " Sampled index out of bounds";
        //Sampled index out of bounds
      }
    } catch (NoGoodBucketException& e) {
      //No good bucket
      EXPECT_TRUE(0) << e.what();
      all_bucket_failures++;
    } catch (MultipleQueryException& e) {
      ASSERT_TRUE(0) << e.what();
    }
  }
  EXPECT_LT(all_bucket_failures, max_fail_probability * num_sketches)
    << "All buckets failed " << all_bucket_failures << '/' << num_sketches
    << " times (expected less than " << max_fail_probability << ')';
}


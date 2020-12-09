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
  int num_updates = 1000;
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
 * Make sure sketch sampling works
 */
void test_sketch_sample(unsigned long num_sketches,
    unsigned long vec_size, unsigned long num_updates,
    double max_sample_fail_prob, double max_bucket_fail_prob) {
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
      //Multiple queries shouldn't happen, but if we do get here fail test
      ASSERT_NE(res.delta, 0) << "Sample is zero";
      ASSERT_LT(res.index, vec_size) << "Sampled index out of bounds";
      if (res.delta != test_vec.get_entry(res.index)) {
        //Undetected sample error
        sample_incorrect_failures++;
      }
    } catch (AllBucketsZeroException& e) {
      //All buckets being 0 implies that the whole vector should be 0
      bool vec_zero = true;
      for (unsigned long j = 0; vec_zero && j < vec_size; j++) {
        if (test_vec.get_entry(j) != 0) {
          vec_zero = false;
        }
      }
      if (!vec_zero) {
        sample_incorrect_failures++;
      }
    } catch (NoGoodBucketException& e) {
      //No good bucket
      all_bucket_failures++;
    } catch (MultipleQueryException& e) {
      //Multiple queries shouldn't happen, but if we do get here fail test
      FAIL() << e.what();
    }
  }
  EXPECT_LE(sample_incorrect_failures, max_sample_fail_prob * num_sketches)
    << "Sample incorrect " << sample_incorrect_failures << '/' << num_sketches
    << " times (expected less than " << max_sample_fail_prob << ')';
  EXPECT_LE(all_bucket_failures, max_bucket_fail_prob * num_sketches)
    << "All buckets failed " << all_bucket_failures << '/' << num_sketches
    << " times (expected less than " << max_bucket_fail_prob << ')';
}

TEST(SketchTestSuite, DISABLED_TestSketchSample) {
  srand (time(NULL));
  test_sketch_sample(10000, 100, 100, 0.005, 0.005);
  test_sketch_sample(1000, 1000, 1000, 0.001, 0.001);
  test_sketch_sample(1000, 10000, 10000, 0.001, 0.001);
}

/**
 * Make sure sketch addition works
 */
void test_sketch_addition(unsigned long num_sketches,
    unsigned long vec_size, unsigned long num_updates,
    double max_sample_fail_prob, double max_bucket_fail_prob) {
  unsigned long all_bucket_failures = 0;
  unsigned long sample_incorrect_failures = 0;
  for (int i = 0; i < num_sketches; i++){
    const long seed = rand();
    Sketch sketch1 = Sketch(vec_size, seed);
    Sketch sketch2 = Sketch(vec_size, seed);
    Testing_Vector test_vec1 = Testing_Vector(vec_size, num_updates);
    Testing_Vector test_vec2 = Testing_Vector(vec_size, num_updates);

    for (unsigned long j = 0; j < num_updates; j++){
      sketch1.update(test_vec1.get_update(j));
      sketch2.update(test_vec2.get_update(j));
    }
    Sketch sketchsum = sketch1 + sketch2;
    try {
      Update res = sketchsum.query();
      ASSERT_NE(res.delta, 0) << "Sample is zero";
      ASSERT_LT(res.index, vec_size) << "Sampled index out of bounds";
      if (res.delta != test_vec1.get_entry(res.index) + test_vec2.get_entry(res.index)) {
        sample_incorrect_failures++;
      }
    } catch (AllBucketsZeroException& e) {
      //All buckets being 0 implies that the whole vector should be 0
      bool vec_zero = true;
      for (unsigned long j = 0; vec_zero && j < vec_size; j++) {
        if (test_vec1.get_entry(j) + test_vec2.get_entry(j) != 0) {
          vec_zero = false;
        }
      }
      if (!vec_zero) {
        sample_incorrect_failures++;
      }
    } catch (NoGoodBucketException& e) {
      //No good bucket
      all_bucket_failures++;
    } catch (MultipleQueryException& e) {
      //Multiple queries shouldn't happen, but if we do get here fail test
      FAIL() << e.what();
    }
  }
  EXPECT_LE(sample_incorrect_failures, max_sample_fail_prob * num_sketches)
    << "Sample incorrect " << sample_incorrect_failures << '/' << num_sketches
    << " times (expected less than " << max_sample_fail_prob << ')';
  EXPECT_LE(all_bucket_failures, max_bucket_fail_prob * num_sketches)
    << "All buckets failed " << all_bucket_failures << '/' << num_sketches
    << " times (expected less than " << max_bucket_fail_prob << ')';
}

TEST(SketchTestSuite, DISABLED_TestSketchAddition){
  srand (time(NULL));
  test_sketch_addition(10000, 100, 100, 0.005, 0.005);
  test_sketch_addition(1000, 1000, 1000, 0.001, 0.001);
  test_sketch_addition(1000, 10000, 10000, 0.001, 0.001);
}
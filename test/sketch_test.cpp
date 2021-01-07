#include <gtest/gtest.h>
#ifdef USE_NATIVE_F
#include "../include/sketch_native.h"
#else
#include "../include/sketch.h"
#endif
#include "util/testing_vector.h"

TEST(SketchTestSuite, TestExceptions) {
  Sketch sketch1 = Sketch(10,rand());
  ASSERT_THROW(sketch1.query(), AllBucketsZeroException);
  ASSERT_THROW(sketch1.query(), MultipleQueryException);

  /**
   * Find a vector that makes no good buckets
   */
  Sketch sketch2 = Sketch(100, 0);
  std::vector<bool> vec_idx(sketch2.n, true);
  unsigned long long num_buckets = bucket_gen(sketch2.n);
  unsigned long long num_guesses = guess_gen(sketch2.n);
  for (unsigned long long i = 0; i < num_buckets; ++i) {
    for (unsigned long long j = 0; j < num_guesses;) {
      long bucket_id = i * num_guesses + j;
      XXH64_hash_t bucket_seed = XXH64(&bucket_id, 8, sketch2.seed);
      uint64_t index = 0;
      for (uint64_t k = 0; k < sketch2.n; ++k) {
        if (vec_idx[k] && sketch2.buckets[bucket_id].contains(k + 1, bucket_seed, 1 << j)) {
          if (index == 0) {
            index = k + 1;
          } else {
            index = 0;
            break;
          }
        }
      }
      if (index) {
        vec_idx[index - 1] = false;
        i = j = 0;
      } else {
        ++j;
      }
    }
  }
  for (uint64_t i = 0; i < sketch2.n; ++i) {
    if (vec_idx[i]) {
      sketch2.update({i, 1});
    }
  }
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

TEST(SketchTestSuite, TestSketchSample) {
  srand (time(NULL));
//  test_sketch_sample(10000, 100, 100, 0.005, 0.005);
  test_sketch_sample(1000, 1000, 1000, 0.001, 0.001);
//  test_sketch_sample(1000, 10000, 10000, 0.001, 0.001);
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

TEST(SketchTestSuite, TestSketchAddition){
  srand (time(NULL));
  test_sketch_addition(10000, 100, 100, 0.005, 0.005);
  test_sketch_addition(1000, 1000, 1000, 0.001, 0.001);
  test_sketch_addition(1000, 10000, 10000, 0.001, 0.001);
}

/**
 * Large sketch test
 */
void test_sketch_large(unsigned long vec_size, unsigned long num_updates) {
  srand(time(NULL));
  Sketch sketch = Sketch(vec_size, rand());
  //Keep seed for replaying update stream later
  unsigned long seed = rand();
  srand(seed);
  for (unsigned long j = 0; j < num_updates; j++){
    sketch.update({(long) rand() % vec_size, rand() % 10 - 5});
  }
  try {
    Update res = sketch.query();
    //Multiple queries shouldn't happen, but if we do get here fail test
    ASSERT_NE(res.delta, 0) << "Sample is zero";
    ASSERT_LT(res.index, vec_size) << "Sampled index out of bounds";
    //Replay update stream, keep track of the sampled index
    srand(seed);
    long actual_delta = 0;
    for (unsigned long j = 0; j < num_updates; j++){
      Update update = {(long) rand() % vec_size, rand() % 10 - 5};
      if (update.index == res.index) {
        actual_delta += update.delta;
      }
    }
    //Undetected sample error, not likely to happen for large vectors
    ASSERT_EQ(res.delta, actual_delta);
  } catch (AllBucketsZeroException& e) {
    //All buckets being 0 implies that the whole vector should be 0, not likely to happen for large vectors
    FAIL() << "AllBucketsZeroException:" << e.what();
  } catch (NoGoodBucketException& e) {
    //No good bucket, not likely to happen for large vectors
    FAIL() << "NoGoodBucketException:" << e.what();
  } catch (MultipleQueryException& e) {
    //Multiple queries shouldn't happen, but if we do get here fail test
    FAIL() << "MultipleQueryException:" << e.what();
  }
}

TEST(SketchTestSuite, TestSketchLarge) {
  test_sketch_large(10000000, 1000000);
}

#include "sketch.h"
#include "bucket.h"
#include <chrono>
#include <gtest/gtest.h>
#include "testing_vector.h"

static const int num_columns = 7;
TEST(SketchTestSuite, TestSampleResults) {
  Sketch sketch1(10, rand(), 1, num_columns);
  ASSERT_EQ(sketch1.sample().second, ZERO);
  sketch1.update(1);
  ASSERT_THROW(sketch1.sample(), OutOfQueriesException);

  /**
   * Find a vector that makes no good buckets
   */
  Sketch sketch2(100, 0, 1, num_columns);
  std::vector<bool> vec_idx(100 * 100, true);
  size_t columns = num_columns;
  size_t guesses = Sketch::calc_bkt_per_col(100);
  size_t total_updates = 2;
  for (size_t i = 0; i < columns;) {
    size_t depth_1_updates = 0;
    size_t k = 0;
    size_t u = 0;
    for (; u < total_updates || depth_1_updates < 2;) {
      if (vec_idx[k] == false) {
        ++k;
        continue;
      }

      col_hash_t depth = Bucket_Boruvka::get_index_depth(k, sketch2.column_seed(i), guesses);
      if (depth >= 2) {
        vec_idx[k] = false; // force all updates to only touch depths <= 1
        i = 0;
        break;
      }
      else if (depth == 1) {
        ++depth_1_updates;
      }
      ++u;
      ++k;
    }
    if (u > total_updates) {
      total_updates = u;
      i = 0;
    }
    else if (u == total_updates) ++i;
  }
  size_t applied_updates = 0;
  for (uint64_t i = 0; i < vec_idx.size(); ++i) {
    if (vec_idx[i]) {
      sketch2.update(static_cast<vec_t>(i));
      if (++applied_updates >= total_updates) break;
    }
  }
  ASSERT_EQ(sketch2.sample().second, FAIL);
}

TEST(SketchTestSuite, GIVENonlyIndexZeroUpdatedTHENitWorks) {
  // GIVEN only the index 0 is updated
  Sketch sketch(40, rand(), 1, num_columns);
  sketch.update(0);
  sketch.update(0);
  sketch.update(0);

  // THEN it works
  std::pair<vec_t, SampleSketchRet> query_ret = sketch.sample();
  vec_t res = query_ret.first;
  SampleSketchRet ret_code = query_ret.second;

  ASSERT_EQ(res, 0) << "Expected: 0" << std::endl << "Actual: " << res;
  ASSERT_EQ(ret_code, GOOD);
}

/**
 * Make sure sketch sampling works
 */
void test_sketch_sample(unsigned long num_sketches,
    unsigned long vec_size, unsigned long num_updates,
    double max_sample_fail_prob, double max_bucket_fail_prob) {
  std::chrono::duration<long double> runtime(0);
  unsigned long all_bucket_failures = 0;
  unsigned long sample_incorrect_failures = 0;
  for (unsigned long i = 0; i < num_sketches; i++) {
    Testing_Vector test_vec = Testing_Vector(vec_size, num_updates);
    Sketch sketch(vec_size, rand(), 1, num_columns);
    auto start_time = std::chrono::steady_clock::now();
    for (unsigned long j = 0; j < num_updates; j++){
      sketch.update(test_vec.get_update(j));
    }
    runtime += std::chrono::steady_clock::now() - start_time;
    try {
      std::pair<vec_t, SampleSketchRet> query_ret = sketch.sample();
      vec_t res_idx = query_ret.first;
      SampleSketchRet ret_code = query_ret.second;

      if (ret_code == GOOD) {
        //Multiple queries shouldn't happen, but if we do get here fail test
        ASSERT_LT(res_idx, vec_size) << "Sampled index out of bounds";
        if (!test_vec.get_entry(res_idx)) {
          //Undetected sample error
          sample_incorrect_failures++;
        }
      }
      else if (ret_code == ZERO) {
        //All buckets being 0 implies that the whole vector should be 0
        bool vec_zero = true;
        for (unsigned long j = 0; vec_zero && j < vec_size; j++) {
          if (test_vec.get_entry(j)) {
            vec_zero = false;
          }
        }
        if (!vec_zero) {
          sample_incorrect_failures++;
        }
      }
      else { // sketch failed
        //No good bucket
        all_bucket_failures++;
      }
    } catch (OutOfQueriesException& e) {
      //Multiple queries shouldn't happen, but if we do get here fail test
      FAIL() << e.what();
    }
  }
  std::cout << "Updating " << num_sketches << " sketches of length "
    << vec_size << " vectors with " << num_updates << " updates took "
    << runtime.count() << std::endl;
  EXPECT_LE(sample_incorrect_failures, max_sample_fail_prob * num_sketches)
    << "Sample incorrect " << sample_incorrect_failures << '/' << num_sketches
    << " times (expected less than " << max_sample_fail_prob << ')';
  EXPECT_LE(all_bucket_failures, max_bucket_fail_prob * num_sketches)
    << "All buckets failed " << all_bucket_failures << '/' << num_sketches
    << " times (expected less than " << max_bucket_fail_prob << ')';
}

TEST(SketchTestSuite, TestSketchSample) {
  test_sketch_sample(10000, 1e2, 100, 0.001, 0.03);
  test_sketch_sample(1000, 1e3, 1000, 0.001, 0.03);
  test_sketch_sample(1000, 1e4, 10000, 0.001, 0.03);
}

/**
 * Make sure sketch merging works
 */
void test_sketch_merge(unsigned long num_sketches,
    unsigned long vec_size, unsigned long num_updates,
    double max_sample_fail_prob, double max_bucket_fail_prob) {

  unsigned long all_bucket_failures = 0;
  unsigned long sample_incorrect_failures = 0;
  for (unsigned long i = 0; i < num_sketches; i++){
    const long seed = rand();
    Sketch sketch1(vec_size, seed, 1, num_columns);
    Sketch sketch2(vec_size, seed, 1, num_columns);
    Testing_Vector test_vec1 = Testing_Vector(vec_size, num_updates);
    Testing_Vector test_vec2 = Testing_Vector(vec_size, num_updates);

    for (unsigned long j = 0; j < num_updates; j++){
      sketch1.update(test_vec1.get_update(j));
      sketch2.update(test_vec2.get_update(j));
    }
    sketch1.merge(sketch2);
    try {
      std::pair<vec_t, SampleSketchRet> query_ret = sketch1.sample();
      vec_t res_idx = query_ret.first;
      SampleSketchRet ret_code = query_ret.second;

      if (ret_code == GOOD) {
        ASSERT_LT(res_idx, vec_size) << "Sampled index out of bounds";
        if (test_vec1.get_entry(res_idx) == test_vec2.get_entry(res_idx)) {
          sample_incorrect_failures++;
        }
      }
      else if (ret_code == ZERO) {
        //All buckets being 0 implies that the whole vector should be 0
        bool vec_zero = true;
        for (unsigned long j = 0; vec_zero && j < vec_size; j++) {
          if (test_vec1.get_entry(j) != test_vec2.get_entry(j)) {
            vec_zero = false;
          }
        }
        if (!vec_zero) {
          sample_incorrect_failures++;
        }
      }
      else { // sketch failed
        //No good bucket
        all_bucket_failures++;
      }
    } catch (OutOfQueriesException& e) {
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

TEST(SketchTestSuite, TestSketchMerge) {
  test_sketch_merge(10000, 1e2, 100, 0.001, 0.03);
  test_sketch_merge(1000, 1e3, 1000, 0.001, 0.03);
  test_sketch_merge(1000, 1e4, 10000, 0.001, 0.03);
}

TEST(SketchTestSuite, TestSketchRangeMerge) {
  Sketch skt1(2048, rand(), 10, 3);
  Sketch skt2(2048, rand(), 10, 3);

  skt1.sample();
  skt1.range_merge(skt2, 1, 1);

  skt1.sample();
  skt1.range_merge(skt2, 2, 1);

  skt1.sample();
  skt1.range_merge(skt2, 3, 1);

  skt1.sample();
  skt1.range_merge(skt2, 4, 1);
}

/**
 * Large sketch test
 */
void test_sketch_large(unsigned long vec_size, unsigned long num_updates) {

  // we use an optimization in our sketching that is valid when solving CC
  // we assume that max number of non-zeroes in vector is vec_size / 4
  // therefore we need to ensure that in this test that we don't do more than that
  num_updates = std::min(num_updates, vec_size / 4);

  Sketch sketch(vec_size, rand(), 1, 2 * log2(vec_size));
  //Keep seed for replaying update stream later
  unsigned long seed = rand();
  srand(seed);
  auto start_time = std::chrono::steady_clock::now();
  for (unsigned long j = 0; j < num_updates; j++){
    sketch.update(static_cast<vec_t>(rand() % vec_size));
  }
  std::cout << "Updating vector of size " << vec_size << " with " << num_updates
    << " updates took " << std::chrono::duration<long double>(
      std::chrono::steady_clock::now() - start_time).count() << std::endl;
  try {
    std::pair<vec_t, SampleSketchRet> query_ret = sketch.sample();
    vec_t res_idx = query_ret.first;
    SampleSketchRet ret_code = query_ret.second;

    if (ret_code == GOOD) {
      //Multiple queries shouldn't happen, but if we do get here fail test
      ASSERT_LT(res_idx, vec_size) << "Sampled index out of bounds";
      //Replay update stream, keep track of the sampled index
      srand(seed);
      bool actual_delta = false;
      for (unsigned long j = 0; j < num_updates; j++){
        vec_t update_idx = static_cast<vec_t>(rand() % vec_size);
        if (update_idx == res_idx) {
          actual_delta = !actual_delta;
        }
      }
      //Undetected sample error, not likely to happen for large vectors
      ASSERT_EQ(actual_delta, true);
    }
    else if (ret_code == ZERO) {
      //All buckets being 0 implies that the whole vector should be 0, not likely to happen for large vectors
      FAIL() << "Got a Zero Sketch!";
    }
    else { // sketch failed
      // No good bucket, not likely to happen for large vectors
      FAIL() << "Sketch Failed: No good bucket.";
    }
  } catch (OutOfQueriesException& e) {
    //Multiple queries shouldn't happen, but if we do get here fail test
    FAIL() << "OutOfQueriesException:" << e.what();
  }
}

TEST(SketchTestSuite, TestSketchLarge) {
  constexpr uint64_t upper_bound = 1e9;
  for (uint64_t i = 1e4; i <= upper_bound; i *= 10) {
    test_sketch_large(i, 1000000);
  }
}

TEST(SketchTestSuite, TestSerialization) {
  unsigned long vec_size = 1 << 10;
  unsigned long num_updates = 10000;
  Testing_Vector test_vec = Testing_Vector(vec_size, num_updates);
  auto seed = rand();
  Sketch sketch(vec_size, seed, 1, num_columns);
  for (unsigned long j = 0; j < num_updates; j++){
    sketch.update(test_vec.get_update(j));
  }
  auto file = std::fstream("./out_sketch.txt", std::ios::out | std::ios::binary | std::ios::trunc);
  sketch.serialize(file);
  file.close();

  auto in_file = std::fstream("./out_sketch.txt", std::ios::in | std::ios::binary);
  Sketch reheated(vec_size, seed, in_file, 1, num_columns);

  ASSERT_EQ(sketch, reheated);
}
/*
TEST(SketchTestSuite, TestSparseSerialization) {
  unsigned long vec_size = 1 << 10;
  unsigned long num_updates = 10000;
  Testing_Vector test_vec = Testing_Vector(vec_size, num_updates);
  auto seed = rand();
  Sketch sketch(vec_size, seed, 1, num_columns);
  for (unsigned long j = 0; j < num_updates; j++){
    sketch.update(test_vec.get_update(j));
  }
  auto file = std::fstream("./out_sketch.txt", std::ios::out | std::ios::binary | std::ios::trunc);
  sketch.serialize(file, SPARSE);
  file.close();

  auto in_file = std::fstream("./out_sketch.txt", std::ios::in | std::ios::binary);
  Sketch reheated(vec_size, seed, in_file, SPARSE, 1, num_columns);

  ASSERT_EQ(sketch, reheated);
}

TEST(SketchTestSuite, TestExhaustiveQuery) {
  size_t runs = 10;
  size_t vec_size = 20;
  for (size_t i = 0; i < runs; i++) {
    Sketch sketch(vec_size, rand(), 1, log2(vec_size));

    sketch.update(1);
    sketch.update(2);
    sketch.update(3);
    sketch.update(4);
    sketch.update(5);
    sketch.update(6);
    sketch.update(7);
    sketch.update(8);
    sketch.update(9);
    sketch.update(10);

    std::pair<std::unordered_set<vec_t>, SampleSketchRet> query_ret = sketch.exhaustive_sample();
    if (query_ret.second != GOOD) {
      ASSERT_EQ(query_ret.first.size(), 0) << query_ret.second;
    }

    // assert everything returned is valid and <= 10 things
    ASSERT_LE(query_ret.first.size(), 10);
    for (vec_t non_zero : query_ret.first) {
      ASSERT_GT(non_zero, 0);
      ASSERT_LE(non_zero, 10);
    }

    // assert everything returned is unique
    std::set<vec_t> unique_elms(query_ret.first.begin(), query_ret.first.end());
    ASSERT_EQ(unique_elms.size(), query_ret.first.size());
  }
}
*/
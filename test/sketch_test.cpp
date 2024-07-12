#include "sketch.h"
#include "bucket.h"
#include <chrono>
#include <gtest/gtest.h>
#include <random>
#include "testing_vector.h"

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

static const int num_columns = 7;
TEST(SketchTestSuite, TestSampleResults) {
  Sketch sketch1(10, get_seed(), 1, num_columns);
  ASSERT_EQ(sketch1.sample().result, ZERO);
  sketch1.update(1);
  ASSERT_THROW(sketch1.sample(), OutOfSamplesException);

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
  ASSERT_EQ(sketch2.sample().result, FAIL);
}

TEST(SketchTestSuite, GIVENonlyIndexZeroUpdatedTHENitWorks) {
  // GIVEN only the index 0 is updated
  Sketch sketch(40, get_seed(), 1, num_columns);
  sketch.update(0);
  sketch.update(0);
  sketch.update(0);

  // THEN it works
  SketchSample query_ret = sketch.sample();
  vec_t res = query_ret.idx;
  SampleResult ret_code = query_ret.result;

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
    Sketch sketch(vec_size, get_seed() + i * 7, 1, num_columns);
    auto start_time = std::chrono::steady_clock::now();
    for (unsigned long j = 0; j < num_updates; j++){
      sketch.update(test_vec.get_update(j));
    }
    runtime += std::chrono::steady_clock::now() - start_time;
    try {
      SketchSample query_ret = sketch.sample();
      vec_t res_idx = query_ret.idx;
      SampleResult ret_code = query_ret.result;

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
    } catch (OutOfSamplesException& e) {
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
    const long seed = get_seed() + 7 * i;
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
      SketchSample query_ret = sketch1.sample();
      vec_t res_idx = query_ret.idx;
      SampleResult ret_code = query_ret.result;

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
    } catch (OutOfSamplesException& e) {
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
  Sketch skt1(2048, get_seed(), 10, 3);
  Sketch skt2(2048, get_seed(), 10, 3);

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

  // Keep seed for replaying update stream later
  unsigned long seed = get_seed();
  Sketch sketch(vec_size, seed, 1, 2 * log2(vec_size));

  std::mt19937_64 gen(seed);
  auto start_time = std::chrono::steady_clock::now();
  for (unsigned long j = 0; j < num_updates; j++){
    sketch.update(static_cast<vec_t>(gen() % vec_size));
  }
  std::cout << "Updating vector of size " << vec_size << " with " << num_updates
    << " updates took " << std::chrono::duration<long double>(
      std::chrono::steady_clock::now() - start_time).count() << std::endl;
  try {
    SketchSample query_ret = sketch.sample();
    vec_t res_idx = query_ret.idx;
    SampleResult ret_code = query_ret.result;

    if (ret_code == GOOD) {
      //Multiple queries shouldn't happen, but if we do get here fail test
      ASSERT_LT(res_idx, vec_size) << "Sampled index out of bounds";
      //Replay update stream, keep track of the sampled index
      gen = std::mt19937_64(seed);
      bool actual_delta = false;
      for (unsigned long j = 0; j < num_updates; j++){
        vec_t update_idx = static_cast<vec_t>(gen() % vec_size);
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
  } catch (OutOfSamplesException& e) {
    //Multiple queries shouldn't happen, but if we do get here fail test
    FAIL() << "OutOfSamplesException:" << e.what();
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
  auto seed = get_seed();
  Sketch sketch(vec_size, seed, 3, num_columns);
  for (unsigned long j = 0; j < num_updates; j++){
    sketch.update(test_vec.get_update(j));
  }
  auto file = std::fstream("./out_sketch.txt", std::ios::out | std::ios::binary | std::ios::trunc);
  sketch.serialize(file);
  file.close();

  auto in_file = std::fstream("./out_sketch.txt", std::ios::in | std::ios::binary);
  Sketch reheated(vec_size, seed, in_file, 3, num_columns);

  ASSERT_EQ(sketch, reheated);
}

TEST(SketchTestSuite, TestSamplesHaveUniqueSeed) {
  size_t num_samples = 50;
  size_t cols_per_sample = 3;
  Sketch sketch(10000, 1024, num_samples, cols_per_sample);
  std::set<size_t> seeds;

  for (size_t i = 0; i < num_samples * cols_per_sample; i++) {
    size_t seed = sketch.column_seed(i);
    ASSERT_EQ(seeds.count(seed), 0);
    seeds.insert(seed);
  }
}

TEST(SketchTestSuite, TestExhaustiveQuery) {
  size_t runs = 10;
  size_t vec_size = 2000;
  for (size_t i = 0; i < runs; i++) {
    Sketch sketch(vec_size, get_seed() + 7 * i, 1, log2(vec_size));

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

    ExhaustiveSketchSample query_ret = sketch.exhaustive_sample();
    if (query_ret.result != GOOD) {
      ASSERT_EQ(query_ret.idxs.size(), 0) << query_ret.result;
    }

    // assert everything returned is valid and <= 10 things
    ASSERT_LE(query_ret.idxs.size(), 10);
    for (vec_t non_zero : query_ret.idxs) {
      ASSERT_GT(non_zero, 0);
      ASSERT_LE(non_zero, 10);
    }

    // assert everything returned is unique
    std::set<vec_t> unique_elms(query_ret.idxs.begin(), query_ret.idxs.end());
    ASSERT_EQ(unique_elms.size(), query_ret.idxs.size());
  }
}

TEST(SketchTestSuite, TestSampleInsertGrinder) {
  size_t nodes = 4096;
  Sketch sketch(Sketch::calc_vector_length(nodes), get_seed(), Sketch::calc_cc_samples(nodes, 1));

  for (size_t src = 0; src < nodes - 1; src++) {
    for (size_t dst = src + 7; dst < nodes; dst += 7) {
      sketch.update(concat_pairing_fn(src, dst));
    }
  }

  size_t successes = 0;
  for (size_t i = 0; i < Sketch::calc_cc_samples(nodes, 1); i++) {
    SketchSample ret = sketch.sample();
    if (ret.result == FAIL) continue;

    ++successes;
    ASSERT_EQ(ret.result, GOOD); // ZERO is not allowed

    // validate the result we got
    Edge e = inv_concat_pairing_fn(ret.idx);
    ASSERT_EQ((e.dst - e.src) % 7, 0);
  }
  ASSERT_GE(successes, 2);
}

TEST(SketchTestSuite, TestSampleDeleteGrinder) {
  size_t nodes = 4096;
  Sketch sketch(Sketch::calc_vector_length(nodes), get_seed(), Sketch::calc_cc_samples(nodes, 1));

  // insert
  for (size_t src = 0; src < nodes - 1; src++) {
    for (size_t dst = src + 7; dst < nodes; dst += 7) {
      sketch.update(concat_pairing_fn(src, dst));
    }
  }

  // delete all odd src
  for (size_t src = 1; src < nodes - 1; src += 2) {
    for (size_t dst = src + 7; dst < nodes; dst += 7) {
      sketch.update(concat_pairing_fn(src, dst));
    }
  }

  size_t successes = 0;
  for (size_t i = 0; i < Sketch::calc_cc_samples(nodes, 1); i++) {
    SketchSample ret = sketch.sample();
    if (ret.result == FAIL) continue;

    ++successes;
    ASSERT_EQ(ret.result, GOOD); // ZERO is not allowed

    // validate the result we got
    Edge e = inv_concat_pairing_fn(ret.idx);
    ASSERT_EQ((e.dst - e.src) % 7, 0);
    ASSERT_EQ(e.src % 2, 0);
  }
  ASSERT_GE(successes, 2);
}

TEST(SketchTestSuite, TestRawBucketUpdate) {
  size_t successes = 0;
  for (size_t t = 0; t < 20; t++) {
    size_t seed = get_seed() + 5 * t;
    Sketch sk1(4096, seed, 1, 1);
    Sketch sk2(4096, seed, 1, 1);

    for (size_t i = 0; i < 1024; i++) {
      sk1.update(i);
    }

    const Bucket *data = sk1.get_readonly_bucket_ptr();

    sk2.merge_raw_bucket_buffer(data);

    SketchSample sample = sk2.sample();

    ASSERT_NE(sample.result, ZERO);
    if (sample.result == GOOD) {
      ++successes;
      ASSERT_GE(sample.idx, 0);
      ASSERT_LT(sample.idx, 1024);
    }

    Bucket *copy_data = new Bucket[sk1.get_buckets()];
    memcpy(copy_data, data, sk1.bucket_array_bytes());
    sk2.merge_raw_bucket_buffer(copy_data);

    sk2.reset_sample_state();
    sample = sk2.sample();
    ASSERT_EQ(sample.result, ZERO);

    delete[] copy_data;
  }
  ASSERT_GT(successes, 0);
}

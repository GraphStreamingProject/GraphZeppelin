#include <vector>
#include <graph.h>
#include <binary_graph_stream.h>
#include <gtest/gtest.h>
#include "../src/cuda_kernel.cu"
#include "../include/l0_sampling/sketch.h"
#include "../include/test/testing_vector.h"
#include "../include/test/sketch_constructors.h"

// Deprecated, needs to be updated 

static const int fail_factor = 100;

bool contains_inclusive(col_hash_t hash, col_hash_t guess) {
  for (col_hash_t i = 1; i <= guess; i<<=1) {
    if (!Bucket_Boruvka::contains(hash, i)) {
      return false;
    }
  }
  return true;
}

TEST(CUDASketchTestSuite, TestExceptions) {
  Sketch::configure(100, fail_factor);
  SketchUniquePtr sketch1 = makeSketch(rand());
  ASSERT_EQ(sketch1->query().second, ZERO);
  ASSERT_THROW(sketch1->query(), MultipleQueryException);

  /**
   * Find a vector that makes no good buckets
   */
  Sketch::configure(10000, fail_factor);
  SketchUniquePtr sketch2 = makeSketch(0);
  std::vector<bool> vec_idx(sketch2->n, true);
  unsigned long long num_buckets = bucket_gen(fail_factor);
  unsigned long long num_guesses = guess_gen(sketch2->n);
  for (unsigned long long i = 0; i < num_buckets; ++i) {
    for (unsigned long long j = 0; j < num_guesses;) {
      uint64_t index = 0;
      for (uint64_t k = 0; k < sketch2->n; ++k) {
        if (vec_idx[k] && contains_inclusive(Bucket_Boruvka::col_index_hash(k, sketch2->seed + i), 1 << j)) {
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
  vec_t *update_indexes; 
  cudaMallocManaged(&update_indexes, sketch2->n * sizeof(vec_t));
  int num_updates = 0;

  for (uint64_t i = 0; i < sketch2->n; ++i) {
    if (vec_idx[i]) {
      update_indexes[i] = static_cast<vec_t>(i);
      num_updates++;
    }
  }
  int num_threads = 1024;
	int num_blocks = (num_updates + num_threads - 1) / num_threads;

  CudaSketch* cudaSketches;
  int num_sketches = 1;
  cudaMallocManaged(&cudaSketches, num_sketches * sizeof(CudaSketch));

  CudaSketch cudaSketch(sketch2->get_bucket_a(), sketch2->get_bucket_c(), sketch2->get_failure_factor(), sketch2->get_num_elems(), sketch2->get_num_buckets(), sketch2->get_num_guesses(), sketch2->get_seed()); 
  cudaSketches[0] = cudaSketch;
	
  sketchUpdate(num_threads, num_blocks, num_updates, update_indexes, cudaSketches);

  ASSERT_EQ(sketch2->query().second, FAIL);
}

TEST(CUDASketchTestSuite, GIVENonlyIndexZeroUpdatedTHENitWorks) {
  // GIVEN only the index 0 is updated
  srand(time(nullptr));
  Sketch::configure(1000, fail_factor);
  SketchUniquePtr sketch = makeSketch(rand());

  vec_t *update_indexes; 
  int num_updates = 3;
  cudaMallocManaged(&update_indexes, num_updates * sizeof(vec_t));

  update_indexes[0] = 0;
  update_indexes[1] = 0;
  update_indexes[2] = 0;

  int num_threads = 1024;
	int num_blocks = (num_updates + num_threads - 1) / num_threads;

  CudaSketch* cudaSketches;
  int num_sketches = 1;
  cudaMallocManaged(&cudaSketches, num_sketches * sizeof(CudaSketch));

  CudaSketch cudaSketch(sketch->get_bucket_a(), sketch->get_bucket_c(), sketch->get_failure_factor(), sketch->get_num_elems(), sketch->get_num_buckets(), sketch->get_num_guesses(), sketch->get_seed()); 
  cudaSketches[0] = cudaSketch;
	
  sketchUpdate(num_threads, num_blocks, num_updates, update_indexes, cudaSketches);

  /*sketch->update(0);
  sketch->update(0);
  sketch->update(0);*/

  // THEN it works
  std::pair<vec_t, SampleSketchRet> query_ret = sketch->query();
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
  Sketch::configure(vec_size, fail_factor);

  srand(time(nullptr));
  std::chrono::duration<long double> runtime(0);
  unsigned long all_bucket_failures = 0;
  unsigned long sample_incorrect_failures = 0;
  for (unsigned long i = 0; i < num_sketches; i++) {
    Testing_Vector test_vec = Testing_Vector(vec_size, num_updates);
    SketchUniquePtr sketch = makeSketch(rand());
    auto start_time = std::chrono::steady_clock::now();

    /*for (unsigned long j = 0; j < num_updates; j++){
      sketch->update(test_vec.get_update(j));
    }*/

    vec_t *update_indexes; 
    cudaMallocManaged(&update_indexes, num_updates * sizeof(vec_t));
    for (unsigned long j = 0; j < num_updates; j++){
      update_indexes[j] = test_vec.get_update(j);
    }
    int num_threads = 1024;
    int num_blocks = (num_updates + num_threads - 1) / num_threads;

    CudaSketch* cudaSketches;
    int num_sketches = 1;
    cudaMallocManaged(&cudaSketches, num_sketches * sizeof(CudaSketch));

    CudaSketch cudaSketch(sketch->get_bucket_a(), sketch->get_bucket_c(), sketch->get_failure_factor(), sketch->get_num_elems(), sketch->get_num_buckets(), sketch->get_num_guesses(), sketch->get_seed()); 
    cudaSketches[0] = cudaSketch;
    
    sketchUpdate(num_threads, num_blocks, num_updates, update_indexes, cudaSketches);
    runtime += std::chrono::steady_clock::now() - start_time;
    try {
      std::pair<vec_t, SampleSketchRet> query_ret = sketch->query();
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
    } catch (MultipleQueryException& e) {
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

TEST(CUDASketchTestSuite, TestSketchSample) {
  srand (time(nullptr));
  test_sketch_sample(10000, 1e3, 100, 0.005, 0.02);
  test_sketch_sample(1000, 1e4, 1000, 0.001, 0.02);
  test_sketch_sample(1000, 1e5, 10000, 0.001, 0.02);
}

/**
 * Make sure sketch addition works
 */
void test_sketch_addition(unsigned long num_sketches,
    unsigned long vec_size, unsigned long num_updates,
    double max_sample_fail_prob, double max_bucket_fail_prob) {
  Sketch::configure(vec_size, fail_factor);

  srand (time(NULL));
  unsigned long all_bucket_failures = 0;
  unsigned long sample_incorrect_failures = 0;
  for (unsigned long i = 0; i < num_sketches; i++){
    const long seed = rand();
    SketchUniquePtr sketch1 = makeSketch(seed);
    SketchUniquePtr sketch2 = makeSketch(seed);
    Testing_Vector test_vec1 = Testing_Vector(vec_size, num_updates);
    Testing_Vector test_vec2 = Testing_Vector(vec_size, num_updates);

    /*for (unsigned long j = 0; j < num_updates; j++){
      sketch1->update(test_vec1.get_update(j));
      sketch2->update(test_vec2.get_update(j));
    }*/

    vec_t *update_indexes1; 
    vec_t *update_indexes2;

    cudaMallocManaged(&update_indexes1, num_updates * sizeof(vec_t));
    cudaMallocManaged(&update_indexes2, num_updates * sizeof(vec_t));

    for (unsigned long j = 0; j < num_updates; j++){
      update_indexes1[j] = test_vec1.get_update(j);
      update_indexes2[j] = test_vec2.get_update(j);
    }
    int num_threads = 1024;
    int num_blocks = (num_updates + num_threads - 1) / num_threads;

    CudaSketch* cudaSketches1, *cudaSketches2;
    int num_sketches = 1;
    cudaMallocManaged(&cudaSketches1, num_sketches * sizeof(CudaSketch));
    cudaMallocManaged(&cudaSketches2, num_sketches * sizeof(CudaSketch));

    CudaSketch cudaSketch1(sketch1->get_bucket_a(), sketch1->get_bucket_c(), sketch1->get_failure_factor(), sketch1->get_num_elems(), sketch1->get_num_buckets(), sketch1->get_num_guesses(), sketch1->get_seed()); 
    cudaSketches1[0] = cudaSketch1;

    CudaSketch cudaSketch2(sketch2->get_bucket_a(), sketch2->get_bucket_c(), sketch2->get_failure_factor(), sketch2->get_num_elems(), sketch2->get_num_buckets(), sketch2->get_num_guesses(), sketch2->get_seed()); 
    cudaSketches2[0] = cudaSketch2;
    
    sketchUpdate(num_threads, num_blocks, num_updates, update_indexes1, cudaSketches1);
    sketchUpdate(num_threads, num_blocks, num_updates, update_indexes2, cudaSketches2);

    *sketch1 += *sketch2;
    try {
      std::pair<vec_t, SampleSketchRet> query_ret = sketch1->query();
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

TEST(CUDASketchTestSuite, TestSketchAddition){
  test_sketch_addition(10000, 1e3, 100, 0.005, 0.02);
  test_sketch_addition(1000, 1e4, 1000, 0.001, 0.02);
  test_sketch_addition(1000, 1e5, 10000, 0.001, 0.02);
}

/**
 * Large sketch test
 */
void test_sketch_large(unsigned long vec_size, unsigned long num_updates) {
  Sketch::configure(vec_size, fail_factor);

  // TEMPORARY FIX
  // we use an optimization in our sketching that is valid when solving CC
  // we assume that max number of non-zeroes in vector is vec_size / 4
  // therefore we need to ensure that in this test that we don't do more than that
  num_updates = std::min(num_updates, vec_size / 4);

  srand(time(nullptr));
  SketchUniquePtr sketch = makeSketch(rand());
  //Keep seed for replaying update stream later
  unsigned long seed = rand();
  srand(seed);
  auto start_time = std::chrono::steady_clock::now();

  /*for (unsigned long j = 0; j < num_updates; j++){
    sketch->update(static_cast<vec_t>(rand() % vec_size));
  }*/

  vec_t *update_indexes; 
  cudaMallocManaged(&update_indexes, num_updates * sizeof(vec_t));
  for (unsigned long j = 0; j < num_updates; j++){
    update_indexes[j] = static_cast<vec_t>(rand() % vec_size);
  }
  int num_threads = 1024;
  int num_blocks = (num_updates + num_threads - 1) / num_threads;

  CudaSketch* cudaSketches;
  int num_sketches = 1;
  cudaMallocManaged(&cudaSketches, num_sketches * sizeof(CudaSketch));

  CudaSketch cudaSketch(sketch->get_bucket_a(), sketch->get_bucket_c(), sketch->get_failure_factor(), sketch->get_num_elems(), sketch->get_num_buckets(), sketch->get_num_guesses(), sketch->get_seed()); 
  cudaSketches[0] = cudaSketch;
  
  sketchUpdate(num_threads, num_blocks, num_updates, update_indexes, cudaSketches);

  std::cout << "Updating vector of size " << vec_size << " with " << num_updates
    << " updates took " << std::chrono::duration<long double>(
      std::chrono::steady_clock::now() - start_time).count() << std::endl;
  try {
    std::pair<vec_t, SampleSketchRet> query_ret = sketch->query();
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
  } catch (MultipleQueryException& e) {
    //Multiple queries shouldn't happen, but if we do get here fail test
    FAIL() << "MultipleQueryException:" << e.what();
  }
}

TEST(CUDASketchTestSuite, TestSketchLarge) {
  constexpr uint64_t upper_bound = 1e18;
  for (uint64_t i = 1e6; i <= upper_bound; i *= 10) {
    test_sketch_large(i, 1000000);
  }
}

TEST(CUDASketchTestSuite, TestBatchUpdate) {
  unsigned long vec_size = 1000000000, num_updates = 10000;
  Sketch::configure(vec_size, fail_factor);
  srand(time(nullptr));
  std::vector<vec_t> updates(num_updates);
  for (unsigned long i = 0; i < num_updates; i++) {
    updates[i] = static_cast<vec_t>(rand() % vec_size);
  }
  auto sketch_seed = rand();
  SketchUniquePtr sketch = makeSketch(sketch_seed);
  SketchUniquePtr sketch_batch = makeSketch(sketch_seed);
  auto start_time = std::chrono::steady_clock::now();
  /*for (const vec_t& update : updates) {
    sketch->update(update);
  }*/

  vec_t *update_indexes; 
  cudaMallocManaged(&update_indexes, num_updates * sizeof(vec_t));
  for (unsigned long j = 0; j < num_updates; j++){
    update_indexes[j] = updates[j];
  }
  int num_threads = 1024;
  int num_blocks = (num_updates + num_threads - 1) / num_threads;

  CudaSketch* cudaSketches;
  int num_sketches = 1;
  cudaMallocManaged(&cudaSketches, num_sketches * sizeof(CudaSketch));

  CudaSketch cudaSketch(sketch->get_bucket_a(), sketch->get_bucket_c(), sketch->get_failure_factor(), sketch->get_num_elems(), sketch->get_num_buckets(), sketch->get_num_guesses(), sketch->get_seed()); 
  cudaSketches[0] = cudaSketch;
  
  sketchUpdate(num_threads, num_blocks, num_updates, update_indexes, cudaSketches);

  std::cout << "One by one updates took " << static_cast<std::chrono::duration<long double>>(std::chrono::steady_clock::now() - start_time).count() << std::endl;
  start_time = std::chrono::steady_clock::now();
  sketch_batch->batch_update(updates);
  std::cout << "Batched updates took " << static_cast<std::chrono::duration<long double>>(std::chrono::steady_clock::now() - start_time).count() << std::endl;

  ASSERT_EQ(*sketch, *sketch_batch);
}

TEST(CUDASketchTestSuite, TestSerialization) {
  printf("starting test!\n");
  unsigned long vec_size = 1 << 20;
  unsigned long num_updates = 10000;
  Sketch::configure(vec_size, fail_factor);
  Testing_Vector test_vec = Testing_Vector(vec_size, num_updates);
  auto seed = rand();
  SketchUniquePtr sketch = makeSketch(seed);

  /*for (unsigned long j = 0; j < num_updates; j++){
    sketch->update(test_vec.get_update(j));
  }*/

  vec_t *update_indexes; 
  cudaMallocManaged(&update_indexes, num_updates * sizeof(vec_t));
  for (unsigned long j = 0; j < num_updates; j++){
    update_indexes[j] = test_vec.get_update(j);
  }
  int num_threads = 1024;
  int num_blocks = (num_updates + num_threads - 1) / num_threads;

  CudaSketch* cudaSketches;
  int num_sketches = 1;
  cudaMallocManaged(&cudaSketches, num_sketches * sizeof(CudaSketch));

  CudaSketch cudaSketch(sketch->get_bucket_a(), sketch->get_bucket_c(), sketch->get_failure_factor(), sketch->get_num_elems(), sketch->get_num_buckets(), sketch->get_num_guesses(), sketch->get_seed()); 
  cudaSketches[0] = cudaSketch;

  sketchUpdate(num_threads, num_blocks, num_updates, update_indexes, cudaSketches);

  auto file = std::fstream("./out_sketch.txt", std::ios::out | std::ios::binary);
  sketch->write_binary(file);
  file.close();

  auto in_file = std::fstream("./out_sketch.txt", std::ios::in | std::ios::binary);
  SketchUniquePtr reheated = makeSketch(seed, in_file);

  ASSERT_EQ(*sketch, *reheated);
}


int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#include "../../include/l0_sampling/cudaSketch.cuh"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <iterator>

// __global__ means this is called from the CPU, and runs on the GPU
__global__ void sketch_query(const vec_t* bucket_a, const vec_hash_t* bucket_c, vec_t* result, size_t num_guesses, uint64_t seed) {
    int bucketId = blockIdx.y * blockDim.y + threadIdx.y;
    int guessId = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned bucket_id = bucketId * num_guesses + guessId;

    if (bucket_is_good(bucket_a[bucket_id], bucket_c[bucket_id], bucketId, 1 << guessId, seed)) {
      result[0] = bucket_a[bucket_id];
    }
}

__host__ __device__  col_hash_t bucket_col_index_hash(const vec_t& update_idx, const long seed_and_col) {
  return col_hash(&update_idx, sizeof(update_idx), seed_and_col);
}

__host__ __device__  vec_hash_t bucket_index_hash(const vec_t& index, long sketch_seed) {
  return vec_hash(&index, sizeof(index), sketch_seed);
}

__host__ __device__  bool bucket_contains(const col_hash_t& col_index_hash, const col_hash_t& guess_nonzero) {
  return (col_index_hash & guess_nonzero) == 0; // use guess_nonzero (power of 2) to check ith bit
}

__host__ __device__ bool bucket_is_good(const vec_t& a, const vec_hash_t& c, const unsigned bucket_col, const vec_t& guess_nonzero, const long& sketch_seed) {
  return c == bucket_index_hash(a, sketch_seed)
    && bucket_contains(bucket_col_index_hash(a, sketch_seed + bucket_col), guess_nonzero);
}

CudaSketch::CudaSketch(size_t numElements, size_t numBuckets, size_t numGuesses, vec_t* bucketA, vec_hash_t* bucketC, uint64_t currentSeed) {
  num_elements = numElements;
  num_buckets = numBuckets;
  num_guesses = numGuesses;
  bucket_a = bucketA;
  bucket_c = bucketC;
  seed = currentSeed;
  result[0] = 0;
}

void CudaSketch::query() {
  vec_t bucket_a_bytes = sizeof(vec_t) * num_elements;
  vec_hash_t bucket_c_bytes = sizeof(vec_hash_t) * num_elements;
  vec_t result_bytes = sizeof(vec_t);

  // Vectors for holding the host-side (CPU-side) data
  std::vector<vec_t> bucket_a_data;
  bucket_a_data.reserve(bucket_a_bytes);
  std::vector<vec_hash_t> bucket_c_data;
  bucket_c_data.reserve(bucket_c_bytes);

  // Initialize each vector
  for (int i = 0; i < num_elements; i++) {
    bucket_a_data.push_back(bucket_a[i]);
    bucket_c_data.push_back(bucket_c[i]);
  }

  // Allocate memory on the device
  vec_t *d_bucket_a;
  vec_hash_t *d_bucket_c;
  vec_t *d_result;
  cudaMalloc(&d_bucket_a, bucket_a_bytes);
  cudaMalloc(&d_bucket_c, bucket_c_bytes);
  cudaMalloc(&d_result, result_bytes);

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpy(d_bucket_a, bucket_a_data.data(), bucket_a_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bucket_c, bucket_c_data.data(), bucket_c_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, result.data(), result_bytes, cudaMemcpyHostToDevice);

  // Threads per CTA
  int THREADS = 128;

  // Blocks per grid dimension
  int BLOCKS = num_elements / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  sketch_query<<<blocks, threads>>>(d_bucket_a, d_bucket_c, d_result, num_guesses, seed);

  // Copy back to the host
  cudaMemcpy(result.data(), d_result, result_bytes, cudaMemcpyDeviceToHost);

  // Free memory on device
  cudaFree(d_bucket_a);
  cudaFree(d_bucket_c);
  cudaFree(d_result);
}
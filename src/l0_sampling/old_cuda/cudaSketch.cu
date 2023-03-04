#include "../../include/l0_sampling/cudaSketch.cuh"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <iterator>

__host__ __device__  bool bucket_contains(const col_hash_t& col_index_hash, const col_hash_t& guess_nonzero) {
  return (col_index_hash & guess_nonzero) == 0; // use guess_nonzero (power of 2) to check ith bit
}

__global__ void sketch_update(vec_t* bucket_a, vec_hash_t* bucket_c, const size_t num_elems, const size_t num_buckets, const size_t num_guesses, const uint64_t seed, 
                              col_hash_t* col_index_hash, const vec_t update_idx, const vec_hash_t update_hash, vec_t* bucket_debug) {

    /*int currentId = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(currentId <= num_elems) {
      // For Debugging:
      bucket_debug[currentId] = bucket_debug[currentId] + 1;

      bucket_a[num_elems - 1] = bucket_a[num_elems - 1] ^ update_idx;
      bucket_c[num_elems - 1] = bucket_c[num_elems - 1] ^ update_hash;

      if(currentId == num_elems - 1){
        return;
      }

      int guessId = currentId % num_guesses;
      int colId = currentId / num_guesses;
      if (bucket_contains(col_index_hash[colId], ((col_hash_t)1) << guessId)){
        bucket_a[currentId] = bucket_a[currentId] ^ update_idx;
        bucket_c[currentId] = bucket_c[currentId] ^ update_hash;
      }
    }
    else {
      return;
    }*/

    int currentId = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if(currentId < num_buckets) {
      bucket_a[num_elems - 1] = bucket_a[num_elems - 1] ^ update_idx;
      bucket_c[num_elems - 1] = bucket_c[num_elems - 1] ^ update_hash;
      for (int guessId = 0; guessId < num_guesses; guessId++) {
        unsigned bucket_id = currentId * num_guesses + guessId;
        if (bucket_contains(col_index_hash[currentId], ((col_hash_t)1) << guessId)){
          bucket_a[bucket_id] = bucket_a[bucket_id] ^ update_idx;
          bucket_c[bucket_id] = bucket_c[bucket_id] ^ update_hash;
        }
        else {
          return;
        }
      }
    }
}

CudaSketch::CudaSketch(size_t numElems, size_t numBuckets, size_t numGuesses, vec_t* &bucketA, vec_hash_t* &bucketC, uint64_t currentSeed) {
  num_elems = numElems;
  num_buckets = numBuckets;
  num_guesses = numGuesses;
  bucket_a = bucketA;
  bucket_c = bucketC;
  seed = currentSeed;
};

void CudaSketch::update(col_hash_t* d_col_index_hash, const vec_t& update_idx, vec_t* &bucket_debug) {

  vec_hash_t update_hash = Bucket_Boruvka::index_hash(update_idx, seed);

  col_hash_t *col_index_hash = new col_hash_t[num_buckets];
  for (unsigned i = 0; i < num_buckets; ++i) {
    col_index_hash[i] = Bucket_Boruvka::col_index_hash(update_idx, seed + i);
  }

  cudaMemcpy(d_col_index_hash, col_index_hash, num_buckets * sizeof(col_hash_t), cudaMemcpyHostToDevice);

  // Threads per CTA (1024)
  int num_threads = 1 << 10;

  // Blocks per grid dimension[i] = 0;
  int num_blocks = (num_buckets + num_threads - 1) / num_threads;

  sketch_update<<<num_blocks, num_threads>>>(bucket_a, bucket_c, num_elems, num_buckets, num_guesses, seed, d_col_index_hash, update_idx, update_hash, bucket_debug);
}
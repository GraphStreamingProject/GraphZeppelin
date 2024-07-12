#include "bucket.h"
#include <cuda_xxhash64.cuh>
#include "../src/cuda_library.cu"
#include "util.h"

#include <iostream>
#include <chrono>

typedef unsigned long long int uint64_cu;
typedef uint64_cu vec_t_cu;

__device__ int ctzll(col_hash_t v) {
  uint64_t c;
  if (v) {
    v = (v ^ (v - 1)) >> 1;
    for (c = 0; v; c++) {
      v >>= 1;
    }
  }
  else {
    c = 8 * sizeof(v);
  }
  return c;
}

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

__device__ vec_hash_t bucket_get_index_hash(const vec_t update_idx, const long sketch_seed) {
  return CUDA_XXH64(&update_idx, sizeof(vec_t), sketch_seed);
}

__device__ col_hash_t bucket_get_index_depth(const vec_t_cu update_idx, const long seed_and_col, const vec_hash_t max_depth) {
  col_hash_t depth_hash = CUDA_XXH64(&update_idx, sizeof(vec_t), seed_and_col);
  depth_hash |= (1ull << max_depth); // assert not > max_depth by ORing

  //return ctzll(depth_hash);
  return __ffsll(depth_hash);
}

__global__ void gpuKernel(vec_t* updates, vec_t* gpu_hash_values, int N, uint64_t seed, size_t num_columns, size_t bkt_per_col) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(tid < N) {
        int column_id = tid / num_columns;
        gpu_hash_values[tid] = bucket_get_index_depth(updates[tid], seed + (column_id * 5), bkt_per_col);
    }
}

int main(int argc, char **argv) {
    vec_t* updates;

    uint64_t seed = get_seed();
    std::cout << "Seed: " << seed << "\n";

    size_t num_columns = 1000;
    size_t bkt_per_col = 100;
    size_t N = num_columns * bkt_per_col;

    // CPU 
    vec_t* cpu_hash_values = new vec_t[N];

    // GPU
    vec_t* gpu_hash_values;

    cudaMallocManaged(&updates, N * sizeof(vec_t));
    cudaMallocManaged(&gpu_hash_values, N * sizeof(vec_t));

    // Initialization
    for (node_id_t i = 0; i < N; i++) {
        updates[i] = static_cast<vec_t>(concat_pairing_fn(i, i));
        cpu_hash_values[i] = 0;
        gpu_hash_values[i] = 0;
    }

    // Run CPU version
    for (node_id_t i = 0; i < N; i++) {
        //cpu_hash_values[i] = Bucket_Boruvka::get_index_hash(updates[i], seed);
        int column_id = i / num_columns;
        cpu_hash_values[i] = Bucket_Boruvka::get_index_depth(updates[i], seed + (column_id * 5), bkt_per_col);
    }

    int num_threads = 1024;
    int num_blocks = (N + num_threads - 1) / num_threads;

    // Run GPU Kernel
    gpuKernel<<<num_blocks,num_threads>>>(updates, gpu_hash_values, N, seed, num_columns, bkt_per_col);
    cudaDeviceSynchronize();

    // Validate values
    for (node_id_t i = 0; i < N; i++) {
        if(cpu_hash_values[i] != gpu_hash_values[i]) {
            std::cout << "Wrong values at i = " << i << " CPU: " << cpu_hash_values[i] << " GPU: " << gpu_hash_values[i] << "\n";
        }
        //std::cout << " i = " << i << " CPU: " << cpu_hash_values[i] << " GPU: " << gpu_hash_values[i] << "\n";
    }

    cudaFree(updates);
    delete[] cpu_hash_values;
    cudaFree(gpu_hash_values);

    return 0;
}
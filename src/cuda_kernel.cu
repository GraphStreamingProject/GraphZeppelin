
#include <vector>
#include <cuda_xxhash64.cuh>
//#include <sketch.h>
#include "../include/cuda_kernel.cuh"

typedef unsigned long long int uint64_cu;
typedef uint64_cu vec_t_cu;

/*
*   
*   Bucket Functions
*
*/

// Source: http://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightLinear
__device__ int ctzll(col_hash_t v) {
  int c;
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

__device__ col_hash_t bucket_get_index_depth(const vec_t_cu update_idx, const long seed_and_col, const vec_hash_t max_depth) {
  // Update CUDA_XXH, confirm they are correct with xxhash_test.cu
  col_hash_t depth_hash = CUDA_XXH64(&update_idx, sizeof(vec_t), seed_and_col);
  depth_hash |= (1ull << max_depth); // assert not > max_depth by ORing

  //return ctzll(depth_hash);
  return __ffsll(depth_hash);
}

__device__ vec_hash_t bucket_get_index_hash(const vec_t update_idx, const long sketch_seed) {
  return CUDA_XXH64(&update_idx, sizeof(vec_t), sketch_seed);
}

__device__ bool bucket_is_good(const vec_t a, const vec_hash_t c, const long sketch_seed) {
  return c == bucket_get_index_hash(a, sketch_seed);
}

__device__ void bucket_update(vec_t_cu& a, vec_hash_t& c, const vec_t_cu& update_idx, const vec_hash_t& update_hash) {
  atomicXor(&a, update_idx);
  atomicXor((vec_t_cu*)&c, (vec_t_cu)update_hash);
}

/*
*   
*   Sketch's Update Functions
*
*/

__global__ void sketchUpdate_kernel(node_id_t src, node_id_t num_nodes, vec_t* edgeUpdates, vec_t update_start_id, size_t update_size,
    size_t num_buckets, size_t d_bucket_id, vec_t* d_bucket_a, vec_hash_t* d_bucket_c, size_t num_samples, size_t num_columns, size_t bkt_per_col, long sketchSeed) {

  extern __shared__ vec_t_cu sketches[];
  vec_t_cu* bucket_a = sketches;
  vec_hash_t* bucket_c = (vec_hash_t*)&bucket_a[num_buckets];

  // Each thread will initialize a bucket
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }

  __syncthreads();

  // Update sketch - each thread works on 1 update for on 1 column
  for (int id = threadIdx.x; id < update_size * num_columns; id += blockDim.x) {

    int column_id = id % num_columns;
    int update_id = id / num_columns;
    
    vec_hash_t checksum = bucket_get_index_hash(edgeUpdates[update_start_id + update_id], sketchSeed);
    
    if (column_id == 0) {
      // Update depth 0 bucket
      bucket_update(bucket_a[num_buckets - 1], bucket_c[num_buckets - 1], edgeUpdates[update_start_id + update_id], checksum);
    }

    // Update higher depth buckets
    col_hash_t depth = bucket_get_index_depth(edgeUpdates[update_start_id + update_id], sketchSeed + (column_id * 5), bkt_per_col);
    size_t bucket_id = column_id * bkt_per_col + depth;
    if(depth < bkt_per_col)
      bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[update_start_id + update_id], checksum);
  }

  // Each thread works on num_column updates for 1 column (Similar performance)
  /*for (int id = threadIdx.x; id < update_size * num_columns; id += blockDim.x) {

    int column_id = id % num_columns;
    int update_offset = (id / num_columns) * num_columns;

    for (int i = 0; i < num_columns; i++) {

      if ((update_offset) + i >= update_size) {
        break;
      }

      vec_hash_t checksum = bucket_get_index_hash(edgeUpdates[update_start_id + update_offset + i], sketchSeed);
    
      if (column_id == 0) {
        // Update depth 0 bucket
        bucket_update(bucket_a[num_buckets - 1], bucket_c[num_buckets - 1], edgeUpdates[update_start_id + update_offset + i], checksum);
      }

      // Update higher depth buckets
      col_hash_t depth = bucket_get_index_depth(edgeUpdates[update_start_id + update_offset + i], sketchSeed + (column_id * 5), bkt_per_col);
      size_t bucket_id = column_id * bkt_per_col + depth;
      if(depth < bkt_per_col)
        bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[update_start_id + update_offset + i], checksum);
    }
  }*/

  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 0; i < num_buckets; i++) {
      d_bucket_a[d_bucket_id + i] = bucket_a[i];
      d_bucket_c[d_bucket_id + i] = bucket_c[i];
    }
  }
}

/*__global__ void k_gtsStream_kernel(node_id_t src, vec_t* edgeUpdates, vec_t prev_offset, size_t update_size, node_id_t num_nodes,
    int num_sketches, size_t num_elems, size_t num_columns, size_t num_guesses, int k, size_t cuda_bucket_id, vec_t* cuda_bucket_a, vec_hash_t* cuda_bucket_c, long* sketchSeeds) {
      
  extern __shared__ vec_t_cu sketches[];
  vec_t_cu* bucket_a = sketches;
  vec_hash_t* bucket_c = (vec_hash_t*)&bucket_a[num_elems * num_sketches];

  for (int k_id = 0; k_id < k; k_id++) {
    __syncthreads();
    // Each thread will initialize
    for (int i = threadIdx.x; i < num_sketches * num_elems; i += blockDim.x) {
      bucket_a[i] = 0;
      bucket_c[i] = 0;
    }

    __syncthreads();

    for (int id = threadIdx.x; id < update_size + num_sketches; id += blockDim.x) {
    
    int sketch_offset = id % num_sketches;
    int update_offset = ((id / num_sketches) * num_sketches);
    
      for (int i = 0; i < num_sketches; i++) {

        if ((prev_offset + update_offset + i) >= prev_offset + update_size) {
          break;
        }

        vec_hash_t checksum = bucket_get_index_hash(edgeUpdates[prev_offset + update_offset + i], sketchSeeds[(src * num_sketches * k) + (k_id * num_sketches) + sketch_offset]);

        // Update depth 0 bucket
        bucket_update(bucket_a[(sketch_offset * num_elems) + num_elems - 1], bucket_c[(sketch_offset * num_elems) + num_elems - 1], edgeUpdates[prev_offset + update_offset + i], checksum);

        // Update higher depth buckets
        for (unsigned j = 0; j < num_columns; ++j) {
          col_hash_t depth = bucket_get_index_depth(edgeUpdates[prev_offset + update_offset + i], sketchSeeds[(src * num_sketches * k) + (k_id * num_sketches) + sketch_offset] + j*5, num_guesses);
          size_t bucket_id = j * num_guesses + depth;
          if(depth < num_guesses)
            bucket_update(bucket_a[(sketch_offset * num_elems) + bucket_id], bucket_c[(sketch_offset * num_elems) + bucket_id], edgeUpdates[prev_offset + update_offset + i], checksum);
        }
      }
    }

    __syncthreads();

    // Write back to global memory
    if (threadIdx.x == 0) {
      int k_bucket_id = cuda_bucket_id + (k_id * num_sketches * num_elems);
      for (int i = 0; i < num_sketches * num_elems; i++) {
        cuda_bucket_a[k_bucket_id + i] = bucket_a[i];
        cuda_bucket_c[k_bucket_id + i] = bucket_c[i];
      }
    }
  }

}*/

// Function that calls sketch update kernel code.
void CudaKernel::sketchUpdate(int num_threads, int num_blocks, node_id_t src, cudaStream_t stream, vec_t update_start_id, size_t update_size, vec_t d_bucket_id, CudaUpdateParams* cudaUpdateParams, long sketchSeed) {
  // Unwarp variables from cudaUpdateParams
  vec_t *edgeUpdates = cudaUpdateParams[0].d_edgeUpdates;

  node_id_t num_nodes = cudaUpdateParams[0].num_nodes;
  
  size_t num_buckets = cudaUpdateParams[0].num_buckets;

  size_t num_samples = cudaUpdateParams[0].num_samples;
  size_t num_columns = cudaUpdateParams[0].num_columns;
  size_t bkt_per_col = cudaUpdateParams[0].bkt_per_col;

  int maxbytes = num_buckets * sizeof(vec_t_cu) + num_buckets * sizeof(vec_hash_t);
  
  sketchUpdate_kernel<<<num_blocks, num_threads, maxbytes, stream>>>(src, num_nodes, edgeUpdates, update_start_id, update_size, num_buckets, d_bucket_id, cudaUpdateParams[0].d_bucket_a, cudaUpdateParams[0].d_bucket_c, num_samples, num_columns, bkt_per_col, sketchSeed);
}

// Function that calls sketch update kernel code. (K-Connectivity Version)
/*void CudaKernel::k_gtsStreamUpdate(int num_threads, int num_blocks, int graph_id, int k, vec_t bucket_id, node_id_t src, cudaStream_t stream, vec_t prev_offset, size_t update_size, CudaUpdateParams* cudaUpdateParams, long* sketchSeeds) {
  // Unwarp variables from cudaUpdateParams
  vec_t *edgeUpdates = cudaUpdateParams[0].d_edgeUpdates;

  node_id_t num_nodes = cudaUpdateParams[0].num_nodes;
  
  int num_sketches = cudaUpdateParams[0].num_sketches;

  size_t num_elems = cudaUpdateParams[0].num_elems;
  size_t num_columns = cudaUpdateParams[0].num_columns;
  size_t num_guesses = cudaUpdateParams[0].num_guesses;

  if ((num_nodes == 0 || num_sketches == 0) || (num_elems == 0 || num_columns == 0) || num_guesses == 0) {
    std::cout << "graph_id: " << graph_id << "\n";
    std::cout << "  num_nodes: " << num_nodes << "\n";
    std::cout << "  num_sketches: " << num_sketches << "\n";
    std::cout << "  num_elems: " << num_elems << "\n";
    std::cout << "  num_columns: " << num_columns << "\n";
    std::cout << "  num_guesses: " << num_guesses << "\n";
  }

  int maxbytes = num_elems * num_sketches * sizeof(vec_t_cu) + num_elems * num_sketches * sizeof(vec_hash_t);

  k_gtsStream_kernel<<<num_blocks, num_threads, maxbytes, stream>>>(src, edgeUpdates, prev_offset, update_size, num_nodes, num_sketches, num_elems, num_columns, num_guesses, k, bucket_id, cudaUpdateParams[0].d_bucket_a, cudaUpdateParams[0].d_bucket_c, sketchSeeds);
}*/

void CudaKernel::updateSharedMemory(size_t maxBytes) {
  cudaFuncSetAttribute(sketchUpdate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxBytes);
  //cudaFuncSetAttribute(k_gtsStream_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxBytes);
}

#include <vector>
#include <cuda_xxhash64.cuh>
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

__device__ col_hash_t bucket_get_index_depth(const vec_t_cu update_idx, const long seed_and_col, const vec_hash_t max_depth) {
  col_hash_t depth_hash = CUDA_XXH64(&update_idx, sizeof(vec_t), seed_and_col);
  depth_hash |= (1ull << max_depth); // assert not > max_depth by ORing

  //return ctzll(depth_hash);
  return __ffsll(depth_hash) - 1;
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

__global__ void sketchUpdate_kernel(node_id_t num_nodes, int num_device_blocks, node_id_t src_vertex, vec_t* edgeUpdates, size_t update_size,
  size_t num_sketch_buckets, Bucket* buckets, int* num_tb_columns, size_t bkt_per_col, long sketchSeed) {

  size_t num_columns = num_tb_columns[blockIdx.x];
  size_t num_buckets = num_columns * bkt_per_col;
  size_t column_offset = 0;

  for (int i = 0; i < blockIdx.x; i++) {
    column_offset += num_tb_columns[i];
  }

  // Increment num_buckets for last thread block
  if (blockIdx.x == (num_device_blocks - 1)) {
    num_buckets++;
  }
  
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
    
    vec_hash_t checksum = bucket_get_index_hash(edgeUpdates[update_id], sketchSeed);
    
    if ((blockIdx.x == (num_device_blocks - 1)) && (column_id == 0)) {
      // Update depth 0 bucket
      bucket_update(bucket_a[num_buckets - 1], bucket_c[num_buckets - 1], edgeUpdates[update_id], checksum);
    }

    // Update higher depth buckets
    col_hash_t depth = bucket_get_index_depth(edgeUpdates[update_id], sketchSeed + ((column_offset + column_id) * 5), bkt_per_col);
    size_t bucket_id = column_id * bkt_per_col + depth;
    if(depth < bkt_per_col)
      bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[update_id], checksum);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    size_t bucket_offset = 0;

    for (int i = 0; i < blockIdx.x; i++) {
      bucket_offset += (num_tb_columns[i] * bkt_per_col);
    }
  
    for (int i = 0; i < num_buckets; i++) {
      atomicXor((vec_t_cu*)&buckets[(src_vertex * num_sketch_buckets) + bucket_offset + i].alpha, bucket_a[i]);
      atomicXor((vec_t_cu*)&buckets[(src_vertex * num_sketch_buckets) + bucket_offset + i].gamma, (vec_t_cu)bucket_c[i]);
    }
  }
  
}

// Function that calls sketch update kernel code.
void CudaKernel::sketchUpdate(int num_threads, int num_blocks, node_id_t src_vertex, cudaStream_t stream, vec_t *edgeUpdates, size_t update_size, CudaUpdateParams* cudaUpdateParams, long sketchSeed) {
  node_id_t num_nodes = cudaUpdateParams[0].num_nodes;
  
  size_t bkt_per_col = cudaUpdateParams[0].bkt_per_col;

  // Calculate the num_buckets assigned to the last thread block
  size_t num_last_tb_buckets = (cudaUpdateParams[0].num_tb_columns[num_blocks-1] * bkt_per_col) + 1;
  
  // Set maxBytes for GPU kernel's shared memory
  size_t maxBytes = (num_last_tb_buckets * sizeof(vec_t_cu)) + (num_last_tb_buckets * sizeof(vec_hash_t));

  if (num_nodes == 0 || bkt_per_col == 0 || num_last_tb_buckets == 0) {
    std::cout << "num_nodes: " << num_nodes << "\n";
    std::cout << "bkt_per_col: " << bkt_per_col << "\n";
    std::cout << "num_last_tb_buckets: " << num_last_tb_buckets << "\n";
  }
  
  sketchUpdate_kernel<<<num_blocks, num_threads, maxBytes, stream>>>(num_nodes, num_blocks, src_vertex, edgeUpdates, update_size, cudaUpdateParams[0].num_buckets, cudaUpdateParams[0].buckets, cudaUpdateParams[0].num_tb_columns, bkt_per_col, sketchSeed);
}

__global__ void single_sketchUpdate_kernel(node_id_t* update_src, vec_t* update_sizes, vec_t* update_start_indexes, vec_t* edgeUpdates, Bucket* buckets, size_t num_buckets, size_t num_columns, size_t bkt_per_col, size_t sketchSeed) {

  extern __shared__ vec_t_cu sketches[];
  vec_t_cu* bucket_a = sketches;
  vec_hash_t* bucket_c = (vec_hash_t*)&bucket_a[num_buckets];

  // Each thread will initialize a bucket in shared memory
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }

  __syncthreads();

  // Update sketch - each thread works on 1 update for on 1 column
  for (int id = threadIdx.x; id < update_sizes[blockIdx.x] * num_columns; id += blockDim.x) {

    int column_id = id % num_columns;
    int update_id = id / num_columns;

    vec_hash_t checksum = bucket_get_index_hash(edgeUpdates[update_start_indexes[blockIdx.x] + update_id], sketchSeed);
    
    if (column_id == 0) {
      // Update depth 0 bucket
      bucket_update(bucket_a[num_buckets - 1], bucket_c[num_buckets - 1], edgeUpdates[update_start_indexes[blockIdx.x] + update_id], checksum);
    }

    // Update higher depth buckets
    col_hash_t depth = bucket_get_index_depth(edgeUpdates[update_start_indexes[blockIdx.x] + update_id], sketchSeed + (column_id * 5), bkt_per_col);
    size_t bucket_id = column_id * bkt_per_col + depth;
    if(depth < bkt_per_col)
      bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[update_start_indexes[blockIdx.x] + update_id], checksum);
  }

  __syncthreads();

  // Have one thread write sketch back to global memory (DRAM)
  if (threadIdx.x == 0) {
    for (int i = 0; i < num_buckets; i++) {
      atomicXor((vec_t_cu*)&buckets[(update_src[blockIdx.x] * num_buckets) + i].alpha, bucket_a[i]);
      atomicXor((vec_t_cu*)&buckets[(update_src[blockIdx.x] * num_buckets) + i].gamma, (vec_t_cu)bucket_c[i]);
    }
  }

}

void CudaKernel::single_sketchUpdate(int num_threads, int num_blocks, vec_t* edgeUpdates, node_id_t* update_src, vec_t* update_sizes, vec_t* update_start_index, CudaUpdateParams* cudaUpdateParams, size_t sketchSeed) {
  size_t bkt_per_col = cudaUpdateParams[0].bkt_per_col;
  size_t num_columns = cudaUpdateParams[0].num_columns;
  size_t num_buckets = cudaUpdateParams[0].num_buckets;
  size_t maxBytes = num_buckets * sizeof(vec_t) + num_buckets * sizeof(vec_hash_t);

  single_sketchUpdate_kernel<<<num_blocks, num_threads, maxBytes>>>(update_src, update_sizes, update_start_index, edgeUpdates, cudaUpdateParams[0].buckets, num_buckets, num_columns, bkt_per_col, sketchSeed);
}

void CudaKernel::updateSharedMemory(size_t maxBytes) {
  cudaFuncSetAttribute(sketchUpdate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxBytes);
  cudaFuncSetAttribute(single_sketchUpdate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxBytes);
}
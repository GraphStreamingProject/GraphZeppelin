#include <vector>
#include <cuda_xxhash64.cuh>
#include <graph.h>

typedef unsigned long long int uint64_cu;
typedef uint64_cu vec_t_cu;

class CudaSketch {
  public:
    vec_t* d_bucket_a;
    vec_hash_t* d_bucket_c;

    vec_t failure_factor; 
    size_t num_elems;
    size_t num_buckets;
    size_t num_guesses;
    uint64_t seed;

    // Default Constructor of CudaSketch
    CudaSketch():d_bucket_a(nullptr), d_bucket_c(nullptr) {};

    CudaSketch(vec_t* d_bucket_a, vec_hash_t* d_bucket_c, vec_t failure_factor, size_t num_elems, size_t num_buckets, size_t num_guesses, uint64_t seed): 
      d_bucket_a(d_bucket_a), d_bucket_c(d_bucket_c), failure_factor(failure_factor), num_elems(num_elems), num_buckets(num_buckets), num_guesses(num_guesses), seed(seed) {};
};

__device__ col_hash_t bucket_col_index_hash(const vec_t_cu& update_idx, const long seed_and_col) {
  return CUDA_XXH64(&update_idx, sizeof(update_idx), seed_and_col);
}

__device__ vec_hash_t bucket_index_hash(const vec_t_cu& index, long sketch_seed) {
  return CUDA_XXH32(&index, sizeof(index), sketch_seed);
}

__device__  bool bucket_contains(const vec_t_cu& col_index_hash, const vec_t_cu& guess_nonzero) {
  return (col_index_hash & guess_nonzero) == 0; // use guess_nonzero (power of 2) to check ith bit
}

__device__ void bucket_update(vec_t_cu& a, vec_hash_t& c, const vec_t_cu& update_idx, const vec_hash_t& update_hash) {
  atomicXor(&a, update_idx);
  atomicXor(&c, update_hash);
}

__device__ void bucket_a_single_update(vec_t_cu& a, const vec_t_cu& update_idx) {
  atomicXor(&a, update_idx);
}

__device__ void bucket_c_single_update(vec_hash_t& c, const vec_hash_t& update_hash) {
  atomicXor(&c, update_hash);
}

// Kernel code for only sketch updates
// Old version, needs to be updated
__global__ void sketch_update(int num_updates, vec_t* update_indexes, CudaSketch* cudaSketches) {

  // Get thread id
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // One thread will be responsible for one update to sketch
  if(tid < num_updates) {
    // Step 1: Get cudaSketch
    CudaSketch curr_cudaSketch = cudaSketches[0];

    // Step 2: Get update_idx
    vec_t update_idx = update_indexes[tid];

    // Step 3: Get all the buckets from cudaSketch
    vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
    vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;
    size_t num_elems = curr_cudaSketch.num_elems;

    // Step 4: Get update_hash
    vec_hash_t update_hash = bucket_index_hash(update_idx, curr_cudaSketch.seed);

    bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], update_idx, update_hash);

    // Step 5: Update current sketch
    for (unsigned i = 0; i < curr_cudaSketch.num_buckets; ++i) {
      col_hash_t col_index_hash = bucket_col_index_hash(update_idx, curr_cudaSketch.seed + i);
      for (unsigned j = 0; j < curr_cudaSketch.num_guesses; ++j) {
        unsigned bucket_id = i * curr_cudaSketch.num_guesses + j;
        if (bucket_contains(col_index_hash, ((col_hash_t)1) << j)){
          bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], update_idx, update_hash);
        } else break;
      }
    }
  }
}

// Kernel code of handling all the stream updates
// Four threads will be responsible for one edge update -> two threads are modifying one node's sketches
// Split the workload for traversing through a node's sketch
/*__global__ void quadStream_update(int* nodeUpdates, int num_updates, int num_nodes, int num_sketches, vec_t* edgeUpdates, 
                                    CudaSketch* cudaSketches) {

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < num_updates * 4){

    // Step 1: Get node based on tid.
    const vec_t_cu node = nodeUpdates[tid / 2];

    // Step 2: Update node's sketches
    for (int i = 0; i < num_sketches; i++) {

      CudaSketch curr_cudaSketch = cudaSketches[(node * num_sketches) + i];
      size_t num_elems = curr_cudaSketch.num_elems;

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[tid / 2], curr_cudaSketch.seed);

      // Let only one thread to update
      if (tid % 2 == 0) {
        bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[tid / 2], update_hash);        
      }

      __syncthreads();

      // Update node's sketches, splitting workloads for two threads
      for (unsigned j = tid % 2; j < curr_cudaSketch.num_buckets; j += 2) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[tid / 2], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
          unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[tid / 2], update_hash);
          } else break;
        }
      }
    }
  }
}*/

// Kernel code of handling all the stream updates
// Four threads will be responsible for one edge update -> two threads are modifying one node's sketches
// One thread is reponsible for bucket_a, the other thread is responsble for bucket_c
__global__ void quadStream_update(int* nodeUpdates, int num_updates, int num_nodes, int num_sketches, vec_t* edgeUpdates, 
                                    CudaSketch* cudaSketches) {

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < num_updates * 4){

    if (tid < num_updates * 2) { // Current thread is reponsible for bucket_a
      // Step 1: Get node based on tid.
      const vec_t_cu node = nodeUpdates[tid];

      // Step 2: Update node's sketches
      for (int i = 0; i < num_sketches; i++) {

        CudaSketch curr_cudaSketch = cudaSketches[(node * num_sketches) + i];
        size_t num_elems = curr_cudaSketch.num_elems;

        // Get buckets based on current sketch and node id
        vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;

        bucket_a_single_update(bucket_a[num_elems - 1], edgeUpdates[tid]);        
        
        // Update node's sketches, splitting workloads for two threads
        for (unsigned j = 0; j < curr_cudaSketch.num_buckets; ++j) {
          col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[tid], curr_cudaSketch.seed + j);
          for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
            unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
            if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
              bucket_a_single_update(bucket_a[bucket_id], edgeUpdates[tid]);
            } else break;
          }
        }
      }
    }
    else { // Current thread is reponsible for bucket_c
      // Step 1: Get node based on tid.
      const vec_t_cu node = nodeUpdates[tid - (num_updates * 2)];

      // Step 2: Update node's sketches
      for (int i = 0; i < num_sketches; i++) {

        CudaSketch curr_cudaSketch = cudaSketches[(node * num_sketches) + i];
        size_t num_elems = curr_cudaSketch.num_elems;

        // Get buckets based on current sketch and node id
        vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

        vec_hash_t update_hash = bucket_index_hash(edgeUpdates[tid - (num_updates * 2)], curr_cudaSketch.seed);

        bucket_c_single_update(bucket_c[num_elems - 1], update_hash);   
        
        // Update node's sketches, splitting workloads for two threads
        for (unsigned j = 0; j < curr_cudaSketch.num_buckets; ++j) {
          col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[tid - (num_updates * 2)], curr_cudaSketch.seed + j);
          for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
            unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
            if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
              bucket_c_single_update(bucket_c[bucket_id], update_hash);
            } else break;
          }
        }
      }
    }
  }
}

// Kernel code of handling all the stream updates
// Two threads will be responsible for one edge update -> one thread is only modifying one node's sketches.
/*__global__ void doubleStream_update(int* nodeUpdates, int num_updates, int num_nodes, int num_sketches, vec_t* edgeUpdates, 
                                    CudaSketch* cudaSketches) {

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < num_updates * 2){

    // Step 1: Get node based on tid.
    const vec_t_cu node = nodeUpdates[tid];

    // Step 2: Update node's sketches
    for (int i = 0; i < num_sketches; i++) {

      CudaSketch curr_cudaSketch = cudaSketches[(node * num_sketches) + i];
      size_t num_elems = curr_cudaSketch.num_elems;

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[tid], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[tid], update_hash);

      // Update node's sketches
      for (unsigned j = 0; j < curr_cudaSketch.num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[tid], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
          unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[tid], update_hash);
          } else break;
        }
      }
    }
  }
}*/

// Kernel code of handling all the stream updates
// Two threads will be responsible for one edge update -> one thread is only modifying one node's sketches.
__global__ void doubleStream_update(int* nodeUpdates, int num_updates, int num_nodes, int num_sketches, vec_t* edgeUpdates, 
                                    CudaSketch* cudaSketches) {

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < num_updates * 2){

    int offset = tid % num_sketches;
    int start_index = (tid / num_sketches) * num_sketches;

    // Step 2: Update node's sketches
    for (int i = 0; i < num_sketches; i++) {
      // To prevent going out of bounds
      if (start_index + i >= num_updates * 2) {
        break;
      }

      // Step 1: Get node based on tid.
      const vec_t_cu node = nodeUpdates[start_index + i];
      CudaSketch curr_cudaSketch = cudaSketches[(node * num_sketches) + offset];

      size_t num_elems = curr_cudaSketch.num_elems;

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[start_index + i], update_hash);

      // Update node's sketches
      for (unsigned j = 0; j < curr_cudaSketch.num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
          unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[start_index + i], update_hash);
          } else break;
        }
      }
    }
  }
}

// Kernel code of handling all the stream updates
/*__global__ void singleStream_update(int* nodeUpdates, int num_updates, int num_nodes, int num_sketches, vec_t* edgeUpdates, 
                                    CudaSketch* cudaSketches) {

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // One thread will be responsible for one edge update = one thread is updating sketches on each endpoint nodes (2).
  if (tid < num_updates){
    // Step 1: Get two endpoint nodes based on tid.
    int node1_id = tid * 2;
    int node2_id = (tid * 2) + 1;

    const vec_t_cu node1 = nodeUpdates[node1_id];
    const vec_t_cu node2 = nodeUpdates[node2_id];

    // Step 2a: Update node1's sketches
    for (int i = 0; i < num_sketches; i++) {

      CudaSketch curr_cudaSketch = cudaSketches[(node1 * num_sketches) + i];
      const size_t num_elems = curr_cudaSketch.num_elems;

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[node1_id], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[node1_id], update_hash);

      // Update node1's sketches
      for (unsigned j = 0; j < curr_cudaSketch.num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[node1_id], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
          unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[node1_id], update_hash);
          } else break;
        }
      }
    }

    // Step 2b: Update node2's sketches
    for (int i = 0; i < num_sketches; i++) {

      CudaSketch curr_cudaSketch = cudaSketches[(node2 * num_sketches) + i];
      size_t num_elems = curr_cudaSketch.num_elems;

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[node2_id], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[node2_id], update_hash);
      
      // Update node2's sketches
      for (unsigned j = 0; j < curr_cudaSketch.num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[node2_id], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
          unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[node2_id], update_hash);
          } else break;
        }
      }
    }
  }
}*/

// Kernel code of handling all the stream updates
__global__ void singleStream_update(int* nodeUpdates, int num_updates, int num_nodes, int num_sketches, vec_t* edgeUpdates, 
                                    CudaSketch* cudaSketches) {

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // One thread will be responsible for one edge update = one thread is updating sketches on each endpoint nodes (2).
  if (tid < num_updates){
    // Step 1: Get two endpoint nodes based on tid.
    int node1_id = tid * 2;
    int node2_id = (tid * 2) + 1;

    int offset = node1_id % num_sketches;
    int start_index = (node1_id / num_sketches) * num_sketches;

    // Step 2a: Update node1's sketches
    for (int i = 0; i < num_sketches; i++) {
      // To prevent going out of bounds
      if (start_index + i >= num_updates * 2) {
        break;
      }

      const vec_t_cu node = nodeUpdates[start_index + i];
      CudaSketch curr_cudaSketch = cudaSketches[(node * num_sketches) + offset];

      const size_t num_elems = curr_cudaSketch.num_elems;

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[start_index + i], update_hash);

      // Update node1's sketches
      for (unsigned j = 0; j < curr_cudaSketch.num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
          unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[start_index + i], update_hash);
          } else break;
        }
      }
    }

    offset = node2_id % num_sketches;
    start_index = (node2_id / num_sketches) * num_sketches;

    // Step 2b: Update node2's sketches
    for (int i = 0; i < num_sketches; i++) {
      // To prevent going out of bounds
      if (start_index + i >= num_updates * 2) {
        break;
      }

      const vec_t_cu node = nodeUpdates[start_index + i];
      CudaSketch curr_cudaSketch = cudaSketches[(node * num_sketches) + offset];

      size_t num_elems = curr_cudaSketch.num_elems;

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[start_index + i], update_hash);
      
      // Update node2's sketches
      for (unsigned j = 0; j < curr_cudaSketch.num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
          unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[start_index + i], update_hash);
          } else break;
        }
      }
    }
  }
}

// Function that calls sketch update kernel code.
void sketchUpdate(int num_threads, int num_blocks, int num_updates, vec_t* update_indexes, CudaSketch* cudaSketches) {
  // Call kernel code
  sketch_update<<<num_blocks, num_threads>>>(num_updates, update_indexes, cudaSketches);
  cudaDeviceSynchronize();
}


// Function that calls stream update kernel code.
void streamUpdate(int num_threads, int num_blocks, int *nodeUpdates, size_t num_updates, node_id_t num_nodes, int num_sketches, int num_elems, vec_t *edgeUpdates, 
                  CudaSketch* cudaSketches, int num_threads_per_update) {

  if(num_threads_per_update == 1) { // Updating sketches with one thread per edge update
    singleStream_update<<<num_blocks, num_threads>>>(nodeUpdates, num_updates, num_nodes, num_sketches, edgeUpdates, cudaSketches);
    cudaDeviceSynchronize();
  }
  else if(num_threads_per_update == 2) { // Updating sketches with two thread per edge update
    doubleStream_update<<<num_blocks, num_threads>>>(nodeUpdates, num_updates, num_nodes, num_sketches, edgeUpdates, cudaSketches);
    cudaDeviceSynchronize();
  }
  else if(num_threads_per_update == 4) { // Updating sketches with four threads per edge update
    quadStream_update<<<num_blocks, num_threads>>>(nodeUpdates, num_updates, num_nodes, num_sketches, edgeUpdates, cudaSketches);
    cudaDeviceSynchronize();
  }
  else {
    std::cout << "(cuda_kernel.cu) ERROR: Invalid number of threads per edge update. Must be 1, 2, or 4." << std::endl;
    return;
  }
}

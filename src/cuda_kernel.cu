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

class CudaSupernode {
  public:
    CudaSketch* cudaSketches;
    CudaSketch* deltaSketches;
    int num_sketches;
    uint64_t seed;

    // Default Constructor of CudaSupernode
    CudaSupernode():cudaSketches(nullptr), seed(0), num_sketches(0) {};

    CudaSupernode(CudaSketch* sketches, uint64_t node_seed, int number_sketches) {
      cudaMallocManaged(&cudaSketches, number_sketches * sizeof(CudaSketch));
      cudaMallocManaged(&deltaSketches, number_sketches * sizeof(CudaSketch));
      seed = node_seed;

      for (size_t i = 0; i < number_sketches; ++i) {
        cudaSketches[i] = sketches[i];
        
        vec_t* delta_bucket_a;
        vec_hash_t* delta_bucket_c;

        cudaMallocManaged(&delta_bucket_a, cudaSketches[i].num_elems * sizeof(vec_t));
        cudaMallocManaged(&delta_bucket_c, cudaSketches[i].num_elems * sizeof(vec_hash_t));

        // initialize bucket values
        for (size_t j = 0; j < cudaSketches[j].num_elems; ++j) {
          delta_bucket_a[j] = 0;
          delta_bucket_c[j] = 0;
        }
        
        CudaSketch deltaSketch(delta_bucket_a, delta_bucket_c, cudaSketches[i].failure_factor, cudaSketches[i].num_elems, cudaSketches[i].num_buckets, cudaSketches[i].num_guesses, seed);
        deltaSketches[i] = deltaSketch;
        seed += guess_gen(cudaSketches[i].failure_factor); // sketch_width
      }
      num_sketches = number_sketches;
    };
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

// Kernel code for only sketch updates
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
    vec_t_cu* cudaSketch_bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
    vec_hash_t* cudaSketch_bucket_c = curr_cudaSketch.d_bucket_c;
    size_t num_elems = curr_cudaSketch.num_elems;

    // Step 4: Get update_hash
    vec_hash_t update_hash = bucket_index_hash(update_idx, curr_cudaSketch.seed);

    bucket_update(cudaSketch_bucket_a[num_elems - 1], cudaSketch_bucket_c[num_elems - 1], update_idx, update_hash);

    // Step 5: Update current sketch
    for (unsigned i = 0; i < curr_cudaSketch.num_buckets; ++i) {
      col_hash_t col_index_hash = bucket_col_index_hash(update_idx, curr_cudaSketch.seed + i);
      for (unsigned j = 0; j < curr_cudaSketch.num_guesses; ++j) {
        unsigned bucket_id = i * curr_cudaSketch.num_guesses + j;
        if (bucket_contains(col_index_hash, ((col_hash_t)1) << j)){
          bucket_update(cudaSketch_bucket_a[bucket_id], cudaSketch_bucket_c[bucket_id], update_idx, update_hash);
        } else break;
      }
    }
  }
}

// Kernel code of handling all the stream updates
__global__ void doubleStream_update(int* nodeUpdates, int num_updates, int num_nodes, vec_t* edgeUpdates, CudaSupernode* cudaSupernodes) {

  // Get thread id
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Two threads will be responsible for one edge update = one thread is only modifying one node's sketches.
  if(tid < (num_updates * 2)) {

    // Step 1: Get node based on tid.
    const vec_t_cu node = nodeUpdates[tid];

    // Step 2: Get number of sketches for current node.
    int node_num_sketches = cudaSupernodes[node].num_sketches;

    // Step 3: Update node's sketches
    for (int i = 0; i < node_num_sketches; i++) {

      CudaSketch curr_cudaSketch = cudaSupernodes[node].cudaSketches[i];
      CudaSketch curr_deltaSketch = cudaSupernodes[node].deltaSketches[i];
      size_t num_elems = curr_cudaSketch.num_elems;

      // Get all the buckets from deltaSketch
      vec_t_cu* deltaSketch_bucket_a = (vec_t_cu*)curr_deltaSketch.d_bucket_a;
      vec_hash_t* deltaSketch_bucket_c = curr_deltaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[tid], curr_deltaSketch.seed);

      bucket_update(deltaSketch_bucket_a[num_elems - 1], deltaSketch_bucket_c[num_elems - 1], edgeUpdates[tid], update_hash);

      // Update node's deltaSketch
      for (unsigned j = 0; j < curr_cudaSketch.num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[tid], curr_deltaSketch.seed + j);
        for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
          unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(deltaSketch_bucket_a[bucket_id], deltaSketch_bucket_c[bucket_id], edgeUpdates[tid], update_hash);
          } else break;
        }
      }
    }
  }
}

// Kernel code of handling all the stream updates
__global__ void singleStream_update(int* nodeUpdates, int num_updates, int num_nodes, vec_t* edgeUpdates, CudaSupernode* cudaSupernodes) {

  // Get thread id
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // One thread will be responsible for one edge update = one thread is updating sketches on each endpoint nodes (2).
  if(tid < num_updates) {
    // Step 1: Get two endpoint nodes based on tid.
    const vec_t_cu node1 = nodeUpdates[tid * 2];
    const vec_t_cu node2 = nodeUpdates[(tid * 2) + 1];

    // Step 2: Get number of sketches for each endpoint nodes
    int node1_num_sketches = cudaSupernodes[node1].num_sketches;
    int node2_num_sketches = cudaSupernodes[node2].num_sketches;

    // Step 3a: Update node1's sketches
    for (int i = 0; i < node1_num_sketches; i++) {

      CudaSketch curr_cudaSketch = cudaSupernodes[node1].cudaSketches[i];
      CudaSketch curr_deltaSketch = cudaSupernodes[node1].deltaSketches[i];
      size_t num_elems = curr_cudaSketch.num_elems;

      // Get all the buckets from deltaSketch
      vec_t_cu* deltaSketch_bucket_a = (vec_t_cu*)curr_deltaSketch.d_bucket_a;
      vec_hash_t* deltaSketch_bucket_c = curr_deltaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[tid * 2], curr_deltaSketch.seed);

      bucket_update(deltaSketch_bucket_a[num_elems - 1], deltaSketch_bucket_c[num_elems - 1], edgeUpdates[tid * 2], update_hash);

      // Update node1's deltaSketch
      for (unsigned j = 0; j < curr_cudaSketch.num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[tid * 2], curr_deltaSketch.seed + j);
        for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
          unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(deltaSketch_bucket_a[bucket_id], deltaSketch_bucket_c[bucket_id], edgeUpdates[tid * 2], update_hash);
          } else break;
        }
      }
    }

    __syncthreads();

    // Step 3b: Update node2's sketches
    for (int i = 0; i < node2_num_sketches; i++) {

      CudaSketch curr_cudaSketch = cudaSupernodes[node2].cudaSketches[i];
      CudaSketch curr_deltaSketch = cudaSupernodes[node2].deltaSketches[i];
      size_t num_elems = curr_cudaSketch.num_elems;

      // Get all the buckets from deltaSketch
      vec_t_cu* deltaSketch_bucket_a = (vec_t_cu*)curr_deltaSketch.d_bucket_a;
      vec_hash_t* deltaSketch_bucket_c = curr_deltaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[(tid * 2) + 1], curr_deltaSketch.seed);

      bucket_update(deltaSketch_bucket_a[num_elems - 1], deltaSketch_bucket_c[num_elems - 1], edgeUpdates[(tid * 2) + 1], update_hash);
      
      // Update node2's deltaSketch
      for (unsigned j = 0; j < curr_cudaSketch.num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[(tid * 2) + 1], curr_deltaSketch.seed + j);
        for (unsigned k = 0; k < curr_cudaSketch.num_guesses; ++k) {
          unsigned bucket_id = j * curr_cudaSketch.num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(deltaSketch_bucket_a[bucket_id], deltaSketch_bucket_c[bucket_id], edgeUpdates[(tid * 2) + 1], update_hash);
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
void streamUpdate(int num_threads, int num_blocks, int *nodeUpdates, size_t num_updates, node_id_t num_nodes, vec_t *edgeUpdates, CudaSupernode* cudaSupernodes, int num_threads_per_update) {

  if(num_threads_per_update == 1) { // Updating sketches with one thread per edge update
    singleStream_update<<<num_blocks, num_threads>>>(nodeUpdates, num_updates, num_nodes, edgeUpdates, cudaSupernodes);
    cudaDeviceSynchronize();
  }
  else if(num_threads_per_update == 2) { // Updating sketches with two thread per edge update
    doubleStream_update<<<num_blocks, num_threads>>>(nodeUpdates, num_updates, num_nodes, edgeUpdates, cudaSupernodes);
    cudaDeviceSynchronize();
  }
  else {
    std::cout << "(cuda_kernel.cu) ERROR: Invalid number of threads per edge update. Must be 1 or 2." << std::endl;
    return;
  }

  // Add all the supernodes' deltaSketch back to cudaSketch (Temporary, can be changed in the future)
  for(int i = 0; i < num_nodes; i++) {
    for(int j = 0; j < cudaSupernodes[i].num_sketches; j++) {
      CudaSketch curr_cudaSketch = cudaSupernodes[i].cudaSketches[j];
      CudaSketch curr_deltaSketch = cudaSupernodes[i].deltaSketches[j];

      for(int k = 0; k < curr_cudaSketch.num_elems; k++) {
        curr_cudaSketch.d_bucket_a[k] += curr_deltaSketch.d_bucket_a[k];
        curr_cudaSketch.d_bucket_c[k] += curr_deltaSketch.d_bucket_c[k];
      }
    }
  }
}

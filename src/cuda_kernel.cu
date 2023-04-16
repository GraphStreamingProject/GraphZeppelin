#include <vector>
#include <cuda_xxhash64.cuh>
#include <graph.h>

typedef unsigned long long int uint64_cu;
typedef uint64_cu vec_t_cu;

// CUDA API Check
// Source: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class CudaSketch {
  public:
    vec_t* d_bucket_a;
    vec_hash_t* d_bucket_c;

    uint64_t seed;

    // Default Constructor of CudaSketch
    CudaSketch():d_bucket_a(nullptr), d_bucket_c(nullptr) {};

    CudaSketch(vec_t* d_bucket_a, vec_hash_t* d_bucket_c, uint64_t seed): d_bucket_a(d_bucket_a), d_bucket_c(d_bucket_c), seed(seed) {};
};

class CudaParams {
  public:
    // List of node ids that thread will be responsible for updating 
    int *nodeUpdates;

    // List of edge ids that thread will be responsble for updating
    vec_t *edgeUpdates;

    // List of num updates for each node
    int *nodeNumUpdates;

    // List of starting index for each node's update
    int *nodeStartIndex;

    // Parameter for entire graph
    node_id_t num_nodes;
    size_t num_updates;
    
    // Parameter for each supernode (consistent with other supernodes)
    int num_sketches;
    
    // Parameter for each sketch (consistent with other sketches)
    size_t num_elems;
    size_t num_buckets;
    size_t num_guesses;

    // Default Constructor of CudaParams
    CudaParams():nodeUpdates(nullptr), edgeUpdates(nullptr), nodeNumUpdates(nullptr), nodeStartIndex(nullptr) {};
    
    CudaParams(node_id_t num_nodes, size_t num_updates, int num_sketches, size_t num_elems, size_t num_buckets, size_t num_guesses):
      num_nodes(num_nodes), num_updates(num_updates), num_sketches(num_sketches), num_elems(num_elems), num_buckets(num_buckets), num_guesses(num_guesses) {
      
      // Allocate memory space for GPU
      gpuErrchk(cudaMallocManaged(&nodeUpdates, 2 * num_updates * sizeof(node_id_t)));
      gpuErrchk(cudaMallocManaged(&edgeUpdates, 2 * num_updates * sizeof(vec_t)));
      gpuErrchk(cudaMallocManaged(&nodeNumUpdates, num_nodes * sizeof(node_id_t)));
      gpuErrchk(cudaMallocManaged(&nodeStartIndex, num_nodes * sizeof(node_id_t)));
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

__device__ void bucket_a_single_update(vec_t_cu& a, const vec_t_cu& update_idx) {
  atomicXor(&a, update_idx);
}

__device__ void bucket_c_single_update(vec_hash_t& c, const vec_hash_t& update_hash) {
  atomicXor(&c, update_hash);
}

// Kernel code for only sketch updates
// Old version, needs to be updated
/*__global__ void sketch_update(int num_updates, vec_t* update_indexes, CudaSketch* cudaSketches) {

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
}*/

// Version 2: Kernel code of handling all the stream updates
// Two threads will be responsible for one edge update -> one thread is only modifying one node's sketches.
/*__global__ void doubleStream_update(CudaParams* cudaParams, CudaSketch* cudaSketches) {

  // Get thread id
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Get variables from cudaParams
  int *nodeUpdates = cudaParams[0].nodeUpdates;
  vec_t *edgeUpdates = cudaParams[0].edgeUpdates;

  node_id_t num_nodes = cudaParams[0].num_nodes;
  size_t num_updates = cudaParams[0].num_updates;
  
  int num_sketches = cudaParams[0].num_sketches;
  
  size_t num_elems = cudaParams[0].num_elems;
  size_t num_buckets = cudaParams[0].num_buckets;
  size_t num_guesses = cudaParams[0].num_guesses;

  if (tid < num_updates * 2){

    // Step 1: Get node based on tid.
    const vec_t_cu node = nodeUpdates[tid];

    // Step 2: Update node's sketches
    for (int i = 0; i < num_sketches; i++) {

      CudaSketch curr_cudaSketch = cudaSketches[(node * num_sketches) + i];

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[tid], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[tid], update_hash);

      // Update node's sketches
      for (unsigned j = 0; j < num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[tid], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < num_guesses; ++k) {
          unsigned bucket_id = j * num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[tid], update_hash);
          } else break;
        }
      }
    }
  }
}*/

// Version 3: Kernel code of handling all the stream updates
// Two threads will be responsible for one edge update -> one thread is only modifying one node's sketches.
/*__global__ void doubleStream_update(CudaParams* cudaParams, CudaSketch* cudaSketches) {

  // Get thread id
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Get variables from cudaParams
  int *nodeUpdates = cudaParams[0].nodeUpdates;
  vec_t *edgeUpdates = cudaParams[0].edgeUpdates;

  node_id_t num_nodes = cudaParams[0].num_nodes;
  size_t num_updates = cudaParams[0].num_updates;
  
  int num_sketches = cudaParams[0].num_sketches;
  
  size_t num_elems = cudaParams[0].num_elems;
  size_t num_buckets = cudaParams[0].num_buckets;
  size_t num_guesses = cudaParams[0].num_guesses;

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

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[start_index + i], update_hash);

      // Update node's sketches
      for (unsigned j = 0; j < num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < num_guesses; ++k) {
          unsigned bucket_id = j * num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[start_index + i], update_hash);
          } else break;
        }
      }
    }
  }
}*/

// Version 4: Kernel code of handling all the stream updates
// Two threads will be responsible for one edge update -> one thread is only modifying one node's sketches.
// Placing sketches in shared memory
/*__global__ void doubleStream_update(int* nodeUpdates, vec_t* edgeUpdates, int* nodeNumUpdates, int* nodeStartIndex, node_id_t num_nodes, size_t num_updates,
    int num_sketches, size_t num_elems, size_t num_buckets, size_t num_guesses, CudaSketch* cudaSketches, long* sketchSeeds) {

  if (blockIdx.x < num_nodes){
    
    extern __shared__ vec_t_cu sketches[];
    vec_t_cu* bucket_a = sketches;
    vec_hash_t* bucket_c = (vec_hash_t*)&bucket_a[num_elems * num_sketches];
    int node = blockIdx.x;
    int startIndex = nodeStartIndex[node];

    // Have one thread to initialize shared memory
    if (threadIdx.x == 0) {
      for (int i = 0; i < num_sketches; i++) {
        for (int j = 0; j < num_elems; j++) {
          bucket_a[(i * num_elems) + j] = 0;
          bucket_c[(i * num_elems) + j] = 0;
        }
      }
    }

    __syncthreads();

    // Update node's sketches
    for (int id = threadIdx.x; id < nodeNumUpdates[node]; id += blockDim.x) {
      
      for (int i = 0; i < num_sketches; i++) {
        vec_hash_t update_hash = bucket_index_hash(edgeUpdates[startIndex + id], sketchSeeds[(node * num_sketches) + i]);

        bucket_update(bucket_a[(i * num_elems) + num_elems - 1], bucket_c[(i * num_elems) + num_elems - 1], edgeUpdates[startIndex + id], update_hash);

        // Update node's sketches
        for (unsigned j = 0; j < num_buckets; ++j) {
          col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[startIndex + id], sketchSeeds[(node * num_sketches) + i] + j);
          for (unsigned k = 0; k < num_guesses; ++k) {
            unsigned bucket_id = j * num_guesses + k;
            if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
              bucket_update(bucket_a[(i * num_elems) + bucket_id], bucket_c[(i * num_elems) + bucket_id], edgeUpdates[startIndex + id], update_hash);
            } else break;
          }
        }
      }
    }

    __syncthreads();

    // Have one thread to transfer sketches back to global memory
    if (threadIdx.x == 0) {
      for (int i = 0; i < num_sketches; i++) {
        CudaSketch curr_cudaSketch = cudaSketches[(node * num_sketches) + i];

        vec_t_cu* curr_bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
        vec_hash_t* curr_bucket_c = curr_cudaSketch.d_bucket_c;

        for (int j = 0; j < num_elems; j++) {
          curr_bucket_a[j] = bucket_a[(i * num_elems) + j];
          curr_bucket_c[j] = bucket_c[(i * num_elems) + j];
        }
      }
    }
  }
}*/

// Version 5: Kernel code of handling all the stream updates
// Two threads will be responsible for one edge update -> one thread is only modifying one node's sketches.
// Placing sketches in shared memory, each thread is doing log n updates on one slice of sketch
__global__ void doubleStream_update(int* nodeUpdates, vec_t* edgeUpdates, int* nodeNumUpdates, int* nodeStartIndex, node_id_t num_nodes, size_t num_updates,
    int num_sketches, size_t num_elems, size_t num_buckets, size_t num_guesses, CudaSketch* cudaSketches, long* sketchSeeds) {

  if (blockIdx.x < num_nodes){
    
    extern __shared__ vec_t_cu sketches[];
    vec_t_cu* bucket_a = sketches;
    vec_hash_t* bucket_c = (vec_hash_t*)&bucket_a[num_elems * num_sketches];
    int node = blockIdx.x;
    int startIndex = nodeStartIndex[node];

    // Have one thread to initialize shared memory
    if (threadIdx.x == 0) {
      for (int i = 0; i < num_sketches; i++) {
        for (int j = 0; j < num_elems; j++) {
          bucket_a[(i * num_elems) + j] = 0;
          bucket_c[(i * num_elems) + j] = 0;
        }
      }
    }

    __syncthreads();

    // Update node's sketches
    for (int id = threadIdx.x; id < nodeNumUpdates[node] + num_sketches; id += blockDim.x) {
      
      int sketch_offset = id % num_sketches;
      int update_offset = ((id / num_sketches) * num_sketches);
      
      for (int i = 0; i < num_sketches; i++) {

        if ((startIndex + update_offset + i) >= startIndex + nodeNumUpdates[node]) {
          break;
        }

        vec_hash_t update_hash = bucket_index_hash(edgeUpdates[startIndex + update_offset + i], sketchSeeds[(node * num_sketches) + sketch_offset]);

        bucket_update(bucket_a[(sketch_offset * num_elems) + num_elems - 1], bucket_c[(sketch_offset * num_elems) + num_elems - 1], edgeUpdates[startIndex + update_offset + i], update_hash);

        // Update node's sketches
        for (unsigned j = 0; j < num_buckets; ++j) {
          col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[startIndex + update_offset + i], sketchSeeds[(node * num_sketches) + sketch_offset] + j);
          for (unsigned k = 0; k < num_guesses; ++k) {
            unsigned bucket_id = j * num_guesses + k;
            if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
              bucket_update(bucket_a[(sketch_offset * num_elems) + bucket_id], bucket_c[(sketch_offset * num_elems) + bucket_id], edgeUpdates[startIndex + update_offset + i], update_hash);
            } else break;
          }
        }
      }
    }

    __syncthreads();

    // Have one thread to transfer sketches back to global memory
    if (threadIdx.x == 0) {
      for (int i = 0; i < num_sketches; i++) {
        CudaSketch curr_cudaSketch = cudaSketches[(node * num_sketches) + i];

        vec_t_cu* curr_bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
        vec_hash_t* curr_bucket_c = curr_cudaSketch.d_bucket_c;

        for (int j = 0; j < num_elems; j++) {
          curr_bucket_a[j] = bucket_a[(i * num_elems) + j];
          curr_bucket_c[j] = bucket_c[(i * num_elems) + j];
        }
      }
    }
  }
}

// Version 2: Kernel code of handling all the stream updates
/*__global__ void singleStream_update(CudaParams* cudaParams, CudaSketch* cudaSketches) {

  // Get thread id
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Get variables from cudaParams
  int *nodeUpdates = cudaParams[0].nodeUpdates;
  vec_t *edgeUpdates = cudaParams[0].edgeUpdates;

  node_id_t num_nodes = cudaParams[0].num_nodes;
  size_t num_updates = cudaParams[0].num_updates;
  
  int num_sketches = cudaParams[0].num_sketches;
  
  size_t num_elems = cudaParams[0].num_elems;
  size_t num_buckets = cudaParams[0].num_buckets;
  size_t num_guesses = cudaParams[0].num_guesses;

  // One thread will be responsible for one edge update = one thread is updating sketches on each endpoint nodes (2).
  if (tid < num_updates){
    // Step 1: Get two endpoint nodes based on tid.
    //int node1_id = tid * 2;
    //int node2_id = (tid * 2) + 1;
    int node1_id = tid;
    int node2_id = tid + num_updates;

    const vec_t_cu node1 = nodeUpdates[node1_id];
    const vec_t_cu node2 = nodeUpdates[node2_id];

    // Step 2a: Update node1's sketches
    for (int i = 0; i < num_sketches; i++) {

      CudaSketch curr_cudaSketch = cudaSketches[(node1 * num_sketches) + i];

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[node1_id], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[node1_id], update_hash);

      // Update node1's sketches
      for (unsigned j = 0; j < num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[node1_id], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < num_guesses; ++k) {
          unsigned bucket_id = j * num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[node1_id], update_hash);
          } else break;
        }
      }
    }

    // Step 2b: Update node2's sketches
    for (int i = 0; i < num_sketches; i++) {

      CudaSketch curr_cudaSketch = cudaSketches[(node2 * num_sketches) + i];

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[node2_id], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[node2_id], update_hash);
      
      // Update node2's sketches
      for (unsigned j = 0; j < num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[node2_id], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < num_guesses; ++k) {
          unsigned bucket_id = j * num_guesses + k;
          if (bucket_contains(col_index_hash, ((col_hash_t)1) << k)){
            bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edgeUpdates[node2_id], update_hash);
          } else break;
        }
      }
    }
  }
}*/

// Version 3: Kernel code of handling all the stream updates
__global__ void singleStream_update(CudaParams* cudaParams, CudaSketch* cudaSketches) {

  // Get thread id
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Get variables from cudaParams
  int *nodeUpdates = cudaParams[0].nodeUpdates;
  vec_t *edgeUpdates = cudaParams[0].edgeUpdates;

  node_id_t num_nodes = cudaParams[0].num_nodes;
  size_t num_updates = cudaParams[0].num_updates;
  
  int num_sketches = cudaParams[0].num_sketches;
  
  size_t num_elems = cudaParams[0].num_elems;
  size_t num_buckets = cudaParams[0].num_buckets;
  size_t num_guesses = cudaParams[0].num_guesses;

  // One thread will be responsible for one edge update = one thread is updating sketches on each endpoint nodes (2).
  if (tid < num_updates){
    // Step 1: Get two endpoint nodes based on tid.
    //int node1_id = tid * 2;
    //int node2_id = (tid * 2) + 1;

    int node1_id = tid;
    int node2_id = tid + num_updates;

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

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[start_index + i], update_hash);

      // Update node1's sketches
      for (unsigned j = 0; j < num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < num_guesses; ++k) {
          unsigned bucket_id = j * num_guesses + k;
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

      // Get buckets based on current sketch and node id
      vec_t_cu* bucket_a = (vec_t_cu*)curr_cudaSketch.d_bucket_a;
      vec_hash_t* bucket_c = curr_cudaSketch.d_bucket_c;

      vec_hash_t update_hash = bucket_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed);

      bucket_update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], edgeUpdates[start_index + i], update_hash);
      
      // Update node2's sketches
      for (unsigned j = 0; j < num_buckets; ++j) {
        col_hash_t col_index_hash = bucket_col_index_hash(edgeUpdates[start_index + i], curr_cudaSketch.seed + j);
        for (unsigned k = 0; k < num_guesses; ++k) {
          unsigned bucket_id = j * num_guesses + k;
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
  //sketch_update<<<num_blocks, num_threads>>>(num_updates, update_indexes, cudaSketches);
  //cudaDeviceSynchronize();
}


// Function that calls stream update kernel code.
void streamUpdate(int num_threads, int num_blocks, CudaParams* cudaParams, CudaSketch* cudaSketches, long* sketchSeeds, int num_threads_per_update) {

  if(num_threads_per_update == 1) { // Updating sketches with one thread per edge update
    singleStream_update<<<num_blocks, num_threads>>>(cudaParams, cudaSketches);
  }
  else if(num_threads_per_update == 2) { // Updating sketches with two thread per edge update
    // Unwarp variables from cudaParams
    int *nodeUpdates = cudaParams[0].nodeUpdates;
    vec_t *edgeUpdates = cudaParams[0].edgeUpdates;
    int *nodeNumUpdates = cudaParams[0].nodeNumUpdates;
    int *nodeStartIndex = cudaParams[0].nodeStartIndex;

    node_id_t num_nodes = cudaParams[0].num_nodes;
    size_t num_updates = cudaParams[0].num_updates;
    
    int num_sketches = cudaParams[0].num_sketches;
    
    size_t num_elems = cudaParams[0].num_elems;
    size_t num_buckets = cudaParams[0].num_buckets;
    size_t num_guesses = cudaParams[0].num_guesses;

    int maxbytes = num_elems * num_sketches * sizeof(vec_t_cu) + num_elems * num_sketches * sizeof(vec_hash_t);
    cudaFuncSetAttribute(doubleStream_update, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

    /*
      Note (Only when using shared memory): I have noticed that unwrapping variables within kernel code
        caused these parameter variables to stay within global memory, creating more latency. Therefore, unwrapping these 
        variables then passing as argument of the kernel code avoids that issue.
    */ 
    doubleStream_update<<<num_blocks, num_threads, maxbytes>>>(nodeUpdates, edgeUpdates, nodeNumUpdates, nodeStartIndex, num_nodes, num_updates, num_sketches, num_elems, num_buckets, num_guesses, cudaSketches, sketchSeeds);
    //doubleStream_update<<<num_blocks, num_threads>>>(cudaParams, cudaSketches);
  }
  else {
    std::cout << "(cuda_kernel.cu) ERROR: Invalid number of threads per edge update. Must be 1 or 2." << std::endl;
    return;
  }
  cudaDeviceSynchronize();
}

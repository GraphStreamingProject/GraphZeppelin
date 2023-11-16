#pragma once
#include <atomic>
#include <graph.h>
#include <sketch.h>
#include "../src/cuda_library.cu"

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

/*
*   
*   Helper Classes for sketches
*
*/

class CudaSketch {
  public:
    vec_t* bucket_a;
    vec_hash_t* bucket_c;

    // Default Constructor of CudaSketch
    CudaSketch():bucket_a(nullptr), bucket_c(nullptr) {};

    CudaSketch(vec_t* bucket_a, vec_hash_t* bucket_c): bucket_a(bucket_a), bucket_c(bucket_c) {};
};

class CudaSupernode {
  public:
    CudaSketch* cudaSketches;
    int src = 0;
    bool deltaApplied = true;

    CudaSupernode(): cudaSketches(nullptr) {};
};

class CudaUpdateParams {
  public:
    // Value of k
    int k;

    size_t num_inserted_updates = 0;

    // List of edge ids that thread will be responsble for updating
    vec_t *h_edgeUpdates, *d_edgeUpdates;

    vec_t *h_bucket_a, *d_bucket_a;
    vec_hash_t *h_bucket_c, *d_bucket_c;

    // Parameter for entire graph
    node_id_t num_nodes;
    vec_t num_updates;
    
    // Parameter for each supernode (consistent with other supernodes)
    int num_sketches;
    
    // Parameter for each sketch (consistent with other sketches)
    size_t num_elems;
    size_t num_columns;
    size_t num_guesses;

    int num_host_threads;
    int batch_size;
    int stream_multiplier; 

    // Default Constructor of CudaUpdateParams
    CudaUpdateParams():h_edgeUpdates(nullptr), d_edgeUpdates(nullptr) {};
    
    CudaUpdateParams(node_id_t num_nodes, size_t num_updates, int num_sketches, size_t num_elems, size_t num_columns, size_t num_guesses, int num_host_threads, int batch_size, int stream_multiplier, int k = 1):
      num_nodes(num_nodes), num_updates(num_updates), num_sketches(num_sketches), num_elems(num_elems), num_columns(num_columns), num_guesses(num_guesses), num_host_threads(num_host_threads), batch_size(batch_size), stream_multiplier(stream_multiplier), k(k) {
      
      // Allocate memory for buffer that stores edge updates
      gpuErrchk(cudaMallocHost(&h_edgeUpdates, stream_multiplier * num_host_threads * batch_size * sizeof(vec_t)));
      gpuErrchk(cudaMalloc(&d_edgeUpdates, stream_multiplier * num_host_threads * batch_size * sizeof(vec_t)));

      // Allocate memory for buckets 
      gpuErrchk(cudaMallocHost(&h_bucket_a, k * stream_multiplier * num_host_threads * num_sketches * num_elems * sizeof(vec_t)));
      gpuErrchk(cudaMalloc(&d_bucket_a, k * stream_multiplier * num_host_threads * num_sketches * num_elems * sizeof(vec_t)));
      gpuErrchk(cudaMallocHost(&h_bucket_c, k * stream_multiplier * num_host_threads * num_sketches * num_elems * sizeof(vec_hash_t)));
      gpuErrchk(cudaMalloc(&d_bucket_c, k * stream_multiplier * num_host_threads * num_sketches * num_elems * sizeof(vec_hash_t)));

      std::cout << "Allocated buckets\n";
      
      // Initialize host buckets
      for (size_t i = 0; i < k * stream_multiplier * num_host_threads * num_sketches * num_elems; i++) {
        h_bucket_a[i] = 0;
        h_bucket_c[i] = 0;
      }

    };
};

struct CudaQuery {
  Edge edge;
  SampleSketchRet ret_code;
};

struct CudaToMerge {
  node_id_t* children;
  int* size;
};

class CudaCCParams {
  public:
    // List of node ids that need to be sampled
    node_id_t* reps;

    node_id_t* temp_reps;

    // List of querys
    CudaQuery* query;

    // List of parent of each node id
    node_id_t* parent;

    // List of parent of each node id
    node_id_t* size;

    // List of node ids to be merged
    CudaToMerge* to_merge;
    node_id_t* merge_children;
    int* merge_size;

    // Number of remaining supernodes in a graph
    // [0]: Current reps size
    // [1]: num_nodes of the graph
    node_id_t* num_nodes;

    // Indicate if supernode has been merged or not
    bool* modified; 

    // List of sample_idx and merged_sketches for all nodes
    size_t* sample_idxs;
    size_t* merged_sketches;

    // Parameter for each supernode (consistent with other supernodes)
    int num_sketches;

    // Parameter for each sketch (consistent with other sketches)
    size_t num_elems;
    size_t num_columns;
    size_t num_guesses;

    CudaCCParams(node_id_t total_nodes, int num_sketches, size_t num_elems, size_t num_columns, size_t num_guesses): 
      num_sketches(num_sketches), num_elems(num_elems), num_columns(num_columns), num_guesses(num_guesses) {

      gpuErrchk(cudaMallocManaged(&num_nodes, 2 * sizeof(node_id_t)));
      num_nodes[0] = total_nodes;
      num_nodes[1] = total_nodes;

      // Allocate memory space for GPU
      gpuErrchk(cudaMallocManaged(&reps, num_nodes[0] * sizeof(node_id_t)));
      gpuErrchk(cudaMallocManaged(&temp_reps, num_nodes[0] * sizeof(node_id_t)));
      gpuErrchk(cudaMallocManaged(&query, num_nodes[0] * sizeof(CudaQuery)));
      gpuErrchk(cudaMallocManaged(&parent, num_nodes[0] * sizeof(node_id_t)));
      gpuErrchk(cudaMallocManaged(&size, num_nodes[0] * sizeof(node_id_t)));

      gpuErrchk(cudaMallocManaged(&to_merge, num_nodes[0] * sizeof(CudaToMerge)));

      gpuErrchk(cudaMallocManaged(&modified, sizeof(bool)));
      gpuErrchk(cudaMallocManaged(&sample_idxs, num_nodes[0] * sizeof(size_t)));
      gpuErrchk(cudaMallocManaged(&merged_sketches, num_nodes[0] * sizeof(size_t)));

      gpuErrchk(cudaMallocManaged(&merge_children, num_nodes[0] * num_nodes[0] * sizeof(node_id_t)));
      memset(merge_children, 0, num_nodes[0] * num_nodes[0] * sizeof(node_id_t));

      gpuErrchk(cudaMallocManaged(&merge_size, num_nodes[0] * sizeof(int)));
      memset(merge_size, 0, num_nodes[0] * sizeof(int));

      for (size_t i = 0; i < num_nodes[0]; i++) {
        to_merge[i] = CudaToMerge{&merge_children[i * num_nodes[0]], &merge_size[i]};
      }
      modified[0] = false;
    };

    void reset() {
      for (size_t i = 0; i < num_nodes[0]; i++) {
        temp_reps[i] = 0;
        for (int j = 0; j < to_merge[i].size[0]; j++) {
          to_merge[i].children[j] = 0;
        }
        to_merge[i].size[0] = 0;
      }
    }
};

class CudaKernel {
  public:
    /*
    *   
    *   Sketch's Update Functions
    *
    */

    void gtsStreamUpdate(int num_threads, int num_blocks, vec_t bucket_id, node_id_t src, cudaStream_t stream, vec_t prev_offset, size_t update_size, CudaUpdateParams* cudaUpdateParams, long* sketchSeeds);
    void k_gtsStreamUpdate(int num_threads, int num_blocks, int graph_id, int k, vec_t bucket_id, node_id_t src, cudaStream_t stream, vec_t prev_offset, size_t update_size, CudaUpdateParams* cudaUpdateParams, long* sketchSeeds);
 
    void kernelUpdateSharedMemory(int maxBytes);

    /*
    *   
    *   Sketch's Query Functions
    *
    */

    void cuda_sample_supernodes(int num_threads, int num_blocks, CudaCCParams* cudaCCParams, CudaSketch* cudaSketches);
    void cuda_supernodes_to_merge(int num_threads, int num_blocks, CudaCCParams* cudaCCParams);
    void cuda_merge_supernodes(int num_threads, int num_blocks, CudaCCParams* cudaCCParams, CudaSketch* cudaSketches);
};
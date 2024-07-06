#pragma once

#include <atomic>
#include <cmath>
#include <map>
#include <mutex>
#include "cc_sketch_alg.h"
#include "cuda_kernel.cuh"

class SKGPUSketchAlg : public CCSketchAlg{
private:
  node_id_t num_nodes;
  size_t num_updates;

  // List of edge ids that thread will be responsble for updating
  vec_t *edgeUpdates;
  vec_t *update_sizes;
  node_id_t *update_src;
  vec_t *update_start_index;

  std::map<uint64_t, uint64_t> batch_sizes;
  std::map<uint64_t, uint64_t> batch_src;
  std::map<uint64_t, uint64_t> batch_start_index;
  std::mutex batch_mutex;

  // Atomic variables
  std::atomic<uint64_t> edgeUpdate_offset;
  std::atomic<uint64_t> batch_count;

  vec_t *bucket_a;
  vec_hash_t *bucket_c;

  size_t sketchSeed;
  size_t maxBytes;

  CudaKernel cudaKernel;

  // Variables from sketch
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;

  // Number of threads and thread blocks for CUDA kernel
  int num_device_threads;
  int num_device_blocks;

  int device_id;

  // Number of CPU's graph workers
  int num_host_threads;

  // Maximum number of edge updates in one batch
  int batch_size;

public:
  SKGPUSketchAlg(node_id_t _num_nodes, size_t _num_updates, int num_threads, size_t seed, CCAlgConfiguration config = CCAlgConfiguration()) : CCSketchAlg(_num_nodes, seed, config){ 

    // Start timer for initializing
    auto init_start = std::chrono::steady_clock::now();

    num_nodes = _num_nodes;
    num_updates = _num_updates;

    edgeUpdate_offset = 0;
    batch_count = 0;
    
    num_host_threads = num_threads;
    sketchSeed = seed;

    // Get variables from sketch
    num_samples = Sketch::calc_cc_samples(num_nodes, 1);
    num_columns = num_samples * Sketch::default_cols_per_sample;
    bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
    num_buckets = num_columns * bkt_per_col + 1;

    std::cout << "num_samples: " << num_samples << "\n";
    std::cout << "num_buckets: " << num_buckets << "\n";
    std::cout << "num_columns: " << num_columns << "\n";
    std::cout << "bkt_per_col: " << bkt_per_col << "\n"; 

    batch_size = get_desired_updates_per_batch();
    std::cout << "Batch Size: " << batch_size << "\n";

    device_id = cudaGetDevice(&device_id);
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "CUDA Device Count: " << device_count << "\n";
    std::cout << "CUDA Device ID: " << device_id << "\n";

    // Allocate memory for buffer that stores edge updates
    gpuErrchk(cudaMallocManaged(&edgeUpdates, num_updates * sizeof(vec_t)));

    // Allocate memory for buckets 
    gpuErrchk(cudaMallocManaged(&bucket_a, num_nodes * num_buckets * sizeof(vec_t)));
    gpuErrchk(cudaMallocManaged(&bucket_c, num_nodes * num_buckets * sizeof(vec_hash_t)));

    // Initialize all buffers with 0
    memset(edgeUpdates, 0, num_updates * sizeof(vec_t));
    memset(bucket_a, 0, num_nodes * num_buckets * sizeof(vec_t));
    memset(bucket_c, 0, num_nodes * num_buckets * sizeof(vec_hash_t));

    // Set shared memory as one sketch size
    maxBytes = num_buckets * sizeof(vec_t) + num_buckets * sizeof(vec_hash_t);
    cudaKernel.updateSharedMemory(maxBytes);
    std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

    std::cout << "Finished SKGPUSketchAlg's Initialization\n";
    std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
    std::cout << "SKGPUSketchAlg's Initialization Duration: " << init_time.count() << std::endl;
  };
  
  /**
   * Update all the sketches for a node, given a batch of updates.
   * @param thr_id         The id of the thread performing the update [0, num_threads)
   * @param src_vertex     The vertex where the edges originate.
   * @param dst_vertices   A vector of destinations.
   */
  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

  void launch_gpu_kernel();
  void apply_delta_sketch();
  uint64_t get_batch_count() { 
    uint64_t temp = batch_count;
    return temp; }
};
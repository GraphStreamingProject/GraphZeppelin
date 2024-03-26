#pragma once

#include "cc_sketch_alg.h"
#include "cuda_kernel.cuh"

struct CudaStream {
  cudaStream_t stream;
  int delta_applied;
  int src_vertex;
};

class CCGPUSketchAlg : public CCSketchAlg{
private:
  CudaUpdateParams* cudaUpdateParams;
  size_t sketchSeed;

  CudaKernel cudaKernel;

  // Variables from sketch
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;

  // Number of threads and thread blocks for CUDA kernel
  int num_device_threads = 1024;
  int num_device_blocks = 1;

  // Number of CPU's graph workers
  int num_host_threads;

  // Maximum number of edge updates in one batch
  int batch_size;

  // Number of CUDA Streams per graph worker
  int stream_multiplier = 4;

  // Vector for storing information for each CUDA Stream
  std::vector<CudaStream> streams;

public:
  CCGPUSketchAlg(node_id_t num_vertices, size_t num_updates, int num_threads, size_t seed, CCAlgConfiguration config = CCAlgConfiguration()) : CCSketchAlg(num_vertices, seed, config){ 

    // Start timer for initializing
    auto init_start = std::chrono::steady_clock::now();

    num_host_threads = num_threads;
    sketchSeed = seed;

    // Get variables from sketch
    num_samples = Sketch::calc_cc_samples(num_vertices, 1);
    num_columns = num_samples * Sketch::default_cols_per_sample;
    bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_vertices));
    num_buckets = num_columns * bkt_per_col + 1;

    std::cout << "num_samples: " << num_samples << "\n";
    std::cout << "num_buckets: " << num_buckets << "\n";
    std::cout << "num_columns: " << num_columns << "\n";
    std::cout << "bkt_per_col: " << bkt_per_col << "\n"; 

    batch_size = get_desired_updates_per_batch();

    // Create cudaUpdateParams
    gpuErrchk(cudaMallocManaged(&cudaUpdateParams, sizeof(CudaUpdateParams)));
    cudaUpdateParams = new CudaUpdateParams(num_vertices, num_updates, num_samples, num_buckets, num_columns, bkt_per_col, num_threads, batch_size, stream_multiplier);

    int device_id = cudaGetDevice(&device_id);
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "CUDA Device Count: " << device_count << "\n";
    std::cout << "CUDA Device ID: " << device_id << "\n";

    // Set maxBytes for GPU kernel's shared memory
    size_t maxBytes = num_buckets * sizeof(vec_t_cu) + num_buckets * sizeof(vec_hash_t);
    cudaKernel.updateSharedMemory(maxBytes);
    std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

    // Initialize CUDA Streams
    for (int i = 0; i < num_host_threads * stream_multiplier; i++) {
      cudaStream_t stream;

      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      streams.push_back({stream, 1, -1});
    }

    std::cout << "Finished CCGPUSketchAlg's Initialization\n";
    std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
    std::cout << "CCGPUSketchAlg's Initialization Duration: " << init_time.count() << std::endl;
  };
  //~CCGPUSketchAlg() : ~CCSketchAlg(){};
  
  /**
   * Update all the sketches for a node, given a batch of updates.
   * @param thr_id         The id of the thread performing the update [0, num_threads)
   * @param src_vertex     The vertex where the edges originate.
   * @param dst_vertices   A vector of destinations.
   */
  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

  // Update with the delta sketches that haven't been applied yet.
  void apply_flush_updates();
};
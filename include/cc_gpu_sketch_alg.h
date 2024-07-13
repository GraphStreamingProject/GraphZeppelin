#pragma once

#include <cmath>
#include "cc_sketch_alg.h"
#include "cuda_kernel.cuh"

struct CudaStream {
  cudaStream_t stream;
  int delta_applied;
  int src_vertex;
};

struct SketchParams {
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;
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
  int num_device_threads = 256;
  int num_device_blocks = 8;

  // Number of CPU's graph workers
  int num_host_threads;

  // Maximum number of edge updates in one batch
  int batch_size;

  // Number of CUDA Streams per graph worker
  int stream_multiplier;

  // Vector for storing information for each CUDA Stream
  std::vector<CudaStream> streams;
  std::vector<int> streams_offset;

public:
  CCGPUSketchAlg(node_id_t num_vertices, size_t num_updates, int num_threads, Bucket* buckets, size_t seed, SketchParams sketchParams, CCAlgConfiguration config = CCAlgConfiguration()) : CCSketchAlg(num_vertices, seed, buckets, config){ 

    // Start timer for initializing
    auto init_start = std::chrono::steady_clock::now();

    num_host_threads = num_threads;
    sketchSeed = seed;

    // Get variables from sketch
    num_samples = sketchParams.num_samples;
    num_columns = sketchParams.num_columns;
    bkt_per_col = sketchParams.bkt_per_col;
    num_buckets = sketchParams.num_buckets;

    batch_size = get_desired_updates_per_batch();
    std::cout << "Batch Size: " << batch_size << "\n";

    int device_id = cudaGetDevice(&device_id);
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    std::cout << "CUDA Device Count: " << device_count << "\n";
    std::cout << "CUDA Device ID: " << device_id << "\n";
    std::cout << "CUDA Device Number of SMs: " << deviceProp.multiProcessorCount << "\n"; 

    stream_multiplier = std::ceil(((double)deviceProp.multiProcessorCount / num_host_threads));
    std::cout << "Stream Multiplier: " << stream_multiplier << "\n";

    // Create cudaUpdateParams
    gpuErrchk(cudaMallocManaged(&cudaUpdateParams, sizeof(CudaUpdateParams)));
    cudaUpdateParams = new CudaUpdateParams(num_vertices, num_updates, buckets, num_samples, num_buckets, num_columns, bkt_per_col, num_threads, 0, batch_size, stream_multiplier, num_device_blocks);

    // Calculate the num_buckets assigned to the last thread block
    size_t num_last_tb_buckets = (cudaUpdateParams[0].num_tb_columns[num_device_blocks-1] * bkt_per_col) + 1;
    
    // Set maxBytes for GPU kernel's shared memory
    size_t maxBytes = (num_last_tb_buckets * sizeof(vec_t_cu)) + (num_last_tb_buckets * sizeof(vec_hash_t));
    cudaKernel.updateSharedMemory(maxBytes);
    std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

    // Initialize CUDA Streams
    for (int i = 0; i < num_host_threads * stream_multiplier; i++) {
      cudaStream_t stream;

      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      streams.push_back({stream, 1, -1});
    }

    for (int i = 0; i < num_host_threads; i++) {
      streams_offset.push_back(0);
    }

    // Prefetch sketches to GPU
    gpuErrchk(cudaMemPrefetchAsync(buckets, num_vertices * sketchParams.num_buckets * sizeof(Bucket), device_id));

    std::cout << "Finished CCGPUSketchAlg's Initialization\n";
    std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
    std::cout << "CCGPUSketchAlg's Initialization Duration: " << init_time.count() << std::endl;
  };

  /**
   * Update all the sketches for a node, given a batch of updates.
   * @param thr_id         The id of the thread performing the update [0, num_threads)
   * @param src_vertex     The vertex where the edges originate.
   * @param dst_vertices   A vector of destinations.
   */
  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

};
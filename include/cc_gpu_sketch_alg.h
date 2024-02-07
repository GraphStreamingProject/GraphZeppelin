#pragma once

#include "cc_sketch_alg.h"
#include "cuda_kernel.cuh"

class CCGPUSketchAlg : public CCSketchAlg{
private:
  CudaUpdateParams** cudaUpdateParams;
  //Sketch** sketches;
  long* sketchSeeds;

  CudaKernel cudaKernel;

  // Number of total buckets in one sketch
  size_t num_buckets;

  // Number of threads and thread blocks for CUDA kernel
  int num_device_threads;
  int num_device_blocks;

  // Number of CPU's graph workers
  int num_host_threads;

  // Maximum number of edge updates in one batch
  int batch_size;

  // Number of CUDA Streams per graph worker
  int stream_multiplier;

  // Flag to see if CCGPUSketchAlg has been configured
  bool isConfigured; 

  // Vectors for storing information for each CUDA Stream
  std::vector<cudaStream_t> streams;
  std::vector<int> streams_deltaApplied;
  std::vector<int> streams_src;
  std::vector<int> streams_num_graphs;

public:
  CCGPUSketchAlg(node_id_t num_vertices, size_t seed, CCAlgConfiguration config = CCAlgConfiguration()) : CCSketchAlg(num_vertices, seed, config){ isConfigured = false; };
  //~CCGPUSketchAlg() : ~CCSketchAlg(){};
  
  void configure(CudaUpdateParams** cudaUpdateParams, long* sketchSeeds, int num_host_threads);
  
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

  void updateKernelSharedMemory(size_t maxBytes) { cudaKernel.updateSharedMemory(maxBytes); };
  
  // getters
  inline Sketch** get_sketches() { return sketches; } 
};
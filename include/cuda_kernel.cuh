#pragma once
#include <atomic>
#include <iostream>
#include "bucket.h"
#include "types.h"
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

class CudaUpdateParams {
  public:
    // List of edge ids that thread will be responsble for updating
    vec_t *h_edgeUpdates, *d_edgeUpdates;

    Bucket* buckets;

    // Number of columns that each thread block will handle
    int *num_tb_columns;

    // Parameter for entire graph
    node_id_t num_nodes;
    size_t num_updates;
    
    // Parameter for each sketch (consistent with other sketches)
    size_t num_samples;
    size_t num_buckets;
    size_t num_columns;
    size_t bkt_per_col;

    int num_host_threads;
    int num_reader_threads;
    int num_device_blocks;
    int batch_size;
    int stream_multiplier; 

    int k;

    // Default Constructor of CudaUpdateParams
    CudaUpdateParams():h_edgeUpdates(nullptr), d_edgeUpdates(nullptr) {};
    
    CudaUpdateParams(node_id_t num_nodes, size_t num_updates, Bucket* buckets, int num_samples, size_t num_buckets, size_t num_columns, size_t bkt_per_col, int num_host_threads, int num_reader_threads, int batch_size, int stream_multiplier, int num_device_blocks, int k = 1):
      num_nodes(num_nodes), num_updates(num_updates), buckets(buckets), num_samples(num_samples), num_buckets(num_buckets), num_columns(num_columns), bkt_per_col(bkt_per_col), num_host_threads(num_host_threads), num_reader_threads(num_reader_threads), batch_size(batch_size), stream_multiplier(stream_multiplier), num_device_blocks(num_device_blocks), k(k) {
      
      // Allocate memory for buffer that stores edge updates

      gpuErrchk(cudaMallocHost(&h_edgeUpdates, stream_multiplier * num_host_threads * batch_size * sizeof(vec_t)));
      gpuErrchk(cudaMalloc(&d_edgeUpdates, stream_multiplier * num_host_threads * batch_size * sizeof(vec_t)));

      gpuErrchk(cudaMallocManaged(&num_tb_columns, num_device_blocks * sizeof(int)));

      for (int i = 0; i < num_device_blocks ; i++) {
        num_tb_columns[i] = num_columns / num_device_blocks;
      }

      // If num_columns doesn't get divided evenly
      size_t leftover_num_columns = num_columns - ((num_columns / num_device_blocks) * num_device_blocks);
      int k_id = num_device_blocks - 1;
      while (leftover_num_columns > 0) {
        num_tb_columns[k_id]++;
        k_id--;
        leftover_num_columns--;
      }
      std::cout << "Number of columns of each thread block: ";
      for (int i = 0; i < num_device_blocks ; i++) {
        std::cout << num_tb_columns[i] << ", ";
      }
      std::cout << "\n";

      for (size_t i = 0; i < stream_multiplier * num_host_threads * batch_size; i++) {
        h_edgeUpdates[i] = 0;
      }
    };
};

class CudaKernel {
  public:
    /*
    *   
    *   Sketch's Update Functions
    *
    */
    void sketchUpdate(int num_threads, int num_blocks, node_id_t src_vertex, cudaStream_t stream, vec_t *edgeUpdates, size_t update_size, CudaUpdateParams* cudaUpdateParams, long sketchSeed);
    void single_sketchUpdate(int num_threads, int num_blocks, vec_t* edgeUpdates, node_id_t* update_src, vec_t* update_sizes, vec_t* update_start_index, CudaUpdateParams* cudaUpdateParams, size_t sketchSeed);

    void updateSharedMemory(size_t maxBytes);

};
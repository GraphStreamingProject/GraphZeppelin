#pragma once

#include <map>
#include "mc_sketch_alg.h"
#include "cuda_kernel.cuh"

struct CudaStream {
  cudaStream_t stream;
  int delta_applied;
  int src_vertex;
  int num_graphs;
};

struct SketchParams {
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;  
};

struct Adjlist_Edge {
  std::pair<int, int> edge;
  int graph_id;
};

struct AdjList {
  // Id: source vertex
  // Content: vector of dst vertices
  std::map<node_id_t, std::vector<node_id_t>> list;
  size_t num_updates;
};

class MCGPUSketchAlg : public MCSketchAlg{
private:
  CudaUpdateParams** cudaUpdateParams;
  size_t sketchSeed;

  node_id_t num_nodes;
  int k;
  int sketches_factor;

  // Number of subgraphs
  int num_graphs;
  
  // Number of subgraphs in sketch representation
  int num_sketch_graphs;
  std::vector<size_t> sketch_num_edges;
  std::vector<std::mutex> sketch_mutexes;

  // Number of adj. list subgraphs
  int num_adj_graphs;
  int num_fixed_adj_graphs; // Number of subgraphs that will always be in adj. list

  // Threshold for switching a subgraph into sketch representation
  int num_req_edges;

  std::vector<std::vector<Adjlist_Edge>> worker_adjlist;
  std::vector<AdjList> adjlists;
  //std::vector<std::mutex> adjlists_mutexes;

  CudaKernel cudaKernel;

  // Variables from sketch
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;

  // Number of threads and thread blocks for CUDA kernel
  int num_device_threads;
  int num_device_blocks;

  // Number of CPU's graph workers
  int num_host_threads;

  // Maximum number of edge updates in one batch
  int batch_size;

  // Number of CUDA Streams per graph worker
  int stream_multiplier = 4;

  // Vector for storing information for each CUDA Stream
  std::vector<CudaStream> streams;

  // Indicate if trimming spanning forest, only apply sketch updates to one sketch subgraph
  bool trim_enabled;
  int trim_graph_id;

public:
  MCGPUSketchAlg(node_id_t num_vertices, size_t num_updates, int num_threads, size_t seed, SketchParams sketchParams, int _num_graphs, int _num_sketch_graphs, int _num_adj_graphs, int _num_fixed_adj_graphs, int _k, CCAlgConfiguration config = CCAlgConfiguration()) : MCSketchAlg(num_vertices, seed, _num_sketch_graphs, config){ 

    // Start timer for initializing
    auto init_start = std::chrono::steady_clock::now();

    sketchSeed = seed;
    num_nodes = num_vertices;
    k = _k;
    sketches_factor = config.get_sketches_factor();
    num_host_threads = num_threads;

    num_device_threads = 1024;
    num_device_blocks = k;

    num_graphs = _num_graphs;
    num_sketch_graphs = _num_sketch_graphs;
    num_adj_graphs = _num_adj_graphs;
    num_fixed_adj_graphs = _num_fixed_adj_graphs;

    // Extract sketchParams variables
    num_samples = sketchParams.num_samples;
    num_columns = sketchParams.num_columns;
    bkt_per_col = sketchParams.bkt_per_col;
    num_buckets = sketchParams.num_buckets;

    std::cout << "num_samples: " << num_samples << "\n";
    std::cout << "num_buckets: " << num_buckets << "\n";
    std::cout << "num_columns: " << num_columns << "\n";
    std::cout << "bkt_per_col: " << bkt_per_col << "\n";

    worker_adjlist.reserve(num_host_threads);
    
    // Initialize adj. list subgraphs 
    for (int i = 0; i < num_adj_graphs; i++) {
      AdjList adjlist;
      adjlist.num_updates = 0;
      adjlists.push_back(adjlist);   
    }

    for (int graph_id = 0; graph_id < num_sketch_graphs; graph_id++) {
      sketch_num_edges.push_back(0);
    }
    sketch_mutexes = std::vector<std::mutex>(num_sketch_graphs);

    // Initialize mutexes for adj. list subgraphs
    //adjlists_mutexes = std::vector<std::mutex>(num_adj_graphs);

    // Create a bigger batch size to apply edge updates when subgraph is turning into sketch representation
    batch_size = get_desired_updates_per_batch();
    
    // Create cudaUpdateParams
    gpuErrchk(cudaMallocManaged(&cudaUpdateParams, num_sketch_graphs * sizeof(CudaUpdateParams)));
    for (int i = 0; i < num_sketch_graphs; i++) {
      cudaUpdateParams[i] = new CudaUpdateParams(num_vertices, num_updates, num_samples, num_buckets, num_columns, bkt_per_col, num_threads, batch_size, stream_multiplier, k);
    }
    
    int device_id = cudaGetDevice(&device_id);
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "CUDA Device Count: " << device_count << "\n";
    std::cout << "CUDA Device ID: " << device_id << "\n";

    // Set maxBytes for GPU kernel's shared memory
    size_t maxBytes = ((num_buckets / k) + 1) * sizeof(vec_t_cu) + ((num_buckets / k) + 1) * sizeof(vec_hash_t);
    cudaKernel.updateSharedMemory(maxBytes);
    std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

    // Initialize CUDA Streams
    for (int i = 0; i < num_host_threads * stream_multiplier; i++) {
      cudaStream_t stream;

      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      streams.push_back({stream, 1, -1, -1});
    }

    trim_enabled = false;
    trim_graph_id = -1;

    std::cout << "Finished MCGPUSketchAlg's Initialization\n";
    std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
    std::cout << "MCGPUSketchAlg's Initialization Duration: " << init_time.count() << std::endl;
  };

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

  void print_subgraph_edges() {
    std::cout << "Number of inserted updates for each subgraph:\n";
    std::cout << "  Sketch Graphs:\n";
    for (int graph_id = 0; graph_id < num_sketch_graphs; graph_id++) {
      std::cout << "  S" << graph_id << ": " << sketch_num_edges[graph_id] << "\n";
    }
    std::cout << "  Adj. list Graphs:\n";
    for (int graph_id = 0; graph_id < num_adj_graphs; graph_id++) {
      std::cout << "  S" << graph_id << ": " << adjlists[graph_id].num_updates << "\n";
    }
  }

  void merge_adjlist();

  std::vector<Edge> get_adjlist_spanning_forests(int graph_id, int k);

  void set_trim_enbled(bool enabled, int graph_id) {
    trim_enabled = enabled;
    trim_graph_id = graph_id;

    if (trim_graph_id < 0 || trim_graph_id > num_sketch_graphs) {
      std::cout << "INVALID trim_graph_id: " << trim_graph_id << "\n";
    }
  }
};
#include <vector>
#include <graph.h>
#include <graph_worker.h>
#include <map>
#include <binary_graph_stream.h>
#include <cuda_graph.cuh>
#include <k_connected_graph.h>

// NOT WORKING PROPERLY: K-CONFIGURE NEEDS TO BE FIXED

constexpr size_t k = 2;
//constexpr size_t vert_multiple = 2;

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file, graph_workers, reader_threads" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string stream_file = argv[1];
  int num_threads = std::atoi(argv[2]);
  if (num_threads < 1) {
    std::cout << "ERROR: Invalid number of graph workers! Must be > 0." << std::endl;
    exit(EXIT_FAILURE);
  }
  int reader_threads = std::atoi(argv[3]);

  BinaryGraphStream_MT stream(stream_file, 1024*32);
  node_id_t num_nodes = stream.nodes();
  size_t num_updates  = stream.edges();
  std::cout << "Running process_stream with CUDA: " << std::endl;
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  if (k > num_nodes) {
    std::cerr << "k must be less than vertices!" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (num_nodes % k != 0) {
    std::cerr << "number of vertices must be a multiple of k!" << std::endl;
    exit(EXIT_FAILURE);
  }

  CudaGraph cudaGraph;

  auto config = GraphConfiguration().gutter_sys(CACHETREE).num_groups(num_threads);
  // Configuration is from cache_exp.cpp
  config.gutter_conf().page_factor(1)
              .buffer_exp(20)
              .fanout(64)
              .queue_factor(8)
              .num_flushers(2)
              .gutter_bytes(32 * 1024)
              .wq_batch_per_elm(8);
  /*GraphConfiguration conf;
  int num_threads = 12;
  conf.num_groups(num_threads);*/
  KConnectedGraph kGraph{num_nodes, config, &cudaGraph, k, reader_threads};

  Supernode** k_supernodes;
  k_supernodes = kGraph.getSupernodes();

  // Get variable from sample supernode
  int num_sketches = k_supernodes[0]->get_num_sktch();
  
  // Get variables from sample sketch
  size_t num_elems = k_supernodes[0]->get_sketch(0)->get_num_elems();
  size_t num_columns = k_supernodes[0]->get_sketch(0)->get_columns();
  size_t num_guesses = k_supernodes[0]->get_sketch(0)->get_num_guesses();

  std::cout << "num_sketches: " << num_sketches << "\n";
  std::cout << "num_elems: " << num_elems << "\n";
  std::cout << "num_columns: " << num_columns << "\n";
  std::cout << "num_guesses: " << num_guesses << "\n";

  // Start timer for initializing
  auto init_start = std::chrono::steady_clock::now();

  GutteringSystem *gts = kGraph.getGTS();
  int batch_size = gts->gutter_size() / sizeof(node_id_t);
  int stream_multiplier = 4;

  std::cout << "Batch_size: " << batch_size << "\n";
  
  CudaUpdateParams* cudaUpdateParams;
  gpuErrchk(cudaMallocManaged(&cudaUpdateParams, sizeof(CudaUpdateParams)));
  cudaUpdateParams = new CudaUpdateParams(num_nodes, num_updates, num_sketches, num_elems, num_columns, num_guesses, num_threads, batch_size, stream_multiplier);
  
  std::cout << "Initialized cudaUpdateParams\n";

  long* sketchSeeds;
  gpuErrchk(cudaMallocManaged(&sketchSeeds, num_nodes * num_sketches * k * sizeof(long)));

  // Initialize sketch seeds
  for (int i = 0; i < num_nodes * k; i++) {
    for (int j = 0; j < num_sketches; j++) {
      Sketch* sketch = k_supernodes[i]->get_sketch(j);
      sketchSeeds[(i * num_sketches) + j] = sketch->get_seed();
    }
  }

  int device_id = cudaGetDevice(&device_id);
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  std::cout << "CUDA Device Count: " << device_count << "\n";
  std::cout << "CUDA Device ID: " << device_id << "\n";

  int maxBytes = num_elems * num_sketches * sizeof(vec_t_cu) + num_elems * num_sketches * sizeof(vec_hash_t);
  cudaGraph.cudaKernel.kernelUpdateSharedMemory(maxBytes);
  std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

  // Prefetch memory to device  
  gpuErrchk(cudaMemPrefetchAsync(sketchSeeds, k * num_nodes * num_sketches * sizeof(long), device_id));

  cudaGraph.k_configure(&cudaUpdateParams, &k_supernodes, sketchSeeds, num_threads, k, 1);

  MT_StreamReader reader(stream);
  GraphUpdate upd;
  
  std::cout << "Finished initializing CUDA parameters\n";
  std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
  std::cout << "CUDA parameters init duration: " << init_time.count() << std::endl;

  // Start timer for kernel
  auto ins_start = std::chrono::steady_clock::now();

  // Call kernel code
  std::cout << "Update Kernel Starting...\n";

  // Do the edge updates
  std::vector<std::thread> threads;
  threads.reserve(reader_threads);
  auto task = [&](const int thr_id) {
    MT_StreamReader reader(stream);
    GraphUpdate upd;
    while(true) {
      upd = reader.get_edge();
      if (upd.type == BREAKPOINT) break;
      Edge &edge = upd.edge;

      gts->insert({edge.src, edge.dst}, thr_id);
      std::swap(edge.src, edge.dst);
      gts->insert({edge.src, edge.dst}, thr_id);
    }
  };

  // start inserters
  for (int t = 0; t < reader_threads; t++) {
    threads.emplace_back(task, t);
  }
  // wait for inserters to be done
  for (int t = 0; t < reader_threads; t++) {
    threads[t].join();
  }

  std::cout << "  Flush Starting...\n";
  
  auto flush_start = std::chrono::steady_clock::now();
  gts->force_flush();
  GraphWorker::pause_workers();
  cudaDeviceSynchronize();
  cudaGraph.k_applyFlushUpdates();
  auto flush_end = std::chrono::steady_clock::now();

  std::cout << "  Flushed Ended.\n";

  std::cout << "Update Kernel finished.\n";

  // End timer for kernel
  auto ins_end = std::chrono::steady_clock::now();

  // Update graph's num_updates value
  kGraph.num_updates += num_updates * 2;

  // Start timer for k_cc
  auto k_cc_start = std::chrono::steady_clock::now();

  std::cout << "Getting k = " << k << " spanning forests\n";
  std::vector<std::vector<Edge>> forests = kGraph.k_spanning_forests(k);

  /*for (int i = 0; i <= 2 * log2(num_nodes); i++) {
    std::cout << "Making Subgraph G_i: " << i << "\n";
    // Do something for i == 0
    if (i == 0) {
      continue;
    }

    // Get hash value product for every edge
    for (size_t edge_id = 0; edge_id < num_updates; edge_id++) {
      int edge_prod = 0;
      for (int j = 0; j <= i; j++) {
        // Uniform hash function
      }
      if (edge_prod == 1) { // Include current edge to the subgraph G_i

      }
    }

    std::cout << "Making Sketch H_i: " << i << "\n";
    
    // Make Sketch H_i with k-edge connectivity with G_i and k = O(e^-2 log n)

  }*/

  std::chrono::duration<double> insert_time = flush_end - ins_start;
  std::chrono::duration<double> flush_time = flush_end - flush_start;
  std::chrono::duration<double> k_cc_time = std::chrono::steady_clock::now() - k_cc_start;

  double num_seconds = insert_time.count();
  std::cout << "Total insertion time(sec): " << num_seconds << std::endl;
  std::cout << "Updates per second: " << stream.edges() / num_seconds << std::endl;
  std::cout << "Flush Gutters(sec): " << flush_time.count() << std::endl;
  std::cout << "K_CC(sec): " << k_cc_time.count() << std::endl;
}

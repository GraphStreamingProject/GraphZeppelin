#include <vector>
#include <graph.h>
#include <graph_worker.h>
#include <map>
#include <binary_graph_stream.h>
#include <cuda_graph.cuh>
#include <k_connected_graph.h>
#include <random>

constexpr size_t epsilon = 1;

void sample_edges(std::pair<Edge, SampleSketchRet> *query, std::vector<node_id_t> &reps) {
  bool except = false;
  std::exception_ptr err;
  #pragma omp parallel for default(none) shared(query, reps, except, err)
  for (node_id_t i = 0; i < reps.size(); ++i) { // NOLINT(modernize-loop-convert)
    // wrap in a try/catch because exiting through exception is undefined behavior in OMP
    try {
      query[reps[i]] = supernodes[(reps[i])]->sample();

    } catch (...) {
      except = true;
      err = std::current_exception();
    }
  }
  // Did one of our threads produce an exception?
  if (except) std::rethrow_exception(err);
}

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

  //size_t k = log2(num_nodes) / (epsilon * epsilon);
  size_t k = 1;

  // TODO: Check format of epsilon
  
  std::cout << "epsilon: " << epsilon << std::endl;
  std::cout << "k: " << k << std::endl;

  CudaGraph cudaGraph;

  auto config = GraphConfiguration().gutter_sys(CACHETREE).num_groups(num_threads);
  // Configuration is from cache_exp.cpp
  config.gutter_conf().page_factor(1)
              .buffer_exp(20)
              .fanout(64)
              .queue_factor(8)
              .num_flushers(2)
              .gutter_factor(1)
              .wq_batch_per_elm(8);
  KConnectedGraph graph{num_nodes, config, &cudaGraph, k, reader_threads};

  Supernode** supernodes;
  supernodes = graph.getSupernodes();

  // Get variable from sample supernode
  int num_sketches = supernodes[0]->get_num_sktch();

  // Get variables from sample sketch
  size_t num_elems = supernodes[0]->get_sketch(0)->get_num_elems();
  size_t num_columns = supernodes[0]->get_sketch(0)->get_columns();
  size_t num_guesses = supernodes[0]->get_sketch(0)->get_num_guesses();

  std::cout << "num_sketches: " << num_sketches << "\n";
  std::cout << "num_elems: " << num_elems << "\n";
  std::cout << "num_columns: " << num_columns << "\n";
  std::cout << "num_guesses: " << num_guesses << "\n";

  // Start timer for initializing
  auto init_start = std::chrono::steady_clock::now();

  GutteringSystem *gts = graph.getGTS();
  int batch_size = gts->gutter_size() / sizeof(node_id_t);
  int stream_multiplier = 4;

  std::cout << "Batch_size: " << batch_size << "\n";
  
  CudaUpdateParams* cudaUpdateParams;
  gpuErrchk(cudaMallocManaged(&cudaUpdateParams, sizeof(CudaUpdateParams)));
  cudaUpdateParams[0] = CudaUpdateParams(num_nodes, num_updates, num_sketches, num_elems, num_columns, num_guesses, num_threads, batch_size, stream_multiplier, k);

  std::cout << "Initialized cudaUpdateParams\n";

  long* sketchSeeds;
  gpuErrchk(cudaMallocManaged(&sketchSeeds, num_nodes * num_sketches * k * sizeof(long)));

  // Initialize sketch seeds
  for (int node_id = 0; node_id < num_nodes * k; node_id++) {
      for (int j = 0; j < num_sketches; j++) {
      Sketch* sketch = supernodes[node_id]->get_sketch(j);
      sketchSeeds[(node_id * num_sketches) + j] = sketch->get_seed();
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

  cudaGraph.k_configure(cudaUpdateParams, supernodes, sketchSeeds, num_threads, k);

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
  graph.num_updates += num_updates * 2;


  //for (int i = 1; i <= 2 * log2(num_nodes); i++) {
  for (int i = 1; i <= 1; i++) {
    std::cout << "Making Subgraph G_" << i << "\n";

    // Prepare for edge sampling process
    bool modified;
    size_t round = 0;
    std::pair<Edge, SampleSketchRet> *query = new std::pair<Edge, SampleSketchRet>[num_nodes];
    std::vector<node_id_t> reps(num_nodes);

    // Initialize reps
    for (node_id_t i = 0; i < num_nodes; ++i) {
      reps[i] = i;
    }

    // WIP 
    /*do {
      round++;
      modified = false;
      sample_edges(query, reps);

      std::vector<std::vector<node_id_t>> to_merge = to_merge_and_forest_edges(forest, query, reps, k_id);

      // make a copy if necessary
      if (first_round)
        backed_up = reps;

      k_merge_supernodes(copy_supernodes, reps, to_merge, first_round, k_id);

    } while (modified);*/

  }

  /*std::chrono::duration<double> insert_time = flush_end - ins_start;
  std::chrono::duration<double> flush_time = flush_end - flush_start;
  std::chrono::duration<double> k_cc_time = std::chrono::steady_clock::now() - k_cc_start;

  double num_seconds = insert_time.count();
  std::cout << "Total insertion time(sec): " << num_seconds << std::endl;
  std::cout << "Updates per second: " << stream.edges() / num_seconds << std::endl;
  std::cout << "Flush Gutters(sec): " << flush_time.count() << std::endl;
  std::cout << "K_CC(sec): " << k_cc_time.count() << std::endl;*/
}

#include <vector>
#include <graph.h>
#include <graph_worker.h>
#include <map>
#include <binary_graph_stream.h>
#include <cuda_graph.cuh>

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
  Graph g{num_nodes, config, &cudaGraph, reader_threads};

  Supernode** supernodes;
  supernodes = g.getSupernodes();

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

  GutteringSystem *gts = g.getGTS();
  int batch_size = gts->gutter_size() / sizeof(node_id_t);
  int stream_multiplier = 4;

  std::cout << "Batch_size: " << batch_size << "\n";
  
  CudaUpdateParams* cudaUpdateParams;
  gpuErrchk(cudaMallocManaged(&cudaUpdateParams, sizeof(CudaUpdateParams)));
  cudaUpdateParams[0] = CudaUpdateParams(num_nodes, num_updates, num_sketches, num_elems, num_columns, num_guesses, num_threads, batch_size, stream_multiplier);

  CudaSketch* cudaSketches;
  gpuErrchk(cudaMallocManaged(&cudaSketches, num_nodes * num_sketches * sizeof(CudaSketch)));

  long* sketchSeeds;
  gpuErrchk(cudaMallocManaged(&sketchSeeds, num_nodes * num_sketches * sizeof(long)));

  // Allocate space for all buckets
  vec_t* d_bucket_a;
  vec_hash_t* d_bucket_c;
  gpuErrchk(cudaMallocManaged(&d_bucket_a, (num_nodes * num_sketches * num_elems * sizeof(vec_t))));
  gpuErrchk(cudaMallocManaged(&d_bucket_c, (num_nodes * num_sketches * num_elems * sizeof(vec_hash_t))));

  for (size_t i = 0; i < (num_nodes * num_sketches * num_elems); i++) {
    d_bucket_a[i] = 0;
    d_bucket_c[i] = 0;
  }

  // Create a vector of cuda supernodes and sketches
  for (int i = 0; i < num_nodes; i++) {
    for (int j = 0; j < num_sketches; j++) {
      Sketch* sketch = supernodes[i]->get_sketch(j);

      int bucket_id = (i * num_sketches * num_elems) + (j * num_elems);
      vec_t* bucket_a = &d_bucket_a[bucket_id];
      vec_hash_t* bucket_c = &d_bucket_c[bucket_id];

      // Rewrite sketch's bucket_a and bucket_c memory location
      sketch->set_bucket_a(bucket_a);
      sketch->set_bucket_c(bucket_c);

      CudaSketch cudaSketch(bucket_a, bucket_c, sketch->get_seed());
      cudaSketches[(i * num_sketches) + j] = cudaSketch;
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
  gpuErrchk(cudaMemPrefetchAsync(cudaSketches, num_nodes * num_sketches * sizeof(CudaSketch), device_id));
  gpuErrchk(cudaMemPrefetchAsync(sketchSeeds, num_nodes * num_sketches * sizeof(long), device_id));
  gpuErrchk(cudaMemPrefetchAsync(d_bucket_a, num_nodes * num_sketches * num_elems * sizeof(vec_t), device_id));
  gpuErrchk(cudaMemPrefetchAsync(d_bucket_c, num_nodes * num_sketches * num_elems * sizeof(vec_hash_t), device_id));

  cudaGraph.configure(cudaUpdateParams, cudaSketches, sketchSeeds, num_threads);
  
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

  auto flush_start = std::chrono::steady_clock::now();
  gts->force_flush();
  GraphWorker::pause_workers();
  cudaDeviceSynchronize();
  auto flush_end = std::chrono::steady_clock::now();

  std::cout << "Update Kernel finished.\n";

  // End timer for kernel
  auto ins_end = std::chrono::steady_clock::now();

  /*for (int i = 0; i < cudaGraph.loop_times.size(); i++) {
    std::cout << "Stream #" << i << ": ";
    double total_loop_time = 0;
    for (int j = 0; j < cudaGraph.loop_times[i].size(); j++) {
      total_loop_time += cudaGraph.loop_times[i][j];
    }
    std::cout << total_loop_time << "\n";
  }*/
  
  // Update graph's num_updates value
  g.num_updates += num_updates * 2;

  // Start timer for cc
  auto cc_start = std::chrono::steady_clock::now();
  auto CC_num = g.connected_components().size();

  std::chrono::duration<double> insert_time = flush_end - ins_start;
  std::chrono::duration<double> cc_time = std::chrono::steady_clock::now() - cc_start;
  std::chrono::duration<double> flush_time = flush_end - flush_start;
  std::chrono::duration<double> cc_alg_time = g.cc_alg_end - g.cc_alg_start;

  double num_seconds = insert_time.count();
  std::cout << "Total insertion time(sec):    " << num_seconds << std::endl;
  std::cout << "Updates per second:           " << stream.edges() / num_seconds << std::endl;
  std::cout << "Total CC query latency:       " << cc_time.count() << std::endl;
  std::cout << "  Flush Gutters(sec):           " << flush_time.count() << std::endl;
  std::cout << "  Boruvka's Algorithm(sec):     " << cc_alg_time.count() << std::endl;
  std::cout << "Connected Components:         " << CC_num << std::endl;
}

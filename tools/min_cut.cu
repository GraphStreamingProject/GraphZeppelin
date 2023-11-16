#include <vector>
#include <map>
#include <random>
#include <fstream>
#include <string>
#include <sstream>

#include <graph.h>
#include <graph_worker.h>
#include <binary_graph_stream.h>
#include <cuda_graph.cuh>
#include <mincut_graph.h>

constexpr size_t epsilon = 1;

static uint64_t fast_atoi(const std::string& str, size_t* line_ptr) {
    uint64_t x = 0;

    while (str[*line_ptr] >= '0' && str[*line_ptr] <= '9') {
        x = (x * 10) + (str[*line_ptr] - '0');
        ++(*line_ptr);
    }
    ++(*line_ptr);
    return x;
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
  int k = 2;

  // TODO: Check format of epsilon
  std::cout << "epsilon: " << epsilon << std::endl;
  std::cout << "k: " << k << std::endl;

  //int num_graphs = 1 + (int)(2 * log2(num_nodes));
  int num_graphs = 2;
  std::cout << "Total num_graphs: " << num_graphs << "\n";

  auto config = GraphConfiguration().gutter_sys(CACHETREE).num_groups(num_threads);
  // Configuration is from cache_exp.cpp
  config.gutter_conf().page_factor(1)
              .buffer_exp(20)
              .fanout(64)
              .queue_factor(8)
              .num_flushers(2)
              .gutter_bytes(32 * 1024)
              .wq_batch_per_elm(8);

  CudaGraph cudaGraph;
  MinCutGraph* graphs[num_graphs];

  for (int i = 0; i < num_graphs; i++) {
    if (i == 0) {
      graphs[i] = new MinCutGraph{num_nodes, config, &cudaGraph, k, reader_threads};
    }
    else {
      // Reuse the GTS made from graphs[0]
      graphs[i] = new MinCutGraph{num_nodes, config, graphs[0]->getGTS(), &cudaGraph, k, reader_threads};
    }
  
  }

  Supernode** supernodes;
  supernodes = graphs[0]->getSupernodes();

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

  GutteringSystem *gts = graphs[0]->getGTS();
  int batch_size = gts->gutter_size() / sizeof(node_id_t);
  int stream_multiplier = 4;

  std::cout << "Batch_size: " << batch_size << "\n";
  
  CudaUpdateParams* cudaUpdateParams;
  gpuErrchk(cudaMallocManaged(&cudaUpdateParams, sizeof(CudaUpdateParams) * num_graphs));
  for (int i = 0; i < num_graphs; i++) {
    cudaUpdateParams[i] = CudaUpdateParams(num_nodes, num_updates, num_sketches, num_elems, num_columns, num_guesses, num_threads, batch_size, stream_multiplier, k);
  }
  
  std::cout << "Initialized cudaUpdateParams\n";

  long* sketchSeeds;
  gpuErrchk(cudaMallocManaged(&sketchSeeds, num_graphs * num_nodes * num_sketches * k * sizeof(long)));

  // Initialize sketch seeds
  std::vector<Supernode**> all_supernodes;
  
  for (int graph_id = 0; graph_id < num_graphs; graph_id++) {
    Supernode** graph_supernodes;
    graph_supernodes = graphs[graph_id]->getSupernodes();
    for (int node_id = 0; node_id < num_nodes * k; node_id++) {
      for (int j = 0; j < num_sketches; j++) {
        Sketch* sketch = graph_supernodes[node_id]->get_sketch(j);
        sketchSeeds[(graph_id * num_nodes * num_sketches * k) + (node_id * num_sketches) + j] = sketch->get_seed();
      }
    }
    all_supernodes.push_back(graph_supernodes);
  }

  int maxBytes = num_elems * num_sketches * sizeof(vec_t_cu) + num_elems * num_sketches * sizeof(vec_hash_t);
  cudaGraph.cudaKernel.kernelUpdateSharedMemory(maxBytes);

  int device_id = cudaGetDevice(&device_id);
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_id);
  std::cout << "CUDA Device Count: " << device_count << "\n";
  std::cout << "CUDA Device ID: " << device_id << "\n";
  std::cout << "Maximum Shared Memory per block: " << props.sharedMemPerBlock << " bytes\n";
  std::cout << "Maximum Shared Memory per block Optin: " << props.sharedMemPerBlockOptin << " bytes\n";

  cudaGraph.k_configure(cudaUpdateParams, all_supernodes.data(), sketchSeeds, num_threads, k, num_graphs);

  std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

  // Prefetch memory to device  
  gpuErrchk(cudaMemPrefetchAsync(sketchSeeds, k * num_nodes * num_sketches * sizeof(long), device_id));

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

  std::cout << "  Applying Flush Updates...\n";

  cudaGraph.k_applyFlushUpdates();

  for(int i = 0; i < num_graphs; i++) {
    graphs[i]->num_updates += cudaUpdateParams[i].num_inserted_updates;
  }
  
  auto flush_end = std::chrono::steady_clock::now();

  std::cout << "  Flushed Ended.\n";

  std::cout << "Update Kernel finished.\n";

  // End timer for kernel
  auto ins_end = std::chrono::steady_clock::now();

  std::cout << "Number of inserted updates for each subgraph:\n";
  for (int i = 0; i < num_graphs; i++) {
    std::cout << "  Subgraph G_" << i << ": " << graphs[i]->num_updates << "\n";
  }
  
  /*std::cout << "\n";
  std::cout << "Getting CC for G_0:\n";
  auto CC_num = graphs[0]->connected_components().size();
  std::cout << "Connected Components: " << CC_num << std::endl;*/

  std::cout << "Getting k = " << k << " spanning forests\n";

  // Get spanning forests then create a METIS format file
  std::cout << "Generating Certificates...\n";
  for (int i = 0; i < num_graphs; i++) {
    std::cout << "  Subgraph G_" << i << ":\n";
    std::vector<std::vector<Edge>> forests = graphs[0]->k_spanning_forests(k); // Note: Getting k spanning forests for only the first graph 

    int sampled_edges = 0;
    for (int k_id = 0; k_id < k; k_id++) {
      std::cout << "    Size of forests #" << k_id << ": " << forests[k_id].size() << "\n"; 
      sampled_edges += forests[k_id].size();
    }
    std::cout << "  Total sampled edges: " << sampled_edges << "\n";

    std::string file_name = "certificates" + std::to_string(i) + ".metis";
    std::ofstream cert (file_name);

    // Read edges then categorize them based on src node
    int sampled_num_nodes = 0;
    int sampled_num_edges = 0;
    std::map<node_id_t, std::vector<node_id_t>> nodes_list;
    for (auto forest : forests) {
      for (auto e : forest) {
        if (nodes_list.find(e.src) == nodes_list.end()) { // Has not been inserted yet
          nodes_list[e.src] = std::vector<node_id_t>();
          sampled_num_nodes++;
        }
        nodes_list[e.src].push_back(e.dst);

        if (nodes_list.find(e.dst) == nodes_list.end()) { // Has not been inserted yet
          nodes_list[e.dst] = std::vector<node_id_t>();
          sampled_num_nodes++;
        }
        nodes_list[e.dst].push_back(e.src); 

        sampled_num_edges++;
      }
    }

    // Write sampled num_nodes and num_edges to file
    cert << sampled_num_nodes << " " << sampled_num_edges << " 0" << "\n";
    for (auto it = nodes_list.begin(); it != nodes_list.end(); it++) {
      for (size_t neighbor = 0; neighbor < it->second.size() - 1; neighbor++) {
        cert << it->second[neighbor] + 1 << " ";
      }
      cert << it->second[it->second.size() - 1] + 1 << "\n";
    }
    cert.close();
  }

  std::cout << "Getting minimum cut of certificates...\n";
  for (int i = 0; i < num_graphs; i++) {
    std::string file_name = "certificates" + std::to_string(i) + ".metis";
    std::string command = "../VieCut/build/mincut_parallel " + file_name + " exact";
    std::system(command.data());
  }

  for (int i = 0; i < num_graphs; i++) {
    delete graphs[i];
  }

  std::chrono::duration<double> insert_time = flush_end - ins_start;
  std::chrono::duration<double> flush_time = flush_end - flush_start;
  //std::chrono::duration<double> k_cc_time = std::chrono::steady_clock::now() - k_cc_start;

  double num_seconds = insert_time.count();
  std::cout << "Total insertion time(sec): " << num_seconds << std::endl;
  std::cout << "Updates per second: " << stream.edges() / num_seconds << std::endl;
  std::cout << "Flush Gutters(sec): " << flush_time.count() << std::endl;
  //std::cout << "K_CC(sec): " << k_cc_time.count() << std::endl;
}

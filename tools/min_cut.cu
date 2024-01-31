#include <vector>
#include <map>
#include <random>
#include <fstream>
#include <string>
#include <cmath>

#include <graph.h>
#include <graph_worker.h>
#include <binary_graph_stream.h>
#include <cuda_graph.cuh>
#include <mincut_graph.h>

constexpr double epsilon = 0.2;

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

  int k = log2(num_nodes) / (epsilon * epsilon);

  std::cout << "epsilon: " << epsilon << std::endl;
  std::cout << "k: " << k << std::endl;

  int num_graphs = 1 + (int)(2 * log2(num_nodes));
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

  double total_graphs_sketch_size = graphs[0]->getTotalSketchSize() * num_graphs;

  if(total_graphs_sketch_size > 1000000000) {
    std::cout << "Total Graphs Sketch Memory Size: " << total_graphs_sketch_size / 1000000000 << " GB\n";
  }
  else if(total_graphs_sketch_size > 1000000) {
    std::cout << "Total Graphs Sketch Memory Size: " << total_graphs_sketch_size / 1000000 << " MB\n";
  }
  else {
    std::cout << "Total Graphs Sketch Memory Size: " << total_graphs_sketch_size / 1000 << " KB\n";
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
  
  CudaUpdateParams** cudaUpdateParams;
  gpuErrchk(cudaMallocManaged(&cudaUpdateParams, sizeof(CudaUpdateParams*) * num_graphs));
  for (int i = 0; i < num_graphs; i++) {
    cudaUpdateParams[i] = new CudaUpdateParams(num_nodes, num_updates, num_sketches, num_elems, num_columns, num_guesses, num_threads, batch_size, stream_multiplier, k);
  }

  std::cout << "Initialized cudaUpdateParams\n";

  long* sketchSeeds;
  size_t sketch_width = Sketch::column_gen(Sketch::get_failure_factor());
  gpuErrchk(cudaMallocManaged(&sketchSeeds, num_graphs * num_nodes * num_sketches * k * sizeof(long)));

  // Initialize sketch seeds
  std::vector<Supernode**> all_supernodes;
  
  for (int graph_id = 0; graph_id < num_graphs; graph_id++) {
    Supernode** graph_supernodes;
    graph_supernodes = graphs[graph_id]->getSupernodes();
    for (int node_id = 0; node_id < num_nodes; node_id++) {
      for (int k_id = 0; k_id < k; k_id++) {
        for (int j = 0; j < num_sketches; j++) {
          Sketch* sketch = graph_supernodes[(node_id * k) + k_id]->get_sketch(j);
          sketchSeeds[(graph_id * num_nodes * num_sketches * k) + (node_id * num_sketches * k) + (k_id * num_sketches) + j] = sketch->get_seed();
        }
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
  gpuErrchk(cudaMemPrefetchAsync(sketchSeeds, num_graphs * num_nodes * num_sketches * k * sizeof(long), device_id));

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
    graphs[i]->num_updates += cudaUpdateParams[i]->num_inserted_updates;
  }
  
  auto flush_end = std::chrono::steady_clock::now();

  std::cout << "  Flushed Ended.\n";

  std::cout << "Update Kernel finished.\n";

  // End timer for kernel
  auto ins_end = std::chrono::steady_clock::now();

  std::cout << "Number of inserted updates for each subgraph:\n";
  int num_zero_graphs = 0;
  for (int i = 0; i < num_graphs; i++) {
    std::cout << "  Subgraph G_" << i << ": " << graphs[i]->num_updates << "\n";
    if (graphs[i]->num_updates == 0) num_zero_graphs++;
  }
  
  std::cout << "Getting k = " << k << " spanning forests\n";

  std::chrono::duration<double> spanning_forests_time = std::chrono::nanoseconds::zero();
  std::chrono::duration<double> cert_write_time = std::chrono::nanoseconds::zero();
  std::chrono::duration<double> viecut_time = std::chrono::nanoseconds::zero();

  // Get spanning forests then create a METIS format file
  std::cout << "Generating Certificates...\n";
  int num_sampled_zero_graphs = 0;
  for (int i = 0; i < num_graphs - num_zero_graphs; i++) {
    std::cout << "Subgraph G_" << i << ":\n";

    auto spanning_forests_start = std::chrono::steady_clock::now();
    std::vector<std::vector<Edge>> forests = graphs[i]->k_spanning_forests(k, i);
    spanning_forests_time += std::chrono::steady_clock::now() - spanning_forests_start;

    int sampled_edges = 0;
    for (int k_id = 0; k_id < k; k_id++) {
      sampled_edges += forests[k_id].size();
    }
    std::cout << "  Total sampled edges: " << sampled_edges << "\n";

    if(sampled_edges == 0) {
      num_sampled_zero_graphs++;
      continue;
    }

    auto cert_write_start = std::chrono::steady_clock::now();
    
    std::string file_name = "certificates" + std::to_string(i) + ".metis";
    std::ofstream cert (file_name);

    // Read edges then categorize them based on src node
    int sampled_num_nodes = 0;
    int sampled_num_edges = 0;
    node_id_t current_node_id = 1;
    int num_self_edges = 0;

    std::map<node_id_t, std::vector<node_id_t>> nodes_list;
    std::map<node_id_t, node_id_t> simplified_node_ids;

    for (auto forest : forests) {
      for (auto e : forest) {
        if (simplified_node_ids.find(e.src) == simplified_node_ids.end()) { // Has not been inserted yet
          simplified_node_ids[e.src] = current_node_id;
          nodes_list[current_node_id] = std::vector<node_id_t>();

          sampled_num_nodes++;
          current_node_id++;
        }

        if (simplified_node_ids.find(e.dst) == simplified_node_ids.end()) {
          simplified_node_ids[e.dst] = current_node_id;
          nodes_list[current_node_id] = std::vector<node_id_t>();

          sampled_num_nodes++;
          current_node_id++;
        }
      
        node_id_t simplified_node1 = simplified_node_ids[e.src];
        node_id_t simplified_node2 = simplified_node_ids[e.dst];
        
        if (simplified_node1 == simplified_node2) {
          num_self_edges++;
        }
        
        nodes_list[simplified_node1].push_back(simplified_node2);
        nodes_list[simplified_node2].push_back(simplified_node1);

        sampled_num_edges++;
      }
    }

    if (num_self_edges > 0) {
      std::cout << "WARNING: There are self edges! " << num_self_edges << "\n";
    }

    // Write sampled num_nodes and num_edges to file
    cert << sampled_num_nodes << " " << sampled_num_edges << " 0" << "\n";

    for (auto it : nodes_list) {
      for (size_t neighbor = 0; neighbor < it.second.size(); neighbor++) {
        if (it.second[neighbor] == it.first) {
          continue;
        }
        cert << (it.second[neighbor]) << " ";
      }
      cert << "\n";  
    }
    cert.close();
    cert_write_time += std::chrono::steady_clock::now() - cert_write_start;
  }


  std::cout << "Getting minimum cut of certificates...\n";
  auto viecut_start = std::chrono::steady_clock::now();
  std::vector<int> mincut_values;
  for (int i = 0; i < num_graphs - num_zero_graphs - num_sampled_zero_graphs; i++) {
    std::string file_name = "certificates" + std::to_string(i) + ".metis";
    std::string output_name = "mincut" + std::to_string(i) + ".txt";
    std::string command = "../VieCut/build/mincut_parallel " + file_name + " exact >" + output_name; // Run VieCut and store the output
    std::system(command.data());

    std::string line;
    std::ifstream output_file(output_name);
    if(output_file.is_open()) {
      std::getline(output_file, line); // Skip first line
      std::getline(output_file, line);

      int start_index = line.find("cut=");
      int end_index = line.find(" n=");

      if (start_index != std::string::npos || end_index != std::string::npos ) {
        int cut = stoi(line.substr((start_index + 4), ((end_index) - (start_index + 4)))); 
        std::cout << "  G_" << i << ": " << cut << "\n";
        mincut_values.push_back(cut);
      }
      else {
        std::cout << "Error: Couldn't find 'cut=' or 'n=' in the output file\n";
      }
      output_file.close();
    }
    else {
      std::cout << "Error: Couldn't find file name: " << output_name << "!\n";
    }
  }
  viecut_time += std::chrono::steady_clock::now() - viecut_start;

  // Go through min cut values of each subgraph and find the minimum cut of subgraph that is smaller than k
  for (int i = 0; i < mincut_values.size(); i++) {
    if(mincut_values[i] < k) {
      std::cout << "Mincut value found! i: " << i << " mincut: " << mincut_values[i] << "\n";
      std::cout << "Final mincut value: " << (mincut_values[i] * (pow(2, i))) << "\n";
      break;
     }
  }

  for (int i = 0; i < num_graphs; i++) {
    delete graphs[i];
  }

  std::chrono::duration<double> insert_time = flush_end - ins_start;
  std::chrono::duration<double> flush_time = flush_end - flush_start;

  double num_seconds = insert_time.count();
  std::cout << "Total insertion time(sec): " << num_seconds << std::endl;
  std::cout << "Updates per second: " << stream.edges() / num_seconds << std::endl;
  std::cout << "Flush Gutters(sec): " << flush_time.count() << std::endl;
  std::cout << "Spanning Forests Time(sec): " << spanning_forests_time.count() << std::endl;

  double total_sampling_forests_time = 0;
  double total_trimming_forests_time = 0;

  for (int i = 0; i < num_graphs; i++) {
    total_sampling_forests_time += graphs[i]->sampling_forests_time.count();
    total_trimming_forests_time += graphs[i]->trimming_forests_time.count();
  }

  std::cout << "  Total Sampling Forests Time(sec): " << total_sampling_forests_time << std::endl;
  std::cout << "  Total Trimming Forests Time(sec): " << total_trimming_forests_time << std::endl;

  std::cout << "Certificate Writing Time(sec): " << cert_write_time.count() << std::endl;
  std::cout << "VieCut Program Time(sec): " << viecut_time.count() << std::endl;

}

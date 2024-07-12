#include <fstream>
#include <vector>
#include <graph_sketch_driver.h>
#include <mc_gpu_sketch_alg.h>
#include <binary_file_stream.h>
#include <thread>
#include <sys/resource.h> // for rusage
#include <cuda_kernel.cuh>

static bool cert_clean_up = false;
static bool shutdown = false;
constexpr double epsilon = 0.75;

static double get_max_mem_used() {
  struct rusage data;
  getrusage(RUSAGE_SELF, &data);
  return (double) data.ru_maxrss / 1024.0;
}

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

/*
 * Function which is run in a seperate thread and will query
 * the graph for the number of updates it has processed
 * @param total       the total number of edge updates
 * @param g           the graph object to query
 * @param start_time  the time that we started stream ingestion
 */
void track_insertions(uint64_t total, GraphSketchDriver<MCGPUSketchAlg> *driver,
                      std::chrono::steady_clock::time_point start_time) {
  total = total * 2; // we insert 2 edge updates per edge

  printf("Insertions\n");
  printf("Progress:                    | 0%%\r"); fflush(stdout);
  std::chrono::steady_clock::time_point start = start_time;
  std::chrono::steady_clock::time_point prev  = start_time;
  uint64_t prev_updates = 0;

  while(true) {
    sleep(1);
    uint64_t updates = driver->get_total_updates();
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_diff = now - start;
    std::chrono::duration<double> cur_diff   = now - prev;

    // calculate the insertion rate
    uint64_t upd_delta = updates - prev_updates;
    // divide insertions per second by 2 because each edge is split into two updates
    // we care about edges per second not about stream updates
    size_t ins_per_sec = (((double)(upd_delta)) / cur_diff.count()) / 2;

    if (updates >= total || shutdown)
      break;

    // display the progress
    int progress = updates / (total * .05);
    printf("Progress:%s%s", std::string(progress, '=').c_str(), std::string(20 - progress, ' ').c_str());
    printf("| %i%% -- %lu per second\r", progress * 5, ins_per_sec); fflush(stdout);
  }

  printf("Progress:====================| Done                             \n");
  return;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file, graph_workers, reader_threads" << std::endl;
    exit(EXIT_FAILURE);
  }

  shutdown = false;
  std::string stream_file = argv[1];
  int num_threads = std::atoi(argv[2]);
  if (num_threads < 1) {
    std::cout << "ERROR: Invalid number of graph workers! Must be > 0." << std::endl;
    exit(EXIT_FAILURE);
  }
  int reader_threads = std::atoi(argv[3]);

  BinaryFileStream stream(stream_file);
  node_id_t num_nodes = stream.vertices();
  size_t num_updates  = stream.edges();
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  int k = log2(num_nodes) / (epsilon * epsilon);
  int reduced_k = (k / log2(num_nodes)) * 2; 

  std::cout << "epsilon: " << epsilon << std::endl;
  std::cout << "k: " << k << std::endl;
  std::cout << "reduced_k: " << reduced_k << std::endl;

  int num_graphs = 1 + (int)(2 * log2(num_nodes));
  std::cout << "Total num_graphs: " << num_graphs << "\n";

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
  auto mc_config = CCAlgConfiguration().batch_factor(3);

  // Get variables from sketch
  // (1) num_samples (2) num_columns (3) bkt_per_col (4) num_buckets
  SketchParams sketchParams;
  sketchParams.num_samples = Sketch::calc_cc_samples(num_nodes, reduced_k);
  sketchParams.num_columns = sketchParams.num_samples * Sketch::default_cols_per_sample;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;

  // Total bytes of sketching datastructure of one subgraph
  int w = 4; // 4 bytes when num_nodes < 2^32
  double sketch_bytes = 4 * w * num_nodes * ((2 * log2(num_nodes)) + 2) * ((reduced_k * log2(num_nodes))/(1 - log2(1.2)));
  double adjlist_edge_bytes = 8;

  std::cout << "Total bytes of sketching data structure of one subgraph: " << sketch_bytes / 1000000000 << "GB\n";

  // Calculate number of minimum adj. list subgraph
  size_t num_edges_complete = (size_t(num_nodes) * (size_t(num_nodes) - 1)) / 2;
  int num_adj_graphs = 0;
  int num_sketch_graphs = 0;
  int min_adj_graphs = 0;
  int max_sketch_graphs = 0;

  for (int i = 0; i < num_graphs; i++) {
    // Calculate estimated memory for current subgraph
    size_t num_est_edges = num_edges_complete / (1 << i);
    double adjlist_bytes = adjlist_edge_bytes * num_est_edges;

    if (adjlist_bytes < sketch_bytes) {
      min_adj_graphs++;
    }
    else {
      max_sketch_graphs++;
    }
  }

  // # of adj. list graphs in the beginning
  num_adj_graphs = num_graphs;

  // Total number of estimated edges of minimum number of adj. list graphs
  size_t num_est_edges_adj_graphs = (2 * num_edges_complete) / (1 << (max_sketch_graphs));
  double total_adjlist_bytes = adjlist_edge_bytes * num_est_edges_adj_graphs;
  double total_sketch_bytes = sketch_bytes * max_sketch_graphs;

  std::cout << "Number of adj. list graphs: " << num_adj_graphs << "\n";
  std::cout << "Number of sketch graphs: " << num_sketch_graphs << "\n";
  std::cout << "  If complete graph with current num_nodes..." << "\n";
  std::cout << "    Minimum number of adj. list graphs: " << min_adj_graphs << "\n";
  std::cout << "    Maximum number of sketch graphs: " << max_sketch_graphs << "\n";
  std::cout << "    Total minimum memory required for minimum number of adj. list graphs: " << total_adjlist_bytes / 1000000000 << "GB\n";
  std::cout << "    Total minimum memory required for maximum number of sketch graphs: " << total_sketch_bytes / 1000000000 << "GB\n";
  std::cout << "    Total minimum memory required for current num_nodes: " << (total_adjlist_bytes + total_sketch_bytes) / 1000000000 << "GB\n";

  // Reconfigure sketches_factor based on reduced_k
  mc_config.sketches_factor(reduced_k);

  MCGPUSketchAlg mc_gpu_alg{num_nodes, num_updates, num_threads, reader_threads, get_seed(), sketchParams, num_graphs, min_adj_graphs, max_sketch_graphs, reduced_k, sketch_bytes, adjlist_edge_bytes, mc_config};
  GraphSketchDriver<MCGPUSketchAlg> driver{&mc_gpu_alg, &stream, driver_config, reader_threads};

  auto ins_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions, num_updates, &driver, ins_start);

  driver.process_stream_until(END_OF_STREAM);

  auto flush_start = std::chrono::steady_clock::now();
  driver.prep_query(KSPANNINGFORESTS);
  cudaDeviceSynchronize();
  mc_gpu_alg.apply_flush_updates();
  mc_gpu_alg.convert_adj_to_sketch();
  // Re-measure flush_end to include time taken for applying delta sketches from flushing
  auto flush_end = std::chrono::steady_clock::now();

  shutdown = true;
  querier.join();

  // Display number of inserted updates to every subgraphs
  mc_gpu_alg.print_subgraph_edges();

  std::chrono::duration<double> sampling_forests_time = std::chrono::nanoseconds::zero();
  std::chrono::duration<double> trim_reading_time = std::chrono::nanoseconds::zero();
  std::chrono::duration<double> trim_flushing_time = std::chrono::nanoseconds::zero();
  std::chrono::duration<double> cert_write_time = std::chrono::nanoseconds::zero();
  std::chrono::duration<double> viecut_time = std::chrono::nanoseconds::zero();

  std::cout << "After Insertion:\n";
  num_adj_graphs = mc_gpu_alg.get_num_adj_graphs();
  num_sketch_graphs = mc_gpu_alg.get_num_sketch_graphs();
  std::cout << "Number of adj. list graphs: " << num_adj_graphs << "\n";
  std::cout << "Number of sketch graphs: " << num_sketch_graphs << "\n";

  // Get spanning forests then create a METIS format file
  std::cout << "Generating Certificates...\n";
  int num_sampled_zero_graphs = 0;
  for (int graph_id = 0; graph_id < num_graphs; graph_id++) {
    std::vector<Edge> spanningForests;
    std::set<Edge> edges;
    
    if (graph_id >= num_sketch_graphs) { // Get Spanning forests from adj list
      std::cout << "S" << graph_id << " (Adj. list):\n";
      auto sampling_forest_start = std::chrono::steady_clock::now();
      spanningForests = mc_gpu_alg.get_adjlist_spanning_forests(graph_id, k);
      sampling_forests_time += std::chrono::steady_clock::now() - sampling_forest_start;
    } 
    else { // Get Spanning forests from sketch subgraph
      std::cout << "S" << graph_id << " (Sketch):\n";
      mc_gpu_alg.set_trim_enbled(true, graph_id); // When trimming, only apply sketch updates to current subgraph
      for (int k_id = 0; k_id < k; k_id++) {
        std::cout << "  Getting spanning forest " << k_id << "\n";

        // Get spanning forest k_id
        auto sampling_forest_start = std::chrono::steady_clock::now();
        SpanningForest spanningForest = mc_gpu_alg.get_k_spanning_forest(graph_id);
        sampling_forests_time += std::chrono::steady_clock::now() - sampling_forest_start;

        // Insert sampled edges from spanningForest to spanningForests
        for (auto edge : spanningForest.get_edges()) {
          spanningForests.push_back(edge);
        }
        
        // Trim spanning forest
        auto trim_reading_start = std::chrono::steady_clock::now();
        driver.trim_spanning_forest(spanningForest.get_edges());
        trim_reading_time += std::chrono::steady_clock::now() - trim_reading_start;

        // Flush sketch updates
        auto trim_flushing_start = std::chrono::steady_clock::now();
        driver.prep_query(KSPANNINGFORESTS);
        trim_flushing_time += std::chrono::steady_clock::now() - trim_flushing_start;

        // Verify sampled edges from spanning forest
        for (auto& edge : spanningForest.get_edges()) {
          if (edges.count(edge) == 0) {
            edges.insert(edge);
          }
          else {
            std::cerr << "ERROR: duplicate edge in forests! {" << edge.src << "," << edge.dst << "}\n";
            exit(EXIT_FAILURE);
          }
        }
      }

    }

    std::cout << "  Number of edges in spanning forests: " << spanningForests.size() << "\n";

    if (spanningForests.size() == 0) { // No need to make certificate for empty spanning forest
      num_sampled_zero_graphs++;
      continue;
    }

    auto cert_write_start = std::chrono::steady_clock::now();

    std::string file_name = "certificates" + std::to_string(graph_id) + ".metis";
    std::ofstream cert (file_name);

    // Read edges then categorize them based on src node
    int sampled_num_nodes = 0;
    int sampled_num_edges = 0;
    node_id_t current_node_id = 1;
    int num_self_edges = 0;

    std::map<node_id_t, std::vector<node_id_t>> nodes_list;
    std::map<node_id_t, node_id_t> simplified_node_ids;

    for (auto& edge : spanningForests) {
      if (simplified_node_ids.find(edge.src) == simplified_node_ids.end()) { // Has not been inserted yet
        simplified_node_ids[edge.src] = current_node_id;
        nodes_list[current_node_id] = std::vector<node_id_t>();

        sampled_num_nodes++;
        current_node_id++;
      }

      if (simplified_node_ids.find(edge.dst) == simplified_node_ids.end()) {
        simplified_node_ids[edge.dst] = current_node_id;
        nodes_list[current_node_id] = std::vector<node_id_t>();

        sampled_num_nodes++;
        current_node_id++;
      }
      
      node_id_t simplified_node1 = simplified_node_ids[edge.src];
      node_id_t simplified_node2 = simplified_node_ids[edge.dst];
      
      if (simplified_node1 == simplified_node2) {
        num_self_edges++;
      }
      
      nodes_list[simplified_node1].push_back(simplified_node2);
      nodes_list[simplified_node2].push_back(simplified_node1);

      sampled_num_edges++;
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
  for (int graph_id = 0; graph_id < num_graphs - num_sampled_zero_graphs; graph_id++) {
    std::string file_name = "certificates" + std::to_string(graph_id) + ".metis";
    std::string output_name = "mincut" + std::to_string(graph_id) + ".txt";
    std::string command = "_deps/viecut-build/mincut_parallel " + file_name + " exact >" + output_name; // Run VieCut and store the output
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
        if (graph_id >= num_sketch_graphs) {
          std::cout << "  S" << graph_id << " (Adj. list): " << cut << "\n";
        }
        else {
          std::cout << "  S" << graph_id << " (Sketch): " << cut << "\n";
        }
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

  std::chrono::duration<double> insert_time = flush_end - ins_start;
  std::chrono::duration<double> flush_time = flush_end - flush_start;

  double num_seconds = insert_time.count();
  std::cout << "Insertion time(sec): " << num_seconds << std::endl;
  std::cout << "  Updates per second: " << stream.edges() / num_seconds << std::endl;
  std::cout << "  Flush Gutters(sec): " << flush_time.count() << std::endl;
  std::cout << "K-Connectivity: (Sketch Subgraphs)" << std::endl;
  std::cout << "  Sampling Forests Time(sec): " << sampling_forests_time.count() << std::endl;
  std::cout << "  Trimming Forests Reading Time(sec): " << trim_reading_time.count() << std::endl;
  std::cout << "  Trimming Forests Flushing Time(sec): " << trim_flushing_time.count() << std::endl;
  std::cout << "Certificate Writing Time(sec): " << cert_write_time.count() << std::endl;
  std::cout << "VieCut Program Time(sec): " << viecut_time.count() << std::endl;
  std::cout << "Maximum Memory Usage(MiB): " << get_max_mem_used() << std::endl;

  // If enabled, remove all certificate and VieCut output files
  if(cert_clean_up) {
    for (int graph_id = 0; graph_id < num_graphs - num_sampled_zero_graphs; graph_id++) {
      std::remove(("certificates" + std::to_string(graph_id) + ".metis").c_str());
      std::remove(("mincut" + std::to_string(graph_id) + ".txt").c_str());
    }
  }

}
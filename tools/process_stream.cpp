#include <binary_file_stream.h>
#include <cc_sketch_alg.h>
#include <KEdgeConnect.h>
#include <graph_sketch_driver.h>
#include <sys/resource.h>  // for rusage
#include <test/mat_graph_verifier.h>

#include <thread>

// TODO: make num_edge_connect an input argument;
// TODO: add verifier to see whether the output spanning forest is a valid forest
//        and whether the graph is indeed disconnected after removing k spanning forests
// TODO: idea to fix the verifier problem: make the KEdgeConnect a separate file for the class
//        then, write unit tests based on the tests in the file graph_test.cpp. 


static bool shutdown = false;

class TwoEdgeConnect {
 public:
  const node_id_t num_nodes;
  CCSketchAlg cc_alg_1;
  CCSketchAlg cc_alg_2;

  explicit TwoEdgeConnect(node_id_t num_nodes, const CCAlgConfiguration &config_1,
                          const CCAlgConfiguration &config_2)
      : num_nodes(num_nodes), cc_alg_1(num_nodes, config_1), cc_alg_2(num_nodes, config_2) {}

  void allocate_worker_memory(size_t num_workers) {
    cc_alg_1.allocate_worker_memory(num_workers);
    cc_alg_2.allocate_worker_memory(num_workers);
  }

  size_t get_desired_updates_per_batch() {
    // I don't want to return double because the updates are sent to both
    return cc_alg_1.get_desired_updates_per_batch();
  }

  node_id_t get_num_vertices() { return num_nodes; }

  void pre_insert(GraphUpdate upd, node_id_t thr_id) {
    cc_alg_1.pre_insert(upd, thr_id);
    cc_alg_2.pre_insert(upd, thr_id);
  }

  void apply_update_batch(size_t thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices) {
    cc_alg_1.apply_update_batch(thr_id, src_vertex, dst_vertices);
    cc_alg_2.apply_update_batch(thr_id, src_vertex, dst_vertices);
  }

  bool has_cached_query() { return cc_alg_1.has_cached_query() && cc_alg_2.has_cached_query(); }

  void print_configuration() { cc_alg_1.print_configuration(); }

  void query() {
    std::vector<std::pair<node_id_t, std::vector<node_id_t>>> forest =
        cc_alg_1.calc_spanning_forest();

    GraphUpdate temp_edge;

    temp_edge.type = DELETE;

    std::cout << "SPANNING FOREST 1" << std::endl;
    for (unsigned int j = 0; j < forest.size(); j++) {
      std::cout << forest[j].first << ":";
      for (auto dst : forest[j].second) {
        std::cout << " " << dst;
        temp_edge.edge.src = forest[j].first;
        temp_edge.edge.dst = dst;
        cc_alg_2.update(temp_edge);
      }
      std::cout << std::endl;
    }

    std::vector<std::pair<node_id_t, std::vector<node_id_t>>> forest2 =
        cc_alg_2.calc_spanning_forest();

    std::cout << "SPANNING FOREST 2" << std::endl;
    for (unsigned int j = 0; j < forest.size(); j++) {
        std::cout << forest[j].first << ":";
        for (auto dst: forest[j].second) {
            std::cout << " " << dst;
        }
        std::cout << std::endl;
    }

    // TODO: reinsert into alg 2?
  }
};

static double get_max_mem_used() {
  struct rusage data;
  getrusage(RUSAGE_SELF, &data);
  return (double)data.ru_maxrss / 1024.0;
}

/*
 * Function which is run in a seperate thread and will query
 * the graph for the number of updates it has processed
 * @param total       the total number of edge updates
 * @param g           the graph object to query
 * @param start_time  the time that we started stream ingestion
 */
template <typename DriverType>

void track_insertions(uint64_t total, DriverType *driver,
                      std::chrono::steady_clock::time_point start_time) {
  total = total * 2;  // we insert 2 edge updates per edge

  printf("Insertions\n");
  printf("Progress:                    | 0%%\r");
  fflush(stdout);
  std::chrono::steady_clock::time_point start = start_time;
  std::chrono::steady_clock::time_point prev = start_time;
  uint64_t prev_updates = 0;

  while (true) {
    sleep(1);
    uint64_t updates = driver->get_total_updates();
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_diff = now - start;
    std::chrono::duration<double> cur_diff = now - prev;

    // calculate the insertion rate
    uint64_t upd_delta = updates - prev_updates;
    // divide insertions per second by 2 because each edge is split into two updates
    // we care about edges per second not about stream updates
    size_t ins_per_sec = (((double)(upd_delta)) / cur_diff.count()) / 2;

    if (updates >= total || shutdown) break;

    // display the progress
    int progress = updates / (total * .05);
    printf("Progress:%s%s", std::string(progress, '=').c_str(),
           std::string(20 - progress, ' ').c_str());
    printf("| %i%% -- %lu per second\r", progress * 5, ins_per_sec);
    fflush(stdout);
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
  size_t reader_threads = std::atol(argv[3]);
  unsigned int num_edge_connect = 5;

  BinaryFileStream stream(stream_file);
  BinaryFileStream stream_ref(stream_file);
  node_id_t num_nodes = stream.vertices();
  size_t num_updates = stream.edges();
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
  std::vector<CCAlgConfiguration> config_vec;

  for (unsigned int i=0;i<num_edge_connect;i++){
      config_vec.push_back(CCAlgConfiguration().batch_factor(1));
  }

  KEdgeConnect k_edge_alg{num_nodes, num_edge_connect, config_vec};

  GraphSketchDriver<KEdgeConnect> driver{&k_edge_alg, &stream, driver_config, reader_threads};

  auto ins_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions<GraphSketchDriver<KEdgeConnect>>, num_updates, &driver, ins_start);

  driver.process_stream_until(END_OF_STREAM);

  auto cc_start = std::chrono::steady_clock::now();
  driver.prep_query();
  k_edge_alg.query();


  size_t m = stream_ref.edges();
  // test the edges in the spanning forest are in the original graph
  std::vector<std::vector<bool>> adj_mat(num_nodes);
  for (node_id_t i = 0; i < num_nodes; i++) adj_mat[i] = std::vector<bool>(num_nodes - i);
  while (m--) {
    GraphStreamUpdate upd;
    stream_ref.get_update_buffer(&upd, 1);
    node_id_t src = upd.edge.src;
    node_id_t dst = upd.edge.dst;
    if (src > dst) std::swap(src, dst);
    dst = dst - src;
    adj_mat[src][dst] = !adj_mat[src][dst];
  }

  MatGraphVerifier kEdgeVerifier(num_nodes, adj_mat);

  std::vector<std::vector<bool>> test_adj_mat(num_nodes);
  test_adj_mat =  kEdgeVerifier.extract_adj_matrix();

  std::cout << "The adjacency list of the vertex 0 in the beginning: " << std::endl;
  for(unsigned int i=0;i<num_nodes; i++){
      std::cout << test_adj_mat[0][i] << "  ";
  }
  std::cout << std::endl;

  Edge temp_edge;
  std::vector<std::pair<node_id_t, std::vector<node_id_t>>> temp_forest;
  for(unsigned int i=0;i<num_edge_connect;i++) {
      temp_forest = k_edge_alg.forests_collection[i];
      for (unsigned int j = 0; j < temp_forest.size(); j++) {
            for (auto dst: temp_forest[j].second) {
                temp_edge.src = temp_forest[j].first;
                temp_edge.dst = dst;
                kEdgeVerifier.verify_edge(temp_edge);
                kEdgeVerifier.edge_update(temp_edge.src, temp_edge.dst);
            }
      }
      test_adj_mat =  kEdgeVerifier.extract_adj_matrix();
      std::cout << "The adjacency list of the vertex 0 after " << (i+1) << "spanning forests: " << std::endl;
      for(unsigned int l=0; l<num_nodes; l++) {
          std::cout << test_adj_mat[0][l] << "  ";
      }
      std::cout << std::endl;
  }

  unsigned long CC_nums[num_edge_connect];
  for(unsigned int i=0;i<num_edge_connect;i++){
      CC_nums[i]= k_edge_alg.cc_alg[i]->connected_components().size();
  }

  std::chrono::duration<double> insert_time = driver.flush_end - ins_start;
  std::chrono::duration<double> cc_time = std::chrono::steady_clock::now() - cc_start;
  std::chrono::duration<double> flush_time = driver.flush_end - driver.flush_start;
  std::chrono::duration<double> cc_alg_time =
          k_edge_alg.cc_alg[num_edge_connect-1]->cc_alg_end - k_edge_alg.cc_alg[0]->cc_alg_start;

  shutdown = true;
  querier.join();

  double num_seconds = insert_time.count();
  std::cout << "Total insertion time(sec):    " << num_seconds << std::endl;
  std::cout << "Updates per second:           " << stream.edges() / num_seconds << std::endl;
  std::cout << "Total CC query latency:       " << cc_time.count() << std::endl;
  std::cout << "  Flush Gutters(sec):           " << flush_time.count() << std::endl;
  std::cout << "  Boruvka's Algorithm(sec):     " << cc_alg_time.count() << std::endl;
  for(unsigned int i=0;i<num_edge_connect;i++){
      std::cout << "Number of connected Component in :         " << i+1 << " is " << CC_nums[i] << std::endl;
  }
  std::cout << "Maximum Memory Usage(MiB):    " << get_max_mem_used() << std::endl;
}

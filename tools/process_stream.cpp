#include <graph.h>
#include <binary_graph_stream.h>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file, graph_workers" << std::endl;
  }

  std::string stream_file = argv[1];
  int num_threads = std::atoi(argv[2]);
  if (num_threads < 1) {
    std::cout << "ERROR: Invalid number of graph workers! Must be > 0." << std::endl;
  }

  BinaryGraphStream stream(stream_file, 1024*32);
  node_id_t num_nodes = stream.nodes();
  size_t num_updates  = stream.edges();
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  auto config = GraphConfiguration().gutter_sys(STANDALONE).num_groups(num_threads);
  config.gutter_conf().gutter_factor(-4);
  Graph g{num_nodes, config};

  auto ins_start = std::chrono::steady_clock::now();
  for (size_t e = 0; e < num_updates; e++)
    g.update(stream.get_edge());

  for(int i = 0; i < g.getSupernodes()[2]->get_num_sktch(); i++) {
    for(size_t j = 0; j < g.getSupernodes()[2]->get_sketch(i)->get_num_elems(); j++) {
      std::cout << g.getSupernodes()[2]->get_sketch(i)->get_bucket_a()[j] << " ";
    }
  }
  std::cout << "\n";

  auto cc_start = std::chrono::steady_clock::now();
  auto CC_num = g.connected_components().size();
  std::chrono::duration<double> insert_time = g.flush_end - ins_start;
  std::chrono::duration<double> cc_time = std::chrono::steady_clock::now() - cc_start;
  double num_seconds = insert_time.count();
  std::cout << "Total insertion time was: " << num_seconds << std::endl;
  std::cout << "Insertion rate was:       " << stream.edges() / num_seconds << std::endl;
  std::cout << "CC query latency:         " << cc_time.count() << std::endl;
  std::cout << "Connected Components:     " << CC_num << std::endl;
}

#include <binary_graph_stream.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Incorrect Number of Arguments!" << std::endl;
    std::cout << "Arguments: stream_file" << std::endl;
    exit(EXIT_FAILURE);
  }
  
  BinaryGraphStream stream(argv[1], 1024*1024);
  node_id_t nodes = stream.nodes();
  size_t edges    = stream.edges();

  std::cout << "Attempting to validate stream " << argv[1] << std::endl;
  std::cout << "Number of nodes   = " << nodes << std::endl;
  std::cout << "Number of updates = " << edges << std::endl;
  std::cout << ", , ,"  << std::endl;

  // validate the src and dst of each node in the stream and ensure there are enough of them
  bool err = false;
  for (size_t e = 0; e < edges; e++) {
    GraphUpdate upd;
    try {
      upd = stream.get_edge();
    } catch (...) {
      std::cerr << "ERROR: Could not get edge at index: " << e << std::endl;
      err = true;
      std::rethrow_exception(std::current_exception());
      break;
    }
    Edge edge = upd.edge;
    UpdateType u = upd.type;
    if (edge.src >= nodes || edge.dst >= nodes || (u != INSERT && u != DELETE) || edge.src == edge.dst) {
      std::cerr << "ERROR: edge idx:" << e << "=(" << edge.src << "," << edge.dst << "), " << u << std::endl;
      err = true;
    }
    if (e % 1000000000 == 0 && e != 0) std::cout << e << std::endl; 
  }

  if (!err) std::cout << "Stream validated!" << std::endl;
  if (err) std::cout << "Stream invalid!" << std::endl;
}


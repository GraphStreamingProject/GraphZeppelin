#include <iostream>
#include <fstream>
#include <graph_zeppelin_common.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Incorrect number of arguments. "
                 "Expected one but got " << argc-1 << std::endl;
    std::cout << "Arguments are: text_stream" << std::endl;
  }

  std::ifstream txt_file(argv[1]);
  std::ofstream out_file("binary_stream.data", std::ios_base::binary);

  node_id_t num_nodes;
  edge_id_t num_edges;

  txt_file >> num_nodes >> num_edges;
  out_file.write((char *) &num_nodes, sizeof(num_nodes));
  out_file.write((char *) &num_edges, sizeof(num_edges));

  node_id_t src;
  node_id_t dst;

  while(num_edges--) {
    bool u = false;
    txt_file >> src >> dst;
    out_file.write((char *) &u, sizeof(u));
    out_file.write((char *) &src, sizeof(src));
    out_file.write((char *) &dst, sizeof(dst));
  }
}
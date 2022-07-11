#include <iostream>
#include <fstream>
#include <vector>
#include <graph_zeppelin_common.h>

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    std::cout << "Incorrect number of arguments. "
                 "Expected at either one or two but got " << argc-1 << std::endl;
    std::cout << "Arguments are: text_stream [update_type]" << std::endl;
    std::cout << "text_stream is the file to parse into binary format" << std::endl;
    std::cout << "update_type is a flag. If present then stream indicates insertions vs deletions" << std::endl;
  }

  std::ifstream txt_file(argv[1]);
  std::ofstream out_file("binary_stream.data", std::ios_base::binary);

  bool update_type = false;
  if (argc == 3) {
    if (std::string(argv[2]) == "update_type")
      update_type = true;
    else {
      std::cerr << "Did not recognize second argument! Expected 'update_type'";
      return EXIT_FAILURE;
    }
  }

  node_id_t num_nodes;
  edge_id_t num_edges;

  txt_file >> num_nodes >> num_edges;
  out_file.write((char *) &num_nodes, sizeof(num_nodes));
  out_file.write((char *) &num_edges, sizeof(num_edges));

  std::vector<std::vector<bool>> adj_mat(num_nodes);
  for (node_id_t i = 0; i < num_nodes; ++i)
    adj_mat[i] = std::vector<bool>(num_nodes - i);

  bool u;
  node_id_t src;
  node_id_t dst;

  while(num_edges--) {
    u = false;
    if (update_type)
      txt_file >> u >> src >> dst;
    else
      txt_file >> src >> dst;

    if (src > dst) {
      if (u != adj_mat[dst][src - dst]) {
        std::cout << "WARNING: update " << u << " " << src << " " << dst;
        std::cout << " is double insert or delete before insert. Correcting." << std::endl;
      }
      u = adj_mat[dst][src - dst];
      adj_mat[dst][src - dst] = !adj_mat[dst][src - dst];
    } else {
      if (u != adj_mat[src][dst - src]) {
        std::cout << "WARNING: update " << u << " " << src << " " << dst;
        std::cout << " is double insert or delete before insert. Correcting." << std::endl;
      }
      u = adj_mat[src][dst - src];
      adj_mat[src][dst - src] = !adj_mat[src][dst - src];
    }

    out_file.write((char *) &u, sizeof(u));
    out_file.write((char *) &src, sizeof(src));
    out_file.write((char *) &dst, sizeof(dst));
  }
}

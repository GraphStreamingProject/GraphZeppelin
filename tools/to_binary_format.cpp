#include <iostream>
#include <fstream>
#include <vector>
#include <errno.h>
#include <string.h>
#include <graph_zeppelin_common.h>

int main(int argc, char **argv) {
  if (argc < 3 || argc > 5) {
    std::cout << "Incorrect number of arguments. "
                 "Expected [2-4] but got " << argc-1 << std::endl;
    std::cout << "Arguments are: ascii_stream out_file_name [--update_type] [--verbose]" << std::endl;
    std::cout << "ascii_stream:  The file to parse into binary format" << std::endl;
		std::cout << "out_file_name: Where the binary stream will be written" << std::endl;
    std::cout << "--update_type: If present then ascii stream indicates insertions vs deletions" << std::endl;
		std::cout << "--silent:      If present then no warnings are printed when stream corrections are made" << std::endl;
		exit(EXIT_FAILURE);
  }

  std::ifstream txt_file(argv[1]);
	if (!txt_file) {
		std::cerr << "ERROR: could not open input file!" << std::endl;
		exit(EXIT_FAILURE);
	}
  std::ofstream out_file(argv[2], std::ios_base::binary | std::ios_base::out);
	if (!out_file) {
		std::cerr << "ERROR: could not open output file! " << argv[2] << ": " << strerror(errno) << std::endl;
		exit(EXIT_FAILURE);
	}

  bool update_type = false;
	bool silent = false;
  for (int i = 3; i < argc; i++) {
    if (std::string(argv[i]) == "--update_type")
      update_type = true;
		else if (std::string(argv[i]) == "--silent") {
			silent = true;
		}
    else {
      std::cerr << "Did not recognize argument: " << argv[i] << " Expected '--update_type' or '--silent'";
      return EXIT_FAILURE;
    }
  }

  node_id_t num_nodes;
  edge_id_t num_edges;

  txt_file >> num_nodes >> num_edges;
	
	std::cout << "Parsed ascii stream header. . ." << std::endl;
	std::cout << "Number of nodes:   " << num_nodes << std::endl;
	std::cout << "Number of updates: " << num_edges << std::endl;	
	if (update_type)
		std::cout << "Assuming that update format is: upd_type src dst" << std::endl;
	else
		std::cout << "Assuming that update format is: src dst" << std::endl;
	

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
      if (!silent && u != adj_mat[dst][src - dst]) {
        std::cout << "WARNING: update " << u << " " << src << " " << dst;
        std::cout << " is double insert or delete before insert. Correcting." << std::endl;
      }
      u = adj_mat[dst][src - dst];
      adj_mat[dst][src - dst] = !adj_mat[dst][src - dst];
    } else {
      if (!silent && u != adj_mat[src][dst - src]) {
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


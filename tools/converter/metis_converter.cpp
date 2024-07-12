#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>

int main(int argc, char** argv) {

  if (argc != 2) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: graph_file" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string file_name = argv[1];
  std::ifstream graph_file(file_name);
  std::string line;
  
  size_t num_nodes = 0;
  size_t num_edges = 0;
  size_t current_node_id = 1;
  size_t num_self_edges = 0;

  std::map<size_t, size_t> simplified_node_ids;
  std::map<size_t, std::vector<size_t>> nodes_list;
  
  std::cout << "Input Graph File: " << file_name << "\n";
  std::cout << "Reading Input Graph File...\n";

  if(graph_file.is_open()) {
    while(std::getline(graph_file, line)) {
      std::istringstream iss(line);
      std::string token;

      size_t node1, node2;

      std::getline(iss, token, ' '); // Make sure to check delimiter
      node1 = std::stoi(token);

      std::getline(iss, token, ' '); // Make sure to check delimiter
      node2 = std::stoi(token);

      if (simplified_node_ids.find(node1) == simplified_node_ids.end()) {
        simplified_node_ids[node1] = current_node_id;
        nodes_list[current_node_id] = std::vector<size_t>();

        num_nodes++;
        current_node_id++;
      }

      if (simplified_node_ids.find(node2) == simplified_node_ids.end()) {
        simplified_node_ids[node2] = current_node_id;
        nodes_list[current_node_id] = std::vector<size_t>();

        num_nodes++;
        current_node_id++;
      }
      
      size_t simplified_node1 = simplified_node_ids[node1];
      size_t simplified_node2 = simplified_node_ids[node2];
      
      if (simplified_node1 == simplified_node2) {
        num_self_edges++;
      }
      
      nodes_list[simplified_node1].push_back(simplified_node2);
      nodes_list[simplified_node2].push_back(simplified_node1);

      num_edges++;
    }
  }
  else {
    std::cout << "Error: Couldn't find file name: " << file_name << "!\n";
  }

  std::cout << "  Num Nodes: " << num_nodes << "\n";
  std::cout << "  Num Input Edges: " << num_edges << "\n";
  std::cout << "  Num Self Edges: " << num_self_edges << "\n";

  num_edges -= num_self_edges;

  std::cout << "  Num Final Edges: " << num_edges << "\n";
  std::cout << "Finished Reading Input Graph File...\n";

  graph_file.close();

  std::string metis_name = file_name + ".metis";
  std::ofstream metis_file(metis_name);

  std::cout << "Writing METIS file...\n";

  metis_file << num_nodes << " " << num_edges << " 0" << "\n";

  for (auto it : nodes_list) {
    for (size_t neighbor = 0; neighbor < it.second.size(); neighbor++) {
      if (it.second[neighbor] == it.first) {
        continue;
      }
      metis_file << (it.second[neighbor]) << " ";
      
    }
    metis_file << "\n";  
  }
  
  metis_file.close();

  std::cout << "Finished Writing METIS file...\n";
}
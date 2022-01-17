#include <stdexcept>
#include "../include/util.h"
#include "../include/graph_worker.h"
#include "../include/graph.h"

const char *config_file = "streaming.conf";

typedef uint32_t ul;
typedef uint64_t ull;

const ull ULLMAX = std::numeric_limits<ull>::max();
const uint8_t num_bits = sizeof(node_id_t) * 8;

unsigned long long int double_to_ull(double d, double epsilon) {
  return (unsigned long long) (d + epsilon);
}

ull nondirectional_non_self_edge_pairing_fn(ul i, ul j) {
  // swap i,j if necessary
  if (i > j) {
    std::swap(i,j);
  }
  return ((ull)i << num_bits) | j;
}

std::pair<ul, ul> inv_nondir_non_self_edge_pairing_fn(ull idx) {
  ul j = idx & 0xFFFFFFFF;
  ul i = idx >> num_bits;
  return {i, j};
}

std::pair<bool, std::string> configure_system() {
  bool use_guttertree = true;
  std::string pre = "./GUTTREEDATA/";
  int num_groups = 1;
  int group_size = 1;
  std::string line;
  std::ifstream conf(config_file);
  if (conf.is_open()) {
    while(getline(conf, line)) {
      if (line[0] == '#' || line[0] == '\n') continue;
      if(line.substr(0, line.find('=')) == "buffering_system") {
        string buf_str = line.substr(line.find('=') + 1);
        if (buf_str == "standalone") {
          use_guttertree = false;
        } else if (buf_str != "tree") {
          printf("WARNING: string %s is not a valid option for " 
                "buffering. Defaulting to GutterTree.\n", buf_str.c_str());
        }
        printf("Using %s for buffering.\n", use_guttertree? "GutterTree" : "StandAloneGutters");
      }
      if(line.substr(0, line.find('=')) == "path_prefix" && use_guttertree) {
        pre = line.substr(line.find('=') + 1);
        printf("GutterTree path_prefix = %s\n", pre.c_str());
      }
      if(line.substr(0, line.find('=')) == "num_groups") {
        num_groups = std::stoi(line.substr(line.find('=') + 1));
        if (num_groups < 1) { 
          printf("num_groups=%i is out of bounds. Defaulting to 1.\n", num_groups);
          num_groups = 1; 
        }
        printf("Number of groups = %i\n", num_groups);
      }
      if(line.substr(0, line.find('=')) == "group_size") {
        group_size = std::stoi(line.substr(line.find('=') + 1));
        if (group_size < 1) { 
          printf("group_size=%i is out of bounds. Defaulting to 1.\n", group_size);
          group_size = 1; 
        }
        printf("Size of groups = %i\n", group_size);
      }
    }
  } else {
    printf("WARNING: Could not open thread configuration file! Using default values.\n");
  }

  GraphWorker::set_config(num_groups, group_size);
  return {use_guttertree, pre};
}

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

std::tuple<bool, bool, std::string> configure_system() {
  bool use_guttertree = true;
  std::string dir = "./";
  int num_groups = 1;
  int group_size = 1;
  bool backup_in_mem = false;
  std::string line;
  std::ifstream conf(config_file);
  if (conf.is_open()) {
    while(getline(conf, line)) {
      if (line[0] == '#' || line[0] == '\n') continue;
      if(line.substr(0, line.find('=')) == "buffering_system") {
        std::string buf_str = line.substr(line.find('=') + 1);
        if (buf_str == "standalone") {
          use_guttertree = false;
        } else if (buf_str != "tree") {
          printf("WARNING: string %s is not a valid option for " 
                "buffering. Defaulting to GutterTree.\n", buf_str.c_str());
        }
      }
      if(line.substr(0, line.find('=')) == "disk_dir") {
        dir = line.substr(line.find('=') + 1) + "/";
      }
      if(line.substr(0, line.find('=')) == "backup_in_mem") {
        std::string flag = line.substr(line.find('=') + 1);
	if (flag == "ON")
          backup_in_mem = true;
	else if (flag == "OFF")
          backup_in_mem = false;
	else
          printf("WARNING: string %s is not a valid option for backup_in_mem"
                 "Defaulting to OFF.\n", flag.c_str());
      }
      if(line.substr(0, line.find('=')) == "num_groups") {
        num_groups = std::stoi(line.substr(line.find('=') + 1));
        if (num_groups < 1) { 
          printf("num_groups=%i is out of bounds. Defaulting to 1.\n", num_groups);
          num_groups = 1; 
        }
      }
      if(line.substr(0, line.find('=')) == "group_size") {
        group_size = std::stoi(line.substr(line.find('=') + 1));
        if (group_size < 1) { 
          printf("group_size=%i is out of bounds. Defaulting to 1.\n", group_size);
          group_size = 1; 
        }
      }
    }
  } else {
    printf("WARNING: Could not open thread configuration file! Using default values.\n");
  }
  
  printf("Configuration:\n");
  printf("Buffering system = %s\n", use_guttertree? "GutterTree" : "StandAloneGutters");
  printf("Number of groups = %i\n", num_groups);
  printf("Size of groups = %i\n", group_size);
  printf("Directory for on disk data = %s\n", dir.c_str());
  printf("Query backups in memory = %s\n", backup_in_mem? "ON" : "OFF");
  GraphWorker::set_config(num_groups, group_size);
  return {use_guttertree, backup_in_mem, dir};
}

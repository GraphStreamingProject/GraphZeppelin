#include <stdexcept>
#include <boost/multiprecision/cpp_int.hpp>
#include "../include/util.h"
#include "../include/graph_worker.h"
#include "../include/graph.h"

const char *config_file = "streaming.conf";
using uint128_t = boost::multiprecision::uint128_t;

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

std::string configure_system() {
  std::string pre;
  int num_groups = 0;
  int group_size = 0;
  std::string line;
  std::ifstream conf(config_file);
  if (conf.is_open()) {
    while(getline(conf, line)) {
      if (line[0] == '#' || line[0] == '\n') continue;
      if(line.substr(0, line.find('=')) == "path_prefix") {
        pre = line.substr(line.find('=') + 1);
        printf("Buffertree path_prefix = %s\n", pre.c_str());
      }
      if(line.substr(0, line.find('=')) == "num_groups") {
        num_groups = std::stoi(line.substr(line.find('=') + 1));
        printf("Number of groups = %i\n", num_groups);
      }
      if(line.substr(0, line.find('=')) == "group_size") {
        group_size = std::stoi(line.substr(line.find('=') + 1));
        printf("Size of groups = %i\n", group_size);
      }
    }
  } else {
    printf("WARNING: Could not open thread configuration file!\n");
  }
#ifdef USE_FBT_F
  if (pre == "") {
    printf("WARNING: Using default buffer-tree path prefix: ./BUFFTREEDATA/\n");
    pre = "./BUFFTREEDATA/";
  }
#else
  if (!pre.empty()) {
    printf("WARNING: Running with in-memory buffering. Buffer-tree path prefix "
           "will be ignored/\n");
  }
#endif
  if (num_groups == 0) {
    printf("WARNING: Defaulting to a single group\n");
    num_groups = 1;
  }
  if (group_size == 0) {
    printf("WARNING: Defaulting to a group size of 1\n");
    group_size = 1;
  }

  GraphWorker::set_config(num_groups, group_size);
  return pre;
}

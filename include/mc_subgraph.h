#include <map>
#include <mutex>
#include <vector>


#include "cuda_kernel.cuh"


enum GraphType {
  SKETCH = 0,
  ADJLIST = 1,
  FIXED_ADJLIST = 2
};

class MCSubgraph {
private:
  int graph_id;
  CudaUpdateParams* cudaUpdateParams;
  GraphType type;
  node_id_t num_nodes;

  int num_streams; // Number of CUDA Streams

  size_t num_sketch_updates;
  size_t num_adj_edges;

  std::map<node_id_t, std::map<node_id_t, node_id_t>> adjlist;

  double sketch_bytes; // Bytes of sketch graph
  double adjlist_edge_bytes; // Bytes of one edge in adj. list

public:
  std::mutex mutex;

  // Constructor
  MCSubgraph(int graph_id, int num_streams, CudaUpdateParams* cudaUpdateParams, GraphType type, node_id_t num_nodes, double sketch_bytes, double adjlist_edge_bytes);

  void insert_adj_edge(node_id_t src, node_id_t dst);

  // Sample from Adj. list
  node_id_t sample_dst_node(node_id_t src);

  // Get methods
  CudaUpdateParams* get_cudaUpdateParams() { return cudaUpdateParams; }
  GraphType get_type() { return type; }
  size_t get_num_updates() { 
    if (type == SKETCH) {
      return num_sketch_updates;
    } 
    else {
      return num_adj_edges;
    }
  }
  std::map<node_id_t, std::map<node_id_t, node_id_t>> get_adjlist() { return adjlist; }

  // Set methods
  void set_type(GraphType new_type) { type = new_type; }
  void increment_num_sketch_updates(int value) { num_sketch_updates += value; }

  // Delete methods
  void adjlist_delete_src(node_id_t src) { adjlist.erase(src); }
};
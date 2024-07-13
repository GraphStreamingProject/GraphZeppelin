#include <mutex>
#include <set>
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
  std::atomic<GraphType> type;
  node_id_t num_nodes;

  int num_streams;  // Number of CUDA Streams

  std::atomic<int> conversion_counter;

  std::atomic<size_t> num_sketch_updates;
  std::atomic<size_t> num_adj_edges;

  std::vector<std::set<node_id_t>> adjlist;

  double sketch_bytes;        // Bytes of sketch graph
  double adjlist_edge_bytes;  // Bytes of one edge in adj. list

 public:
  std::mutex* adj_mutex;

  // Constructor
  MCSubgraph(int graph_id, int num_streams, CudaUpdateParams* cudaUpdateParams, GraphType type,
             node_id_t num_nodes, double sketch_bytes, double adjlist_edge_bytes);
  ~MCSubgraph();

  void insert_adj_edge(node_id_t src, std::vector<node_id_t> dst_vertices);

  // Get methods
  CudaUpdateParams* get_cudaUpdateParams() { return cudaUpdateParams; }
  GraphType get_type() { return type; }
  size_t get_num_updates() {
    if (type == SKETCH) {
      return num_sketch_updates;
    } else {
      return num_adj_edges;
    }
  }
  size_t get_num_adj_edges() { return num_adj_edges; }
  const std::vector<std::set<node_id_t>>& get_adjlist() { return adjlist; }
  const std::set<node_id_t>& get_neighbor_nodes(node_id_t src) { return adjlist[src]; }

  bool try_acq_conversion() {
    int org_val = 0;
    int new_val = 1;
    return conversion_counter.compare_exchange_strong(org_val, new_val);
  }

  // Set methods
  void set_type(GraphType new_type) { type = new_type; }
  void increment_num_sketch_updates(int value) { num_sketch_updates += value; }
};
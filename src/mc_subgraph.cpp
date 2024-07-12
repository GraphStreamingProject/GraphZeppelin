#include "mc_subgraph.h"

#include <chrono>

// Constructor
MCSubgraph::MCSubgraph(int graph_id, int num_streams, CudaUpdateParams* cudaUpdateParams,
                       GraphType type, node_id_t num_nodes, double sketch_bytes,
                       double adjlist_edge_bytes)
    : graph_id(graph_id),
      num_streams(num_streams),
      cudaUpdateParams(cudaUpdateParams),
      type(type),
      num_nodes(num_nodes),
      adjlist(num_nodes),
      sketch_bytes(sketch_bytes),
      adjlist_edge_bytes(adjlist_edge_bytes) {
  conversion_counter = 0;

  num_sketch_updates = 0;
  num_adj_edges = 0;

  adj_mutex = new std::mutex[num_nodes];
}

MCSubgraph::~MCSubgraph() {
  delete[] adj_mutex;
}

void MCSubgraph::insert_adj_edge(node_id_t src, std::vector<node_id_t> dst_vertices) {
  std::lock_guard<std::mutex> lk(adj_mutex[src]);
  for (auto dst : dst_vertices) {
    if (adjlist[src].find(dst) == adjlist[src].end()) {
      adjlist[src].insert(dst);
      num_adj_edges++;
    } else {
      adjlist[src].erase(dst);  // Current edge already exist, so delete
      num_adj_edges--;
    }
  }
}

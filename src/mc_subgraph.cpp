#include <chrono>

#include "mc_subgraph.h"


// Constructor
MCSubgraph::MCSubgraph(int graph_id, int num_streams, CudaUpdateParams* cudaUpdateParams, GraphType type, node_id_t num_nodes, double sketch_bytes, double adjlist_edge_bytes) : 
  graph_id(graph_id), num_streams(num_streams), cudaUpdateParams(cudaUpdateParams), type(type), num_nodes(num_nodes), sketch_bytes(sketch_bytes), adjlist_edge_bytes(adjlist_edge_bytes) {

  num_sketch_updates = 0;
  num_adj_edges = 0;

  fixed_adj_mutex = new std::mutex[num_nodes];

  for (node_id_t i = 0; i < num_nodes; i++) {
    adjlist[i] = std::map<node_id_t, node_id_t>();
  }
}

void MCSubgraph::insert_adj_edge(node_id_t src, node_id_t dst) {
  if (adjlist[src].find(dst) == adjlist[src].end()) {
    adjlist[src].insert({dst, 1});
    num_adj_edges++;
  }
  else {
    adjlist[src].erase(dst); // Current edge already exist, so delete
    num_adj_edges--;
  }
}

void MCSubgraph::insert_fixed_adj_edge(node_id_t src, node_id_t dst) {
  std::lock_guard<std::mutex> lk(fixed_adj_mutex[src]);
  if (adjlist[src].find(dst) == adjlist[src].end()) {
    adjlist[src].insert({dst, 1});
    num_adj_edges++;
  }
  else {
    adjlist[src].erase(dst); // Current edge already exist, so delete
    num_adj_edges--;
  }
}

// Sample neighbor node from adjlist
node_id_t MCSubgraph::sample_dst_node(node_id_t src) {
  if ((adjlist.find(src) == adjlist.end()) || (adjlist[src].size() == 0)) { // Doesn't exist
    return -1;
  }
  node_id_t dst = adjlist[src].begin()->first;
  adjlist[src].erase(dst);
  return dst;
}
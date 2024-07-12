#include "k_connected_graph.h"

#include <omp.h>

constexpr size_t num_verts = 1 << 16;
constexpr size_t k = 1000;
constexpr size_t vert_multiple = 200; // for k_path verts = k * vert_multiple

void cc_complete(KConnectedGraph& graph, node_id_t vertices) {
#pragma omp parallel for num_threads(4)
  for (node_id_t i = 0; i < vertices; i++) {
    for (node_id_t j = i+1; j < vertices; j++) {
      graph.update({{i, j}, INSERT}, omp_get_thread_num());
    }
  }
}

void path_on_complete_chunks(KConnectedGraph& graph, node_id_t vertices, node_id_t chunk_size,
                             node_id_t edges_in_cut) {
  if (vertices % chunk_size != 0) {
    std::cerr << "ERROR: vertices must be multiple of chunk_size!" << std::endl;
    exit(EXIT_FAILURE);
  }
  node_id_t num_chunks = vertices / chunk_size;
  for (node_id_t c = 0; c < num_chunks; c++) {
    // std::cout << "COMPLETE:" << std::endl;

    // create complete chunk
    for (node_id_t i = 0; i < chunk_size; i++) {
      for (node_id_t j = i+1; j < chunk_size; j++) {
        node_id_t src, dst;
        src = c * chunk_size + i;
        dst = c * chunk_size + j;
        graph.update({{src, dst}, INSERT});
        // std::cout << src << " " << dst << std::endl;
      }
    }

    // create edges to next chunk
    // choose some vertices from chunk, connect them to vertices in next chunk
    if (c < num_chunks-1) {
      // std::cout << "EDGES BETWEEN COMPONENT AND NEXT:" << std::endl;
      for (node_id_t i = 0; i < edges_in_cut; i++) {
        node_id_t src, dst;
        src = (i + 7*c) % chunk_size + c * chunk_size;
        dst = (i + 11*(c+1)) % chunk_size + (c+1) * chunk_size;
        graph.update({{src, dst}, INSERT});
        // std::cout << src << " " << dst << std::endl;
      }
    }
  }
}

void erdos_graph(KConnectedGraph& graph, node_id_t vertices, double density) {
  if (density <= 0 || density > 1)
    std::cerr << "Erdos density must be (0, 1]" << std::endl;
  edge_id_t max_edges = vertices * (vertices - 1) / 2;
  edge_id_t num_edges = max_edges * density;

  std::cout << "Stream: Vertices = " << vertices << " edges = " << num_edges << std::endl;
  edge_id_t p = 29;
  edge_id_t edge = 1;
  for (edge_id_t i = 1; i <= num_edges; i++) {
    edge = (edge * p) % max_edges;
    ++edge;
    graph.update({inv_nondir_non_self_edge_pairing_fn(edge), INSERT});
  }
}

void k_path_graph(KConnectedGraph& graph, node_id_t vertices, node_id_t k) {
  if (k > vertices) {
    std::cerr << "k must be less than vertices!" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (vertices % k != 0) {
    std::cerr << "number of vertices must be a multiple of k!" << std::endl;
    exit(EXIT_FAILURE);
  }
  
  node_id_t hop = k;
  for (node_id_t i = 0; i < vertices; i++) {
    for (node_id_t j = i+1; j <= i + hop && j < vertices; j++) {
      // std::cout << "Edge = " << i << ", " << j << std::endl;
      graph.update({{i, j}, INSERT});
    }
    --hop;
    if (hop == 0) hop = k;
  }
}

int main(int argc, char **argv) {
  GraphConfiguration conf;
  conf.num_groups(40);
  KConnectedGraph graph(k * vert_multiple, conf, 4);
  std::cout << "total vertices = " << k * vert_multiple << std::endl;
  std::cout << "total edges    = " << (k * (k - 1) / 2) * vert_multiple << std::endl;

  //path_on_complete_chunks(graph, num_verts, num_verts / 512, 256);
  k_path_graph(graph, k * vert_multiple, k);
  
  std::vector<std::vector<Edge>> forests = graph.k_spanning_forests(k);
  // for (auto forest : forests) {
  //   for (auto e : forest)
  //     std::cout << e.src << " " << e.dst << std::endl;
  //   std::cout << std::endl;
  // }
}
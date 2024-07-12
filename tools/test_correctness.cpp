#include <fstream>
#include <unordered_map>
#include <iostream>
#include <thread>
#include <algorithm>
#include <chrono>

#include <cc_sketch_alg.h>
#include <graph_verifier.h>

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

struct CorrectnessResults {
  size_t num_failures = 0;
  std::vector<size_t> num_round_hist;
};

CorrectnessResults test_path_correctness(size_t num_vertices, size_t num_graphs,
                                         size_t samples_per_graph) {
  CorrectnessResults results;

  size_t num_rounds = Sketch::calc_cc_samples(num_vertices, 1);
  for (size_t r = 0; r < num_rounds; r++)
    results.num_round_hist.push_back(0);

  std::vector<node_id_t> vertices;
  for (size_t i = 0; i < num_vertices; i++) {
    vertices.push_back(i);
  }

  size_t t = std::chrono::steady_clock::now().time_since_epoch().count();
  std::mt19937_64 gen(t);

  for (size_t g = 0; g < num_graphs; g++) {
    size_t edge_seed = gen();
    std::vector<node_id_t> copy_vertices(vertices);
    std::shuffle(copy_vertices.begin(), copy_vertices.end(), std::mt19937_64(edge_seed));
    GraphVerifier verifier(num_vertices);

    node_id_t cur_node = copy_vertices[0];
    for (size_t i = 1; i < num_vertices; i++) {
      verifier.edge_update({cur_node, copy_vertices[i]});
      cur_node = copy_vertices[i];
    }

    for (size_t s = 0; s < samples_per_graph; s++) {
      CCSketchAlg cc_alg(num_vertices, get_seed());
      
      node_id_t cur_node = copy_vertices[0];
      for (size_t i = 1; i < num_vertices; i++) {
        cc_alg.update({{cur_node, copy_vertices[i]}, INSERT});
        cur_node = copy_vertices[i];
      }
      cc_alg.set_verifier(std::make_unique<GraphVerifier>(verifier));

      std::cout << "graph: " << g << " sample: " << s << " ";
      try {
        cc_alg.connected_components();
      } catch (...) {
        std::cout << " FAILED!" << std::endl;
        results.num_failures += 1;
      }
      results.num_round_hist[cc_alg.last_query_rounds] += 1;
    }
  }

  return results;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Incorrect number of arguments. "
                 "Expected two but got " << argc-1 << std::endl;
    std::cout << "Arguments are: num_vertices, graphs, samples" << std::endl;
    exit(EXIT_FAILURE);
  }

  node_id_t vertices = std::stol(argv[1]);
  unsigned graphs = std::stoi(argv[2]);
  unsigned samples = std::stoi(argv[3]);

  CorrectnessResults results = test_path_correctness(vertices, graphs, samples);

  std::cout << "=== CORRECTNESS RESULTS ===" << std::endl;
  std::cout << " Performed " << graphs * samples << " samples" << std::endl;
  std::cout << " There were " << results.num_failures << " failures" << std::endl;
  std::cout << " Number of Rounds per Query (Histogram): " << std::endl;
  for (size_t r = 0; r < results.num_round_hist.size(); r++) {
    std::cout << "   " << r+1 << ": " << results.num_round_hist[r] << std::endl;
  }
}

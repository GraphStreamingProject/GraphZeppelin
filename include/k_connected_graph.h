#include <graph.h>

class KConnectedGraph : public Graph {
public:
  // Returns k edge-disjoint spanning trees
  std::vector<std::vector<Edge>> k_spanning_forests(size_t k);

  explicit KConnectedGraph(node_id_t num_nodes, int num_inserters = 1)
      : Graph(num_nodes, GraphConfiguration(), num_inserters){};
  explicit KConnectedGraph(const std::string &input_file, int num_inserters = 1)
      : Graph(input_file, GraphConfiguration(), num_inserters){};
  explicit KConnectedGraph(const std::string &input_file, GraphConfiguration config,
                           int num_inserters = 1)
      : Graph(input_file, config, num_inserters){};
  explicit KConnectedGraph(node_id_t num_nodes, GraphConfiguration config, int num_inserters = 1)
      : Graph(num_nodes, config, num_inserters){};
  explicit KConnectedGraph(node_id_t num_nodes, GraphConfiguration config, CudaGraph* cudaGraph, int num_inserters=1)
      : Graph(num_nodes, config, cudaGraph, num_inserters){};

 private:
  // delete all the edges in the spanning forest
  void trim_spanning_forest(std::vector<Edge> &tree);

  // query the sketches for a spanning forest
  std::vector<Edge> get_spanning_forest();

  std::vector<std::vector<node_id_t>> to_merge_and_forest_edges(std::vector<Edge> &forest, 
      std::pair<Edge, SampleSketchRet> *query, std::vector<node_id_t> &reps);

  // perform a rotation by 1 of the Supernodes
  void cycle_supernodes();

  // verify the contents of the forests we've constructed
  void verify_spanning_forests(std::vector<std::vector<Edge>> forests);
};
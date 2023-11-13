#include <graph.h>

class MinCutGraph : public Graph {
public:
  // K-Value
  int k;

  // Returns k edge-disjoint spanning trees
  std::vector<std::vector<Edge>> k_spanning_forests(size_t k);

  explicit MinCutGraph(node_id_t num_nodes, int num_inserters = 1)
      : Graph(num_nodes, GraphConfiguration(), num_inserters){};
  explicit MinCutGraph(const std::string &input_file, int num_inserters = 1)
      : Graph(input_file, GraphConfiguration(), num_inserters){};
  explicit MinCutGraph(const std::string &input_file, GraphConfiguration config, int num_inserters = 1)
      : Graph(input_file, config, num_inserters){};
  explicit MinCutGraph(node_id_t num_nodes, GraphConfiguration config, int num_inserters = 1)
      : Graph(num_nodes, config, num_inserters){};
  explicit MinCutGraph(node_id_t num_nodes, GraphConfiguration config, CudaGraph* cudaGraph, int _k, int num_inserters = 1)
      : Graph(num_nodes, config, cudaGraph, _k, num_inserters){ k = _k; };
  explicit MinCutGraph(node_id_t num_nodes, GraphConfiguration config, GutteringSystem* gutteringSystem, CudaGraph* cudaGraph, int _k, int num_inserters = 1)
      : Graph(num_nodes, config, gutteringSystem, cudaGraph, _k, num_inserters){ k = _k; };

  ~MinCutGraph() override;

private:
  node_id_t k_get_parent(node_id_t node, int k_id);

  std::vector<std::vector<node_id_t>> to_merge_and_forest_edges(std::vector<Edge> &forest, 
      std::pair<Edge, SampleSketchRet> *query, std::vector<node_id_t> &reps, int k_id);

  void k_sample_supernodes(std::pair<Edge, SampleSketchRet> *query, std::vector<node_id_t> &reps, int k_id);

  void k_merge_supernodes(Supernode** copy_supernodes, std::vector<node_id_t> &new_reps,
                        std::vector<std::vector<node_id_t>> &to_merge, bool make_copy, int k_id);

  // query the sketches for a spanning forest
  std::vector<Edge> get_spanning_forest(int k_id);

  // delete all the edges in the spanning forest
  void trim_spanning_forest(std::vector<Edge> &tree);

  // perform a rotation by 1 of the Supernodes
  void cycle_supernodes();

  // verify the contents of the forests we've constructed
  void verify_spanning_forests(std::vector<std::vector<Edge>> forests);
};
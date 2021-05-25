#ifndef TEST_GRAPH_VERIFIER_H
#define TEST_GRAPH_VERIFIER_H

#include <set>
#include "../../include/supernode.h"

/**
 * Runs Kruskal's (deterministic) CC algo.
 * @param input_file the file to read input from.
 * @return an array of connected components.
 */
std::vector<std::set<Node>> kruskal(const string& input_file = "cum_sample.txt");

/**
 * A plugin for the Graph class that runs Boruvka alongside the graph algorithm
 * and verifies the edges and connected components that the graph algorithm
 * generates.
 */
class GraphVerifier {
  std::vector<std::set<Node>> kruskal_ref;
  std::vector<std::set<Node>> boruvka_cc;
  std::vector<std::set<Node>> det_graph;
  Node* parent;
  Node* size;
public:
  GraphVerifier(const string& input_file = "./cum_sample.txt");
  ~GraphVerifier();
  /**
   * Verifies an edge exists in the graph. Verifies that the edge is in the cut
   * of both the endpoints of the edge.
   * @param edge the edge to be tested.
   * @param det_graph the adjacency list representation of the graph in question.
   * @throws BadEdgeException if the edge does not satisfy both conditions.
   */
  void verify_edge(Edge edge);

  /**
   * Verifies the supernode of the given node is a (maximal) connected component.
   * That is, there are no edges in its cut.
   * @param node the node to be tested.
   * @throws NotCCException   if the supernode is not a connected component.
   */
  void verify_cc(Node node);

  /**
   * Verifies the connected components solution is correct. Compares
   * retval against kruskal_ref.
   */
  void verify_soln(vector<set<Node>>& retval);
};

class BadEdgeException : public exception {
  virtual const char* what() const throw() {
    return "The edge is not in the cut of the sample! (standard)";
  }
};


class NotCCException : public exception {
  virtual const char* what() const throw() {
    return "The supernode is not a connected component. It has edges in its "
           "cut!";
  }
};

#endif //TEST_GRAPH_VERIFIER_H

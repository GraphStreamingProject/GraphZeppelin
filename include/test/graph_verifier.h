#pragma once
#include <set>
#include "../supernode.h"

/**
 * A plugin for the Graph class that runs Boruvka alongside the graph algorithm
 * and verifies the edges and connected components that the graph algorithm
 * generates.
 */
class GraphVerifier {
protected:
  std::vector<std::vector<bool>> adj_matrix;

public:
  /**
   * Verifies an edge exists in the graph. Verifies that the edge is in the cut
   * of both the endpoints of the edge.
   * @param edge the edge to be tested.
   * @param det_graph the adjacency list representation of the graph in question.
   * @throws BadEdgeException if the edge does not satisfy both conditions.
   */
  virtual void verify_edge(Edge edge) = 0;

  /**
   * Verifies the supernode of the given node is a (maximal) connected component.
   * That is, there are no edges in its cut.
   * @param node the node to be tested.
   * @throws NotCCException   if the supernode is not a connected component.
   */
  virtual void verify_cc(node_id_t node) = 0;

  /**
   * Verifies the connected components solution is correct. Compares
   * retval against kruskal_ref.
   */
  virtual void verify_soln(std::vector<std::set<node_id_t>> &retval) = 0;

  std::vector<std::vector<bool>> extract_adj_matrix() {return adj_matrix;}

  virtual ~GraphVerifier() {};
};

class BadEdgeException : public std::exception {
  virtual const char* what() const throw() {
    return "The edge is not in the cut of the sample! (standard)";
  }
};


class NotCCException : public std::exception {
  virtual const char* what() const throw() {
    return "The supernode is not a connected component. It has edges in its "
           "cut!";
  }
};

#pragma once
#include <set>
#include <vector>

#include "types.h"

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
   * Verifies an edge exists in the graph.
   * @param edge the edge to be tested.
   * @throws BadEdgeException if the edge does not exist in the graph.
   */
  virtual void verify_edge(Edge edge) = 0;

  /**
   * Verifies the connected components solution is correct. Compares
   * retval against kruskal_ref.
   */
  virtual void verify_soln(std::vector<std::set<node_id_t>> &retval) = 0;

  std::vector<std::vector<bool>> extract_adj_matrix() { return adj_matrix; }

  GraphVerifier() = default;
  GraphVerifier(std::vector<std::vector<bool>> _adj) : adj_matrix(std::move(_adj)) {};

  virtual ~GraphVerifier() {};
};

class BadEdgeException : public std::exception {
  virtual const char* what() const throw() {
    return "The edge is not in the cut of the sample!";
  }
};

class IncorrectCCException : public std::exception {
  virtual const char* what() const throw() {
    return "The connected components are incorrect!";
  }
};

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
   * Verifies an edge exists in the graph.
   * @param edge the edge to be tested.
   * @throws BadEdgeException if the edge does not exist in the graph.
   */
  virtual void verify_edge(Edge edge) = 0;

  /**
   * Verifies the connected components solution is correct. Compares
   * retval against kruskal_ref.
   */
  virtual void verify_soln(std::vector<std::set<node_id_t>>& retval) = 0;

  std::vector<std::vector<bool>> extract_adj_matrix() { return adj_matrix; }

  GraphVerifier() = default;
  GraphVerifier(std::vector<std::vector<bool>> _adj) : adj_matrix(std::move(_adj)){};

  virtual ~GraphVerifier(){};
};

class BadEdgeException : public std::exception {
 private:
  Edge edge;
  std::string msg = "";

 public:
  BadEdgeException(Edge e) : edge(e) {
    msg = "The edge: {" + std::to_string(edge.src) + ", " + std::to_string(edge.dst);
    msg += "} does not exist in the Graph!";
  }
  virtual const char* what() const throw() { return msg.c_str(); }
};

class IncorrectCCException : public std::exception {
 private:
  std::vector<std::set<node_id_t>> expected_cc;
  std::vector<std::set<node_id_t>> user_cc;
  std::string msg;

 public:
  IncorrectCCException(std::vector<std::set<node_id_t>> exp, std::vector<std::set<node_id_t>> usr)
      : expected_cc(exp), user_cc(usr) {
    msg = "Components Expected: {\n";
    for (auto& cc : expected_cc) {
      msg += " [";
      for (auto& v : cc) {
        msg += std::to_string(v) + ",";
      }
      msg.erase(msg.length() - 1, 1);
      msg += "]\n";
    }
    msg += "}\nComponents Given: {\n";
    for (auto& cc : user_cc) {
      msg += " [";
      for (auto& v : cc) {
        msg += std::to_string(v) + ",";
      }
      msg.erase(msg.length() - 1, 1);
      msg += "]\n";
    }
    msg += "}\n";
  }
  virtual const char* what() const throw() { return msg.c_str(); }
};

#ifndef TEST_DETERMINISTIC_H
#define TEST_DETERMINISTIC_H

#include <set>
#include "../../include/supernode.h"
#define VERIFY_SAMPLES_F

// TODO: test if can declare kruskal_run in graph_test
extern bool kruskal_run;

/**
 * Runs Kruskal's (deterministic) CC algo.
 * @param input_file the file to read input from.
 * @return an array of connected components.
 */
std::vector<std::set<Node>> kruskal(const string& input_file = "cum_sample.txt");

/**
 * Verifies an edge exists in the graph. Verifies that the edge is in the cut
 * of both the endpoints of the edge.
 * @param edge the edge to be tested.
 * @throws BadEdgeException if the edge does not satisfy both conditions.
 * @throws NoPrepException  if this function is called before kruskal() is
 *                          run on the graph.
 */
void verify_edge(Edge edge);

/**
 * Verifies the supernode of the given node is a (maximal) connected component.
 * That is, there are no edges in its cut.
 * @param node the node to be tested.
 * @throws NotCCException   if the supernode is not a connected component.
 * @throws NoPrepException  if this function is called before kruskal() is
 *                          run on the graph.
 */
void verify_cc(Node node);

class BadEdgeException : public exception {
  virtual const char* what() const throw() {
    return "The edge is not in the cut of the sample!";
  }
};

class NotCCException : public exception {
  virtual const char* what() const throw() {
    return "The supernode is not a connected component. It has edges in its "
           "cut!";
  }
};

class NoPrepException : public exception {
  virtual const char* what() const throw() {
    return "Cannot run verifier function without running Kruskal on this graph "
           "stream first!";
  }
};

#endif //TEST_DETERMINISTIC_H

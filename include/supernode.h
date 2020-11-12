#pragma once
#include <boost/optional.hpp>
#include "sketch.h"

typedef unsigned long long int Node;
typedef std::pair<Node, Node> Edge;

/**
 * This interface implements the "supernode" so Boruvka can use it as a black
 * box without needing to worry about implementing l_0.
 */
class Supernode {
  // collection of logn sketches to query from, since we can't query from one
  // sketch more than once
  vector<Sketch*> sketches;
  int idx;
  int logn;
public:
  /**
   * @param n     the total number of nodes in the graph.
   * @param seed  the (fixed) seed value passed to each supernode.
   */
  Supernode(unsigned long long int n, long seed);

  /**
   * Function to sample an edge from the cut of a supernode.
   * @return an edge in the cut, otherwise None.
   */
  boost::optional<Edge> sample();

  /**
   * In-place merge function. Guaranteed to update the caller Supernode.
   */
  void merge(Supernode& other);

  /**
   * Insert or delete an edge into the supernode. Guaranteed to be processed
   * BEFORE Boruvka starts.
   */
  void update(pair<Edge, int> update);
};


class OutOfQueriesException : public exception {
  virtual const char* what() const throw() {
    return "This supernode cannot be sampled more times!";
  }
};
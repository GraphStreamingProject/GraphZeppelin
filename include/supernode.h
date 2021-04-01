#pragma once
#include <boost/optional.hpp>
#include "sketch.h"

using namespace std;

typedef uint64_t Node;
typedef std::pair<Node, Node> Edge;

/**
 * This interface implements the "supernode" so Boruvka can use it as a black
 * box without needing to worry about implementing l_0.
 */
class Supernode {
  /* collection of logn sketches to query from, since we can't query from one
     sketch more than once */
  vector<Sketch*> sketches;
  int idx;
  int logn;
public:
#ifdef EXT_MEM_POST_PROC_F
  Node ext_mem_parent_ptr; // points to this supernode's "parent" in post-processing
  Node ext_mem_size; // size of the DSU set corresponding to this supernode
  bool ext_mem_destroyed; // whether or not this supernode has been
  // absorbed into another
#endif
  /**
   * @param n     the total number of nodes in the graph.
   * @param seed  the (fixed) seed value passed to each supernode.
   */
  Supernode(uint64_t n, long seed);

  /**
   * Function to sample an edge from the cut of a supernode.
   * @return                        an edge in the cut, represented as an Edge
   *                                with LHS <= RHS, otherwise None.
   * @throws OutOfQueriesException  if the sketch collection has been sampled
   *                                too many times.
   * @throws NoGoodBucketException  if no "good" bucket can be found,
   *                                according to the specification of L0
   *                                sampling.
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
  void update(Edge update);

  /**
   * Update all the sketches in a supernode, given a batch of updates
   * @param updates A vector of Updates to process
   */
  void batch_update(const std::vector<vec_t>& updates);
};


class OutOfQueriesException : public exception {
  virtual const char* what() const throw() {
    return "This supernode cannot be sampled more times!";
  }
};

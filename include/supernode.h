#pragma once
#include <boost/optional.hpp>
#include <fstream>
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
  std::mutex node_mt;

  FRIEND_TEST(SupernodeTestSuite, TestBatchUpdate);
  FRIEND_TEST(SupernodeTestSuite, TestConcurrency);
  FRIEND_TEST(SupernodeTestSuite, TestSerialization);
  FRIEND_TEST(EXPR_Parallelism, N10kU100k);

public:
  const uint64_t n; // for creating a copy
  const long seed; // for creating a copy
  /**
   * @param n     the total number of nodes in the graph.
   * @param seed  the (fixed) seed value passed to each supernode.
   */
  Supernode(uint64_t n, long seed);

  /**
   * Utility constructor to deserialize from a binary input stream.
   * @param n
   * @param seed
   * @param in          the stream to read from.
   */
  Supernode(uint64_t n, long seed, std::fstream& binary_in);

  ~Supernode();

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
   * Insert or delete an (encoded) edge into the supernode. Guaranteed to be
   * processed BEFORE Boruvka starts.
   */
  void update(vec_t update);

  /**
   * Update all the sketches in a supernode, given a batch of updates.
   * @param delta_node  a delta supernode created through calling
   *                    Supernode::delta_supernode.
   */
  void apply_delta_update(const Supernode* delta_node);

  /**
   * Create new delta supernode with given initial parmameters and batch of
   * updates to apply.
   * @param n       see declared constructor.
   * @param seed    see declared constructor.
   * @param updates the batch of updates to apply.
   * @return
   */
  static Supernode* delta_supernode(uint64_t n, long seed, const
  std::vector<vec_t>& updates);

  /**
   * Serialize the supernode to a binary output stream.
   * @param out the stream to write to.
   */
  void write_binary(fstream &binary_out);
};


class OutOfQueriesException : public exception {
  virtual const char* what() const throw() {
    return "This supernode cannot be sampled more times!";
  }
};

#pragma once
#include <boost/optional.hpp>
#include <fstream>
#include <sys/mman.h>

#include "l0_sampling/sketch.h"

using namespace std;

typedef std::pair<node_t, node_t> Edge;

/**
 * This interface implements the "supernode" so Boruvka can use it as a black
 * box without needing to worry about implementing l_0.
 */
class Supernode {
  // the size of a super-node in bytes including the all sketches off the end
  static uint32_t bytes_size; 
  int idx;
  int logn;
  std::mutex node_mt;

  FRIEND_TEST(SupernodeTestSuite, TestBatchUpdate);
  FRIEND_TEST(SupernodeTestSuite, TestConcurrency);
  FRIEND_TEST(SupernodeTestSuite, TestSerialization);
  FRIEND_TEST(GraphTestSuite, TestCorrectnessOfReheating);
  FRIEND_TEST(EXPR_Parallelism, N10kU100k);

public:
  const uint64_t n; // for creating a copy
  const long seed; // for creating a copy
  
private:
  size_t sketch_size;

  /* collection of logn sketches to query from, since we can't query from one
     sketch more than once */
  // The sketches, off the end.
  alignas(Sketch) char sketch_buffer[1];
  
  /**
   * @param n     the total number of nodes in the graph.
   * @param seed  the (fixed) seed value passed to each supernode.
   */
  Supernode(uint64_t n, long seed);

  /**
   * @param n         the total number of nodes in the graph.
   * @param seed      the (fixed) seed value passed to each supernode.
   * @param binary_in A stream to read the file from.
   */
  Supernode(uint64_t n, long seed, std::fstream &binary_in);

public:
//  Supernode(const Supernode& s);
  static Supernode* makeSupernode(uint64_t n, long seed);
  static Supernode* makeSupernode(uint64_t n, long seed, std::fstream &binary_in);

  /**
   * Makes a supernode at the provided location (and does no error checking).
   * @param loc     the memory location to put the supernode.
   * @param n       the total number of nodes in the graph.
   * @param seed    the (fixed) seed value passed to each supernode.
   * @return        a pointer to loc, the location of the supernode.
   */
  static Supernode* makeSupernode(void* loc, uint64_t n, long seed);

  ~Supernode();

  // get the ith sketch in the sketch array
  inline Sketch* get_sketch(size_t i) {
    return reinterpret_cast<Sketch*>(sketch_buffer + i * sketch_size);
  }

  // get the ith sketch in the sketch array
  inline const Sketch* get_sketch(size_t i) const {
    return reinterpret_cast<const Sketch*>(sketch_buffer + i * sketch_size);
  }

  static inline void configure(uint64_t n, double num_bucket_factor=0.5) {
    Sketch::configure(n*n, num_bucket_factor);
    bytes_size = sizeof(Supernode) + log2(n) * Sketch::sketchSizeof() - sizeof(char);
  }

  static inline uint32_t get_size() {
    return bytes_size;
  }

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
   * @param loc     the location to place the delta in
   */
  static void delta_supernode(uint64_t n, long seed, const
  std::vector<vec_t>& updates, void *loc);

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

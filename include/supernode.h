#pragma once
#include <fstream>
#include <sys/mman.h>
#include <graph_zeppelin_common.h>

#include "l0_sampling/sketch.h"

typedef std::pair<node_id_t, node_id_t> Edge;

/**
 * This interface implements the "supernode" so Boruvka can use it as a black
 * box without needing to worry about implementing l_0.
 */
class Supernode {
  // the size of a super-node in bytes including the all sketches off the end
  static size_t bytes_size; 
  static size_t serialized_size; // the size of a supernode that has been serialized
  int idx;
  int num_sketches;
  std::mutex node_mt;

  FRIEND_TEST(SupernodeTestSuite, TestBatchUpdate);
  FRIEND_TEST(SupernodeTestSuite, TestConcurrency);
  FRIEND_TEST(SupernodeTestSuite, TestSerialization);
  FRIEND_TEST(GraphTestSuite, TestCorrectnessOfReheating);
  FRIEND_TEST(GraphTest, TestSupernodeRestoreAfterCCFailure);
  FRIEND_TEST(EXPR_Parallelism, N10kU100k);

public:
  const uint64_t n; // for creating a copy
  const uint64_t seed; // for creating a copy
  
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
  Supernode(uint64_t n, uint64_t seed);

  /**
   * @param n         the total number of nodes in the graph.
   * @param seed      the (fixed) seed value passed to each supernode.
   * @param binary_in A stream to read the data from.
   */
  Supernode(uint64_t n, uint64_t seed, std::istream &binary_in);

  Supernode(const Supernode& s);

  // get the ith sketch in the sketch array
  inline Sketch* get_sketch(size_t i) {
    return reinterpret_cast<Sketch*>(sketch_buffer + i * sketch_size);
  }

  // version of above for const supernode objects
  inline const Sketch* get_sketch(size_t i) const {
    return reinterpret_cast<const Sketch*>(sketch_buffer + i * sketch_size);
  }

public:
  /**
   * Supernode construtors
   * @param n       the total number of nodes in the graph.
   * @param seed    the (fixed) seed value passed to each supernode.
   * @param loc     (Optional) the memory location to put the supernode.
   * @return        a pointer to the newly created supernode object
   */
  static Supernode* makeSupernode(uint64_t n, long seed, void *loc = malloc(bytes_size));
  
  // create supernode from file
  static Supernode* makeSupernode(uint64_t n, long seed, std::istream &binary_in, 
                                  void *loc = malloc(bytes_size));
  // copy 'constructor'
  static Supernode* makeSupernode(const Supernode& s, void *loc = malloc(bytes_size));

  ~Supernode();

  static inline void configure(uint64_t n, vec_t sketch_fail_factor=100) {
    Sketch::configure(n*n, sketch_fail_factor);
    bytes_size = sizeof(Supernode) + log2(n)/(log2(3)-1) * Sketch::sketchSizeof() - sizeof(char);
    serialized_size = bytes_size - log2(n)/(log2(3)-1) * (sizeof(Sketch) - sizeof(char));
    serialized_size -= sizeof(Supernode) - sizeof(char);
  }

  static inline size_t get_size() {
    return bytes_size;
  }

  // return the size of a supernode that has been serialized using write_binary()
  static inline size_t get_serialized_size() {
    return serialized_size;
  }

  inline size_t get_sketch_size() {
    return sketch_size;
  }

  // return the number of sketches held in this supernode
  int get_num_sktch() { return num_sketches; };

  inline bool out_of_queries() {
    return idx == num_sketches;
  }

  inline int curr_idx() {
    return idx;
  }

  inline void incr_idx() {
    ++idx;
  }

  // reset the supernode query metadata
  // we use this when resuming insertions after CC made copies in memory
  inline void reset_query_state() { 
    for (int i = 0; i < idx; i++) {
      get_sketch(i)->reset_queried();
    }
    idx = 0;
  }

  // get the ith sketch in the sketch array as a const object
  inline const Sketch* get_const_sketch(size_t i) {
    return reinterpret_cast<Sketch*>(sketch_buffer + i * sketch_size);
  }

  /**
   * Function to sample an edge from the cut of a supernode.
   * @return   an edge in the cut, represented as an Edge with LHS <= RHS, 
   *           if one exists. Additionally, returns a code representing the
   *           sample result (good, zero, or fail)
   */
  std::pair<Edge, SampleSketchRet> sample();

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
  static void delta_supernode(uint64_t n, uint64_t seed, const
  std::vector<vec_t>& updates, void *loc);

  /**
   * Serialize the supernode to a binary output stream.
   * @param out the stream to write to.
   */
  void write_binary(std::ostream &binary_out);
};


class OutOfQueriesException : public std::exception {
  virtual const char* what() const throw() {
    return "This supernode cannot be sampled more times!";
  }
};

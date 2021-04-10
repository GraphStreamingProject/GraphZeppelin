#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include "../../include/supernode.h"

const long seed = 7000000001;
const unsigned long long int num_nodes = 2000;

class EXPR_Parallelism : public testing::Test {
protected:
  static vector<Edge>* graph_edges;
  static vector<Edge>* odd_graph_edges;
  static bool* prime;
  static void SetUpTestSuite() {
    srand(1000000007);
    graph_edges = new vector<Edge>();
    odd_graph_edges = new vector<Edge>();
    for (unsigned i=2;i<num_nodes;++i) {
      for (unsigned j = i*2; j < num_nodes; j+=i) {
        graph_edges->push_back({i,j});
        if ((j/i)%2) odd_graph_edges->push_back({i,j});
      }
    }

    // sieve
    prime = (bool*) malloc(num_nodes*sizeof(bool));
    fill(prime,prime+num_nodes,true);
    for (unsigned i = 2; i < num_nodes; i++) {
      if (prime[i]) {
        for (unsigned j = i * i; j < num_nodes; j += i) prime[j] = false;
      }
    }
  }
  static void TearDownTestSuite() {
    delete graph_edges;
    delete odd_graph_edges;
    free(prime);
  }

  void SetUp() override {}
  void TearDown() override {}
};

vector<Edge>* EXPR_Parallelism::graph_edges;
vector<Edge>* EXPR_Parallelism::odd_graph_edges;
bool* EXPR_Parallelism::prime;


TEST_F(EXPR_Parallelism, N10kU100k) {
  unsigned long vec_size = 100000000, num_updates = 100000;
  srand(time(NULL));
  std::vector<vec_t> updates(num_updates);
  for (unsigned long i = 0; i < num_updates; i++) {
    updates[i] = static_cast<vec_t>(rand() % vec_size);
  }
  auto seed = rand();
  Supernode supernode(vec_size, seed);
  Supernode supernode_batch(vec_size, seed);
  std::cout << "logn = " << supernode_batch.logn << endl;
  auto start_time = std::chrono::steady_clock::now();
  for (const auto& update : updates) {
    supernode.update(update);
  }
  std::cout << "One by one updates took " << static_cast<std::chrono::duration<long double>>(std::chrono::steady_clock::now() - start_time).count() << std::endl;
  start_time = std::chrono::steady_clock::now();
  supernode_batch.batch_update(updates);
  std::cout << "Batched updates took " << static_cast<std::chrono::duration<long double>>(std::chrono::steady_clock::now() - start_time).count() << std::endl;

  ASSERT_EQ(supernode.logn, supernode_batch.logn);
  ASSERT_EQ(supernode.idx, supernode_batch.idx);
  for (int i=0;i<supernode.logn;++i) {
    Sketch* sketch = supernode.sketches[i];
    Sketch* sketch_batch = supernode_batch.sketches[i];
    ASSERT_EQ(sketch->seed, sketch_batch->seed);
    ASSERT_EQ(sketch->n, sketch_batch->n);
    ASSERT_EQ(sketch->buckets.size(), sketch_batch->buckets.size());
    for (auto it1 = sketch->buckets.cbegin(), it2 = sketch_batch->buckets.cbegin();
         it1 != sketch->buckets.cend(); it1++, it2++) {
      ASSERT_EQ(it1->a, it2->a);
      ASSERT_EQ(it1->c, it2->c);
    }
  }
}

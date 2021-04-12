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

void parallel_test(unsigned long vec_size, unsigned long num_updates) {
  srand(time(NULL));
  std::vector<vec_t> updates(num_updates);
  for (unsigned long i = 0; i < num_updates; i++) {
    updates[i] = static_cast<vec_t>(rand() % vec_size);
  }
  auto seed = rand();
  Supernode supernode(vec_size, seed);
  Supernode supernode_batch(vec_size, seed);
  auto start_time = std::chrono::steady_clock::now();
  for (const auto& update : updates) {
    supernode.update(update);
  }
  long double one, batch;
  one = static_cast<std::chrono::duration<long double>>(std::chrono::steady_clock::now() - start_time).count();
  std::cout << "One by one updates took " << one << std::endl;
  start_time = std::chrono::steady_clock::now();
  supernode_batch.batch_update(updates);
  batch = static_cast<std::chrono::duration<long double>>(std::chrono::steady_clock::now() - start_time).count();
  std::cout << "Batched updates took " << batch << std::endl;
  std::cout << batch/one << std::endl;
}

TEST_F(EXPR_Parallelism, Experiment) {
  for (int i = 2; i <=30; i+=2) {
    unsigned long vec_size = (unsigned long) pow(2,i) + 1;
    for (int j = 3; j <= 7; ++j) {
      unsigned long num_updates = (unsigned long) pow(10,j);
      std::cout << "-------<logn:" << i << ", upd:10e" << j << ">-------" << endl;
      parallel_test(vec_size,num_updates);
      std::cout << std::endl;
    }
  }
}

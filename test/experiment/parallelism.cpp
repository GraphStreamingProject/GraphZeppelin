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
  std::cout << "logn = " << (int) log2(vec_size) << std::endl;
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
  std::cout << "One by one updates took " << static_cast<std::chrono::duration<long double>>(std::chrono::steady_clock::now() - start_time).count() << std::endl;
  start_time = std::chrono::steady_clock::now();
  supernode_batch.batch_update(updates);
  std::cout << "Batched updates took " << static_cast<std::chrono::duration<long double>>(std::chrono::steady_clock::now() - start_time).count() << std::endl;
}

TEST_F(EXPR_Parallelism, L6U6) {
  unsigned long vec_size = 10e6, num_updates = 10e6;
  parallel_test(vec_size,num_updates);
}

TEST_F(EXPR_Parallelism, L6U7) {
  unsigned long vec_size = 10e6, num_updates = 10e7;
  parallel_test(vec_size,num_updates);
}

TEST_F(EXPR_Parallelism, L6U8) {
  unsigned long vec_size = 10e6, num_updates = 10e8;
  parallel_test(vec_size,num_updates);
}

TEST_F(EXPR_Parallelism, L8U5) {
  unsigned long vec_size = 10e8, num_updates = 10e5;
  parallel_test(vec_size,num_updates);
}

TEST_F(EXPR_Parallelism, L8U6) {
  unsigned long vec_size = 10e8, num_updates = 10e6;
  parallel_test(vec_size,num_updates);
}


TEST_F(EXPR_Parallelism, L8U7) {
  unsigned long vec_size = 10e8, num_updates = 10e7;
  parallel_test(vec_size,num_updates);
}

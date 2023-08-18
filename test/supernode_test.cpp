#include "../include/supernode.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <thread>

#include "../include/graph_worker.h"

const long seed = 7000000001;
const unsigned long long int num_nodes = 2000;

class SupernodeTestSuite : public testing::Test {
protected:
  static std::vector<Edge> graph_edges;
  static std::vector<Edge> odd_graph_edges;
  static std::vector<bool> prime;
  static void SetUpTestSuite() {
    graph_edges = std::vector<Edge>();
    odd_graph_edges = std::vector<Edge>();
    for (unsigned i = 2; i < num_nodes; ++i) {
      for (unsigned j = i * 2; j < num_nodes; j += i) {
        graph_edges.push_back({i, j});
        if ((j / i) % 2) odd_graph_edges.push_back({i, j});
      }
    }

    // sieve
    prime = std::vector<bool>(num_nodes, true);
    for (unsigned i = 2; i < num_nodes; i++) {
      if (prime[i]) {
        for (unsigned j = i * i; j < num_nodes; j += i) prime[j] = false;
      }
    }
  }
  static void TearDownTestSuite() {
    graph_edges = std::vector<Edge>();
    odd_graph_edges = std::vector<Edge>();
    prime = std::vector<bool>();
  }

  void SetUp() override { Supernode::configure(num_nodes); }
  void TearDown() override {}
};

std::vector<Edge> SupernodeTestSuite::graph_edges;
std::vector<Edge> SupernodeTestSuite::odd_graph_edges;
std::vector<bool> SupernodeTestSuite::prime;

TEST_F(SupernodeTestSuite, GIVENnoEdgeUpdatesIFsampledTHENnoEdgeIsReturned) {
  Supernode* s = Supernode::makeSupernode(num_nodes, seed);
  SampleSketchRet ret_code = s->sample().second;
  ASSERT_EQ(ret_code, ZERO) << "Did not get ZERO when sampling empty vector";
}

TEST_F(SupernodeTestSuite, IFsampledTooManyTimesTHENthrowOutOfQueries) {
  Supernode* s = Supernode::makeSupernode(num_nodes, seed);
  for (int i = 0; i < Supernode::get_max_sketches(); ++i) {
    s->sample();
  }
  ASSERT_THROW(s->sample(), OutOfQueriesException);
}

TEST_F(SupernodeTestSuite, SketchesHaveUniqueSeeds) {
  Supernode* s = Supernode::makeSupernode(num_nodes, seed);
  std::set<size_t> seeds;

  for (int i = 0; i < Supernode::get_max_sketches(); ++i) {
    Sketch* sketch = s->get_sketch(i);
    for (size_t i = 0; i < sketch->get_columns(); i++) {
      size_t seed = sketch->column_seed(i);
      ASSERT_EQ(seeds.count(seed), 0);
      seeds.insert(seed);
    }
  }
}

TEST_F(SupernodeTestSuite, TestSampleInsertGrinder) {
  std::vector<Supernode*> snodes;
  snodes.reserve(num_nodes);
  for (unsigned i = 0; i < num_nodes; ++i) snodes[i] = Supernode::makeSupernode(num_nodes, seed);

  // insert all edges
  for (auto edge : graph_edges) {
    vec_t encoded = concat_pairing_fn(edge.src, edge.dst);
    snodes[edge.src]->update(encoded);
    snodes[edge.dst]->update(encoded);
  }

  // must have at least logn successes per supernode
  int successes = 0;

  Edge sampled;
  for (unsigned i = 2; i < num_nodes; ++i) {
    for (int j = 0; j < (int)Supernode::get_max_sketches(); ++j) {
      std::pair<Edge, SampleSketchRet> sample_ret = snodes[i]->sample();
      sampled = sample_ret.first;
      SampleSketchRet ret_code = sample_ret.second;
      if (ret_code == FAIL) continue;

      successes++;
      if (i >= num_nodes / 2 && prime[i]) {
        ASSERT_EQ(ret_code, ZERO) << "False positive in sample " << i;
      } else {
        ASSERT_NE(ret_code, ZERO) << "False negative in sample " << i;
        ASSERT_TRUE(std::max(sampled.src, sampled.dst) % std::min(sampled.src, sampled.dst) == 0 &&
                    (i == sampled.src || i == sampled.dst))
            << "Failed on {" << sampled.src << "," << sampled.dst << "} with i = " << i;
      }
    }
    ASSERT_GE(successes, (int)log2(num_nodes))
        << "Fewer than logn successful queries: supernode " << i;
  }
  for (unsigned i = 0; i < num_nodes; ++i) free(snodes[i]);
}

TEST_F(SupernodeTestSuite, TestSampleDeleteGrinder) {
  std::vector<Supernode*> snodes;
  snodes.reserve(num_nodes);
  for (unsigned i = 0; i < num_nodes; ++i) snodes[i] = Supernode::makeSupernode(num_nodes, seed);

  // insert all edges
  for (auto edge : graph_edges) {
    vec_t encoded = concat_pairing_fn(edge.src, edge.dst);
    snodes[edge.src]->update(encoded);
    snodes[edge.dst]->update(encoded);
  }
  // then remove half of them (odds)
  for (auto edge : odd_graph_edges) {
    vec_t encoded = concat_pairing_fn(edge.src, edge.dst);
    snodes[edge.src]->update(encoded);
    snodes[edge.dst]->update(encoded);
  }

  // must have at least logn successes per supernode
  int successes = 0;

  Edge sampled;
  for (unsigned i = 2; i < num_nodes; ++i) {
    for (int j = 0; j < (int)Supernode::get_max_sketches(); ++j) {
      std::pair<Edge, SampleSketchRet> sample_ret = snodes[i]->sample();
      sampled = sample_ret.first;
      SampleSketchRet ret_code = sample_ret.second;
      if (ret_code == FAIL) continue;

      successes++;
      if (i >= num_nodes / 2 && i % 2) {
        ASSERT_EQ(ret_code, ZERO) << "False positive in sample " << i;
      } else {
        ASSERT_NE(ret_code, ZERO) << "False negative in sample " << i;
        ASSERT_TRUE(std::max(sampled.src, sampled.dst) % std::min(sampled.src, sampled.dst) == 0 &&
                    (std::max(sampled.src, sampled.dst) / std::min(sampled.src, sampled.dst)) % 2 ==
                        0 &&
                    (i == sampled.src || i == sampled.dst))
            << "Failed on {" << sampled.src << "," << sampled.dst << "} with i = " << i;
      }
    }
    ASSERT_GE(successes, (int)log2(num_nodes))
        << "Fewer than logn successful queries: supernode " << i;
  }
  for (unsigned i = 0; i < num_nodes; ++i) free(snodes[i]);
}

void inline apply_delta_to_node(Supernode* node, const std::vector<vec_t>& updates) {
  auto* loc = (Supernode*)malloc(Supernode::get_size());
  Supernode::delta_supernode(node->n, node->seed, updates, loc);
  node->apply_delta_update(loc);
  free(loc);
}

TEST_F(SupernodeTestSuite, TestBatchUpdate) {
  unsigned long vec_size = 1000000000, num_updates = 100000;
  srand(time(nullptr));
  std::vector<vec_t> updates(num_updates);
  for (unsigned long i = 0; i < num_updates; i++) {
    updates[i] = static_cast<vec_t>(rand() % vec_size);
  }
  auto seed = rand();
  Supernode::configure(vec_size);
  Supernode* supernode = Supernode::makeSupernode(vec_size, seed);
  Supernode* supernode_batch = Supernode::makeSupernode(vec_size, seed);
  auto start_time = std::chrono::steady_clock::now();
  for (const auto& update : updates) {
    supernode->update(update);
  }
  std::cout << "One by one updates took "
            << static_cast<std::chrono::duration<long double>>(std::chrono::steady_clock::now() -
                                                               start_time)
                   .count()
            << std::endl;
  start_time = std::chrono::steady_clock::now();
  apply_delta_to_node(supernode_batch, updates);
  std::cout << "Batched updates took "
            << static_cast<std::chrono::duration<long double>>(std::chrono::steady_clock::now() -
                                                               start_time)
                   .count()
            << std::endl;

  ASSERT_EQ(supernode->samples_remaining(), supernode_batch->samples_remaining());
  ASSERT_EQ(supernode->sample_idx, supernode_batch->sample_idx);
  for (int i = 0; i < supernode->samples_remaining(); ++i) {
    ASSERT_EQ(*supernode->get_sketch(i), *supernode_batch->get_sketch(i));
  }
}

TEST_F(SupernodeTestSuite, TestConcurrency) {
  unsigned num_threads = std::thread::hardware_concurrency() - 1;
  unsigned vec_len = 1000000;
  unsigned num_updates = 100000;
  Supernode::configure(vec_len);

  std::vector<std::vector<vec_t>> test_vec(num_threads, std::vector<vec_t>(num_updates));
  for (unsigned i = 0; i < num_threads; ++i) {
    for (unsigned long j = 0; j < num_updates; ++j) {
      test_vec[i][j] = static_cast<vec_t>(rand() % vec_len);
    }
  }
  int seed = rand();

  Supernode* supernode = Supernode::makeSupernode(vec_len, seed);
  Supernode* piecemeal = Supernode::makeSupernode(vec_len, seed);

  // concurrently run batch_updates
  std::thread thd[num_threads];
  for (unsigned i = 0; i < num_threads; ++i) {
    thd[i] = std::thread(apply_delta_to_node, piecemeal, std::ref(test_vec[i]));
  }

  // do single-update sketch in the meantime
  for (unsigned i = 0; i < num_threads; ++i) {
    for (unsigned long j = 0; j < num_updates; j++) {
      supernode->update(test_vec[i][j]);
    }
  }

  for (unsigned i = 0; i < num_threads; ++i) {
    thd[i].join();
  }

  for (int i = 0; i < Supernode::get_max_sketches(); ++i) {
    ASSERT_EQ(*supernode->get_sketch(i), *piecemeal->get_sketch(i));
  }
}

TEST_F(SupernodeTestSuite, TestSerialization) {
  std::vector<Supernode*> snodes;
  snodes.reserve(num_nodes);
  for (unsigned i = 0; i < num_nodes; ++i) snodes[i] = Supernode::makeSupernode(num_nodes, seed);

  // insert all edges
  for (auto edge : graph_edges) {
    vec_t encoded = concat_pairing_fn(edge.src, edge.dst);
    snodes[edge.src]->update(encoded);
    encoded = concat_pairing_fn(edge.dst, edge.src);
    snodes[edge.dst]->update(encoded);
  }

  auto file = std::fstream("./out_supernode.txt", std::ios::out | std::ios::binary);
  snodes[num_nodes / 2]->write_binary(file);
  file.close();

  auto in_file = std::fstream("./out_supernode.txt", std::ios::in | std::ios::binary);

  Supernode* reheated = Supernode::makeSupernode(num_nodes, seed, in_file);

  for (int i = 0; i < Supernode::get_max_sketches(); ++i) {
    ASSERT_EQ(*snodes[num_nodes / 2]->get_sketch(i), *reheated->get_sketch(i));
  }
}

TEST_F(SupernodeTestSuite, ExhaustiveSample) {
  size_t runs = 10;
  size_t vertices = 11;
  Supernode::configure(vertices);
  for (size_t i = 0; i < runs; i++) {
    Supernode* s_node = Supernode::makeSupernode(vertices, seed);

    for (size_t s = 1; s < vertices; s++) {
      for (size_t d = s+1; d < vertices; d++) {
        s_node->update(concat_pairing_fn(s, d));
      }
    }

    // do 4 samples
    for (size_t i = 0; i < 4; i++) {
      std::pair<std::unordered_set<Edge>, SampleSketchRet> query_ret = s_node->exhaustive_sample();
      if (query_ret.second != GOOD) {
        ASSERT_EQ(query_ret.first.size(), 0);
      }

      // assert everything returned is valid
      for (Edge e : query_ret.first) {
        ASSERT_GT(e.src, 0);
        ASSERT_LE(e.src, 10);
        ASSERT_GT(e.dst, e.src);
        ASSERT_LE(e.dst, 10);
      }

      // assert everything returned is unique
      std::set<Edge> unique_elms(query_ret.first.begin(), query_ret.first.end());
      ASSERT_EQ(unique_elms.size(), query_ret.first.size());
    }
    free(s_node);
  }
}

TEST_F(SupernodeTestSuite, TestPartialSparseSerialization) {
  Supernode* s_node = Supernode::makeSupernode(num_nodes, seed);
  Supernode* empty_node = Supernode::makeSupernode(*s_node);
  for (size_t i = 0; i < 10000; i++) {
    node_id_t src = rand() % (num_nodes-1);
    node_id_t dst = rand() % num_nodes;
    if (src == dst)
      ++dst;
    s_node->update(concat_pairing_fn(src, dst));
  }
  int sketches = Supernode::get_max_sketches();
  for (int beg = 0; beg < sketches / 4; beg++) {
    for (int num = sketches - beg; num > 3 * sketches / 4; num--) {
      auto file = std::fstream("./out_supernode.txt", std::ios::out | std::ios::binary);
      s_node->write_binary_range(file, beg, num, true);
      file.close();

      auto in_file = std::fstream("./out_supernode.txt", std::ios::in | std::ios::binary);
      Supernode* reheated = Supernode::makeSupernode(num_nodes, seed, in_file);
      in_file.close();

      for (int j = 0; j < beg; j++) {
        ASSERT_EQ(*reheated->get_sketch(j), *empty_node->get_sketch(j));
      }
      for (int j = beg; j < beg + num; ++j) {
        ASSERT_EQ(*reheated->get_sketch(j), *s_node->get_sketch(j));
      }
      for (int j = beg + num; j < sketches; ++j) {
        ASSERT_EQ(*reheated->get_sketch(j), *empty_node->get_sketch(j));
      }
      free(reheated);
    }
  }
  free(s_node);
  free(empty_node);
}

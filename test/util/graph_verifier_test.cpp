#include <gtest/gtest.h>
#include <graph_verifier.h>

const std::string fname = __FILE__;
size_t pos = fname.find_last_of("\\/");
const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0,pos);

constexpr size_t num_primes = 97;
const size_t primes[num_primes] {
    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,  59,
    61,  67,  71,  73,  79,  83,  89,  97,  101, 103, 107, 109, 113, 127, 131, 137, 139,
    149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337,
    347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439,
    443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509};

// finds a factor of x that is less than x.
static node_id_t find_a_factor(node_id_t x) {
  for (auto prime : primes) {
    if (prime >= x) break;
    if (x % prime == 0) return prime;
  }

  return 0;
}

TEST(GraphVerifierTest, TestCorrectNumCC) {
  GraphVerifier verifier(1024, curr_dir+"/../res/multiples_graph_1024.txt");

  ASSERT_EQ(78, verifier.get_num_kruskal_ccs());
}

TEST(GraphVerifierTest, TestEdgeVerifier) {
  GraphVerifier verifier(1024, curr_dir+"/../res/multiples_graph_1024.txt");
  // add edges of the form {i,2i}
  for (node_id_t i = 2; i < 512; ++i) {
    verifier.verify_edge({i, i*2});
  }
  // throw on nonexistent edge
  ASSERT_THROW(verifier.verify_edge({69,420}), BadEdgeException);
  ASSERT_THROW(verifier.verify_edge({420,69}), BadEdgeException);
}

TEST(GraphVerifierTest, TestVerifySpanningForest) {
  GraphVerifier verifier(1024, curr_dir+"/../res/multiples_graph_1024.txt");

  {
    // create a partial spanning forest
    std::unordered_set<node_id_t> adj_list[1024];
    for (node_id_t i = 2; i < 512; i++) {
      adj_list[i].insert(i*2);
    }

    // This spanning forest should be incorrect,
    // it is incomplete
    ASSERT_THROW(
      verifier.verify_spanning_forests(std::vector<SpanningForest>{SpanningForest(1024, adj_list)}),
      IncorrectForestException
    );
  }
  {
    // create a partial spanning forest
    std::unordered_set<node_id_t> adj_list[1024];
    for (node_id_t i = 2; i < 512; i++) {
      adj_list[i].insert(i*2);
    }
    adj_list[2].insert(5); // this is the bad edge
    for (node_id_t i = 6; i < 1024; i+=2) {
      adj_list[2].insert(i);
    }

    // This spanning forest should be incorrect,
    // it contains an edge not found in the original graph
    ASSERT_THROW(
      verifier.verify_spanning_forests(std::vector<SpanningForest>{SpanningForest(1024, adj_list)}),
      BadEdgeException
    );
  }
  {
    // This is a correct spanning forest
    std::unordered_set<node_id_t> adj_list[1024];
    for (node_id_t i = 2; i < 1024; i++) {
      node_id_t factor = find_a_factor(i);
      if (factor != 0) adj_list[factor].insert(i);
    }
    for (auto prime : primes) {
      if (prime == 2) continue;
      adj_list[prime].insert(prime * 2);
    }

    SpanningForest forest(1024, adj_list);
    verifier.verify_spanning_forests(std::vector<SpanningForest>{forest});
  }
}

#include <gtest/gtest.h>
#include <fstream>
#include <chrono>
#include "../../include/graph.h"

#ifdef VERIFY_SAMPLES_F
#include "../util/graph_verifier.h"
#endif

const int TRIALS = 10;

long double mean(std::vector<long double>& vec) {
  long double res = 0.0;
  for (auto num : vec) {
    res += num;
  }
  return res/vec.size();
}

/**
 * @param stream a filepath, rooted at ../res/
 * @param cum a filepath, rooted at ../res/
 */
long double run_expr(const std::string& stream, const std::string& cum) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "/" : fname.substr
        (0, pos+1) + "../res/";
  ifstream in{curr_dir + stream};
  Node num_nodes;
  in >> num_nodes;
  long m;
  in >> m;
  Node a, b;
  int t;
  Graph g{num_nodes};
  while (m--) {
    in >> t >> a >> b;
    g.update({{a, b}, INSERT});
  }
#ifdef VERIFY_SAMPLES_F
  g.set_cum_in(curr_dir + cum);
#endif
  auto start_time = std::chrono::steady_clock::now();
  g.connected_components();
  auto retval = static_cast<std::chrono::duration<long double>>
  (std::chrono::steady_clock::now() - start_time).count();
  std::cout << retval << std::endl;
  return retval;
}

TEST(Experiment, 1K) {
  std::vector<long double> results(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    results[i] = run_expr("1000_0.95_0.5.stream", "1000_0.95_0.5.cum");
  }
#ifdef EXT_MEM_POST_PROC_F
  std::cout << "Average time (new): ";
#else
  std::cout << "Average time (old): ";
#endif
  std::cout << mean(results) << endl;
}

TEST(Experiment, 2K) {
  std::vector<long double> results(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    results[i] = run_expr("2000_0.95_0.5.stream", "2000_0.95_0.5.cum");
  }
#ifdef EXT_MEM_POST_PROC_F
  std::cout << "Average time (new): ";
#else
  std::cout << "Average time (old): ";
#endif
  std::cout << mean(results) << endl;
}

TEST(Experiment, 3K) {
  std::vector<long double> results(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    results[i] = run_expr("3000_0.95_0.5.stream", "3000_0.95_0.5.cum");
  }
#ifdef EXT_MEM_POST_PROC_F
  std::cout << "Average time (new): ";
#else
  std::cout << "Average time (old): ";
#endif
  std::cout << mean(results) << endl;
}

TEST(Experiment, DISABLED_10K) {
  std::vector<long double> results(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    results[i] = run_expr("10000_0.95_0.5.stream", "10000_0.95_0.5.cum");
  }
#ifdef EXT_MEM_POST_PROC_F
  std::cout << "Average time (new): ";
#else
  std::cout << "Average time (old): ";
#endif
  std::cout << mean(results) << endl;
}

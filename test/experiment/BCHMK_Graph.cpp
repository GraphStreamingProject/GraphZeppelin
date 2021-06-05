#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <ctime>
#include <thread>
#include "../../include/graph.h"
#include "../util/graph_verifier.h"
#include "../util/graph_gen.h"

/*
 * Function which is run in a seperate thread and will query
 * the graph for the number of updates it has processed
 * the thread writes that information to a given file
 * @param total       the total number of edge updates
 * @param g           the graph object to query
 * @param start_time  the time that we started stream ingestion
 */
void query_insertions(uint64_t total, Graph *g, std::chrono::steady_clock::time_point start_time) {
  total = total * 2;                // we insert 2 edge updates per edge
  ofstream out{"runtime_data.txt"}; // open the outfile

  printf("Insertions\n");
  printf("Progress:                    | 0%%\r"); fflush(stdout);
  std::chrono::steady_clock::time_point start = start_time;
  uint64_t prev_updates = 0;

  while(true) {
    sleep(5);
    uint64_t updates = g->num_updates;
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = now - start;
    start = now; // reset start time to right after query

    // calculate the insertion rate and write to file
    uint64_t upd_delta = updates - prev_updates;
    // divide insertions per second by 2 because each edge is split into two updates
    // we care about edges per second not about stream updates
    float ins_per_sec = (((float)(upd_delta)) / diff.count()) / 2;

    prev_updates += upd_delta;
    out << (prev_updates / 2) << "\t" << ins_per_sec << "\n";
    
    if (updates >= total)
      break;

    // display the progress
    int percent = updates / (total * .05);
    printf("Progress:%s%s", std::string(percent, '=').c_str(), std::string(20 - percent, ' ').c_str());
    printf("| %i%% -- %.2f per second\r", percent * 5, ins_per_sec); fflush(stdout);
  }
  printf("Progress:====================| Done\n");
  std::chrono::duration<double> runtime = g->end_time - start_time;

  // calculate the insertion rate and write to file
  float ins_per_sec = (((float)(total)) / runtime.count()) / 2;
  out << "DONE\n";
  out << (total / 2) << "\t" << ins_per_sec << "\n";
  out.close();
  return;
}


TEST(Benchmark, BCHMKGraph) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  
  // before running this experiment, copy the test file to
  // the 'current_test.stream' file. This is so tests can
  // be automated.
  ifstream in{curr_dir + "/../res/current_test.stream"};
  Node num_nodes;
  in >> num_nodes;
  long m;
  in >> m;
  long total = m;
  Node a, b;
  uint8_t u;
  Graph g{num_nodes};

  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::thread querier(query_insertions, total, &g, start);

  while (m--) {
    in >> u >> a >> b;
    if (u == INSERT)
      g.update({{a, b}, INSERT});
    else
      g.update({{a,b}, DELETE});
  }
  printf("Number of connected components is %lu\n", g.connected_components().size());
  querier.join();
}

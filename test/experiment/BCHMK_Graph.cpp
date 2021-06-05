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
void query_insertions(uint64_t total, Graph *g, clock_t start_time) {
  total = total * 2;                // we insert 2 edge updates per edge
  ofstream out{"runtime_data.txt"}; // open the outfile

  printf("Insertions\n");
  printf("Progress:                    | 0%%\r"); fflush(stdout);
  clock_t start = start_time;
  clock_t true_start = start_time; // start will change. This tracks real start time
  uint64_t prev_updates = 0;
  while(true) {
    sleep(5);
    uint64_t updates = g->num_updates;
    clock_t diff = clock() - start;
    start = clock(); // reset start time to right after query

    // calculate the insertion rate and write to file
    uint64_t upd_delta = g->num_updates - prev_updates;
    // divide insertions per second by 2 because each edge is split into two updates
    // we care about edges per second not about stream updates
    float ins_per_sec = (((float)(upd_delta)) / diff) * CLOCKS_PER_SEC / 2;
    
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
  clock_t runtime = g->end_time - true_start;

  // calculate the insertion rate and write to file
  float ins_per_sec = (((float)(total)) / runtime) * CLOCKS_PER_SEC / 2;
  out << "DONE\n";
  out << (total / 2) << "\t" << ins_per_sec << "\n";
  out.close();
  return;
}


TEST(Benchmark, BCHMKGraph) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  ifstream in{curr_dir + "/../res/3000.test.stream"};
  Node num_nodes;
  in >> num_nodes;
  long m;
  in >> m;
  long total = m;
  Node a, b;
  uint8_t u;
  Graph g{num_nodes};

  clock_t start = clock();
  std::thread querier(query_insertions, total, &g, start);

  while (m--) {
    in >> u >> a >> b;
    if (u == INSERT)
      g.update({{a, b}, INSERT});
    else
      g.update({{a,b}, DELETE});
  }
  ASSERT_EQ(1, g.connected_components().size());
  querier.join();
}

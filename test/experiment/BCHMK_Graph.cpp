// #include <gtest/gtest.h>
// #include <fstream>
// #include <string>
// #include <ctime>
// #include "../../include/graph.h"
// #include "../util/graph_verifier.h"
// #include "../util/graph_gen.h"

// TEST(Benchmark, BCHMKGraph) {
//   const std::string fname = __FILE__;
//   size_t pos = fname.find_last_of("\\/");
//   const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
//   ifstream in{curr_dir + "/../res/1000_0.95_0.5.stream"};
//   Node num_nodes;
//   in >> num_nodes;
//   long m;
//   in >> m;
//   long total = m;
//   Node a, b;
//   uint8_t u;
//   Graph g{num_nodes};
//   printf("Insertions\n");
//   printf("Progress:                    | 0%%\r"); fflush(stdout);
//   clock_t start = clock();
//   while (m--) {
//     if ((total - m) % (int)(total * .05) == 0) {
//       clock_t diff = clock() - start;
//       float num_seconds = diff / CLOCKS_PER_SEC;
//       int percent = (total - m) / (total * .05);
//       printf("Progress:%s%s", std::string(percent, '=').c_str(), std::string(20 - percent, ' ').c_str());
//       printf("| %i%% -- %.2f per second\r", percent * 5, (total-m)/num_seconds); fflush(stdout);
//     }
//     in >> u >> a >> b;
//     //printf("a = %lu b = %lu\n", a, b);
//     if (u == INSERT)
//       g.update({{a, b}, INSERT});
//     else
//       g.update({{a,b}, DELETE});
//   }
//   printf("Progress:====================| Done\n");
//   ASSERT_EQ(1, g.connected_components().size());
// }

#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>

#include "../../include/graph.h"
#include "../../include/graph_stream.h"

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
  ofstream out{"runtime_stats.txt"}; // open the outfile

  printf("Insertions\n");
  printf("Progress:                    | 0%%\r"); fflush(stdout);
  std::chrono::steady_clock::time_point start = start_time;
  std::chrono::steady_clock::time_point prev  = start_time;
  uint64_t prev_updates = 0;
  int percent = 0;

  while(true) {
    sleep(5);
    uint64_t updates = g->num_updates;
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_diff = now - start;
    std::chrono::duration<double> cur_diff   = now - prev;

    // calculate the insertion rate and write to file
    uint64_t upd_delta = updates - prev_updates;
    // divide insertions per second by 2 because each edge is split into two updates
    // we care about edges per second not about stream updates
    int ins_per_sec = (((float)(upd_delta)) / cur_diff.count()) / 2;

    int amount = upd_delta / (total * .01);
    if (amount > 1) {
      percent += amount;
      out << percent << "% :\n";
      out << "Updates per second sinces last entry: " << ins_per_sec << "\n";
      out << "Time since last entry: " << (int) cur_diff.count() << "\n";
      out << "Total runtime so far: " << (int) total_diff.count() << "\n\n";

      prev_updates += upd_delta;
      prev = now; // reset start time to right after query
    }
    
    if (updates >= total)
      break;

    // display the progress
    int progress = updates / (total * .05);
    printf("Progress:%s%s", std::string(progress, '=').c_str(), std::string(20 - progress, ' ').c_str());
    printf("| %i%% -- %i per second\r", progress * 5, ins_per_sec); fflush(stdout);
  }
  printf("Progress:====================| Done      \n");
  out.close();
  return;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Incorrect number of arguments. "
                 "Expected one but got " << argc-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  // create the input stream with a buffer size of 32KB
  uint32_t buf_size = 32 * 1024;
  GraphStream stream(argv[1], buf_size);

  Graph g{stream.nodes};
  uint8_t u;
  uint32_t a, b;

  auto start = std::chrono::steady_clock::now();
  std::thread querier(query_insertions, stream.edges, &g, start);

  for(uint64_t i = 0; i < stream.edges; i++) {
    stream.parse_line(&u, &a, &b);

    if (u == INSERT)
      g.update({{a, b}, INSERT});
    else
      g.update({{a, b}, DELETE});
  }

  std::cout << "Starting CC" << std::endl;

  uint64_t num_CC = g.connected_components().size();

  querier.join();
  printf("querier done\n");
  long double time_taken = static_cast<std::chrono::duration<long double>>(g.end_time - start).count();
  long double CC_time = static_cast<std::chrono::duration<long double>>(g.CC_end_time - g.end_time).count();

  ofstream out{"runtime_stats.txt",  std::ofstream::out | std::ofstream::app}; // open the outfile
  std::cout << "Number of connected components is " << num_CC << std::endl;
  std::cout << "Writing runtime stats to runtime_stats.txt\n";

  std::chrono::duration<double> runtime  = g.end_time - start;

  // calculate the insertion rate and write to file
  // insertion rate measured in stream updates 
  // (not in the two sketch updates we process per stream update)
  float ins_per_sec = (((float)(stream.edges)) / runtime.count());
  out << "Procesing " <<stream.edges << " updates took " << time_taken << " seconds, " << ins_per_sec << " per second\n";

  out << "Connected Components algorithm took " << CC_time << " and found " << num_CC << " CC\n";
  out.close();
}
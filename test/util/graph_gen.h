#pragma once
#include <utility>

typedef struct genSet {
  long n;            // number of nodes
  double p;          // prob of edge between nodes
  double r;          // geometric insertion/removal
  int max_appearances;  // the maximum number of times an edge can show up
                            // in the stream. 0 for no limit.
  std::string out_file; // file to write stream
  std::string cum_out_file; // file to write cum graph
  genSet(long n, double p, double r, int max_appearances,
         std::string out_file, std::string cum_out_file)
         : n(n), p(p), r(r), max_appearances
         (max_appearances), out_file(std::move(out_file)), cum_out_file(std::move(cum_out_file)) {}
} GraphGenSettings;

/**
 * Generates a 1024-node graph with approximately 60,000 edge insert/deletes.
 * Writes stream output to sample.txt
 * Writes cumulative output to cum_sample.txt
 */
void generate_stream(const GraphGenSettings& settings =
      {1024,0.03,0.5,0,"./sample.txt", "./cum_sample.txt"});

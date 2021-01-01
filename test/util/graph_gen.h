#include <utility>

#ifndef TEST_GRAPH_GEN_H
#define TEST_GRAPH_GEN_H

typedef struct genSet {
  long n;            // number of nodes
  double p;          // prob of edge between nodes
  long m;           // number of edges
  bool graph_model;     // G(n,p) -- 0, G(n,m) -- 1
  int max_appearances;  // the maximum number of times an edge can show up
                            // in the stream
  std::string out_file; // file to write stream
  std::string cum_out_file; // file to write cum graph
  genSet(long n, double p, long m, bool graph_model, int max_appearances,
         std::string out_file, std::string cum_out_file)
         : n(n), p(p), m(m), graph_model(graph_model), max_appearances
         (max_appearances), out_file(std::move(out_file)), cum_out_file(std::move(cum_out_file)) {}
} GraphGenSettings;

/**
 * Generates a 1024-node graph with approximately 30,000 edge insert/deletes.
 * Writes stream output to sample.txt
 * Writes cumulative output to cum_sample.txt
 */
void generate_stream(GraphGenSettings settings =
      {1024,0.03,30000,0,4,"./sample.txt", "./cum_sample.txt"});

#endif //TEST_GRAPH_GEN_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <utility>
#include <vector>
#include <map>
#include "graph_gen.h"
#include "../../include/graph.h"

using namespace std;

void validate(int n, vector<pair<pair<int,int>,bool>>& stream,
              vector<pair<int,int>>& reduced_stream
) {
  bool* inserted = static_cast<bool *>(malloc(
        ((n + 1) * (n + 1) + 2) * sizeof(bool)));
  fill(inserted, inserted + (n + 1) * (n + 1) + 2, DELETE);
  for (auto entry : stream) {
    if (inserted[entry.first.first*n+entry.first.second] == entry.second) {
      free(inserted);
      throw std::runtime_error("Invalid stream, no output written");
    }
    inserted[entry.first.first*n+entry.first.second] = !inserted[entry.first.first*n+entry.first.second];
  }
  unsigned num_cum = 0;
  for (int i = 0; i < (n + 1) * (n + 1) + 2; ++i) {
    if (inserted[i] == INSERT) ++num_cum;
  }
  free(inserted);
  if (num_cum != reduced_stream.size()) throw std::runtime_error(
        "Mismatch with reduced stream, no output written");
  cout << "Successful!" << endl;
}

/**
 * takes a sample.gr file of random type and transforms it into a stream with
 * insertions and deletions.
 * Writes stream output to file (default sample.txt)
 * Writes cumulative output to file (default cum_sample.txt)
 */
void transform(GraphGenSettings settings) {
  srand(time(NULL));
  string str;
  ifstream in("./sample.gr");
  // strip header summary
  for (int i = 0; i < 7; ++i) {
    getline(in,str);
  }
  in >> str >> str;

  char a;
  int n, m, full_m = 0; in >> n >> m;
  int f,s,w;
  bool* inserted = static_cast<bool *>(malloc(
        ((n + 1) * (n + 1) + 2) * sizeof(bool)));
  vector<pair<pair<int,int>,bool>> stream;
  map<pair<int,int>,int> tot_edges;
  vector<pair<int,int>> cum_stream;
  fill(inserted, inserted+(n+1)*(n+1)+2, DELETE);
  for (int i = 0; i < m; ++i) {
    in >> a >> f >> s >> w;
    --f; --s; // adjustment for 0-indexing
    full_m += w;
    tot_edges[{f,s}] += w;
    while (w--) stream.push_back({{f,s},INSERT});
  }
  in.close();
  for (const auto& entry : tot_edges) {
    if (entry.second % 2) cum_stream.push_back(entry.first);
  }

  // write cumulative output
  ofstream cum_out(settings.cum_out_file);
  cum_out << n << " " << cum_stream.size() << endl;
  for (auto entry : cum_stream) {
    cum_out << entry.first << " " << entry.second << endl;
  }
  cum_out.close();

  // randomize
  for (int i = full_m-1; i > 0; --i) {
    f = rand()%(i+1);
    swap(stream[i], stream[f]);
  }

  for (int i = 0; i < full_m; ++i) {
    f = stream[i].first.first*n+stream[i].first.second;
    if (inserted[f] == DELETE) {
      stream[i].second = INSERT;
      inserted[f] = INSERT;
    } else {
      stream[i].second = DELETE;
      inserted[f] = DELETE;
    }
  }

  free(inserted);

  validate(n, stream, cum_stream);

  ofstream out(settings.out_file);
  out << n << " " << m << " " << full_m << endl;
  // output order: [type] [first node] [second node]
  for (auto edge : stream) {
    out << edge.second << " " << edge.first.first << " " << edge.first.second
         <<
         endl;
  }
  out.close();
}

/**
 * Takes a settings struct and writes the corresponding config file to
 * "./gtconfig"
 */
void generate_config(const GraphGenSettings& settings) {
  ofstream out {"./gtconfig"};
  out << "GRAPH_MODEL " << settings.graph_model << "\n";
  out << "n " << settings.n << "\n";
  out << "p " << settings.p << "\n";
  out << "m " << settings.m << "\n";
  out << "SELF_LOOPS 0\n";
  out << "MAX_WEIGHT " << settings.max_appearances << "\n";
  out << "MIN_WEIGHT 1\n";
  out << "STORE_IN_MEMORY 1\n"
         "SORT_EDGELISTS 0\n"
         "SORT_TYPE 1\n"
         "WRITE_TO_FILE 0" << endl;
  out.close();
}

void generate_stream(GraphGenSettings settings) {
  generate_config(settings);
  std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  string cmd_str = curr_dir + "/GTgraph-random -c ./gtconfig -o sample.gr";
  cout << "Running command:" << endl << cmd_str << endl;
  if (system(cmd_str.c_str())) {
    cout << "Could not generate graph. Aborting..." << endl;
    return;
  }
  transform(settings);
  cout << "Graph written to: " << settings.out_file << endl;
  cout << "Cumulative graph written to: " << settings.cum_out_file << endl;
}
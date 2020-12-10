#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include "graph_gen.h"

#define INSERT 1
#define DELETE 0
#define endl '\n'

using namespace std;

void validate(int n, vector<pair<pair<int,int>,bool>>& stream,
              vector<pair<int,int>>& reduced_stream
) {
  bool inserted[(n + 1) * (n + 1) + 2];
  fill(inserted, inserted + (n + 1) * (n + 1) + 2, 0);
  for (auto entry : stream) {
    if (inserted[entry.first.first*n+entry.first.second] == entry.second) {
      throw "Invalid stream, no output written";
    }
    inserted[entry.first.first*n+entry.first.second] = !inserted[entry.first.first*n+entry.first.second];
  }
  unsigned num_cum = 0;
  for (int i = 0; i < (n + 1) * (n + 1) + 2; ++i) {
    if (inserted[i]) ++num_cum;
  }
  if (num_cum != reduced_stream.size()) throw "Mismatch with reduced stream, "
                                              "no output written";
  cout << "Successful!" << endl;
}

/**
 * takes a sample.gr file of random type and transforms it into a stream with
 * insertions and deletions.
 * Writes stream output to sample.txt
 * Writes cumulative output to cum_sample.txt
 */
void transform() {
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
  int inserted[(n+1)*(n+1)+2];
  vector<pair<pair<int,int>,bool>> stream;
  stream.reserve(m*3);
  vector<pair<int,int>> cum_stream;
  cum_stream.reserve(m);
  fill(inserted, inserted+(n+1)*(n+1)+2, 0);
  for (int i = 0; i < m; ++i) {
    in >> a >> f >> s >> w;
    full_m += w;
    if (w%2) cum_stream.push_back({f,s});
    while (w--) stream.push_back({{f,s},INSERT});
  }
  in.close();

  // write cumulative output
  ofstream cum_out("./cum_sample.txt");
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
    if (inserted[f] % 2) {
      stream[i].second = DELETE;
    } else {
      stream[i].second = INSERT;
    }
    ++inserted[f];
  }

  validate(n, stream, cum_stream);

  ofstream out("./sample.txt");
  out << n << " " << m << " " << full_m << endl;
  // output order: [type] [first node] [second node]
  for (auto edge : stream) {
    out << edge.second << " " << edge.first.first << " " << edge.first.second
         <<
         endl;
  }
  out.close();
}

void generate_stream() {
  std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  string cmd_str = curr_dir + "/GTgraph-random -c " +
        curr_dir + "/gtgraph-random_config -o "
                             "sample.gr";
  cout << "Running command:" << endl << cmd_str << endl;
  if (system(cmd_str.c_str())) {
    cout << "Could not generate graph. Aborting..." << endl;
    return;
  }
  transform();
  cout << "Graph written to: sample.txt" << endl;
  cout << "Cumulative graph written to: cum_sample.txt" << endl;
}
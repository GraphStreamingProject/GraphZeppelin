#include "graph_gen.h"
#include "types.h"
#include "util.h"

#include <fstream>
#include <algorithm>
#include <random>
#include <iostream>

#define endl '\n'

typedef uint32_t ul;
typedef uint64_t ull;

const ull ULLMAX = std::numeric_limits<ul>::max();


std::ofstream& operator<< (std::ofstream &os, const std::pair<ull,ull> p) {
  os << p.first << " " << p.second;
  return os;
}

void write_edges(long n, double p, const std::string& out_f) {
  ul num_edges = (n*(n-1))/2;
  ull* arr = (ull*) malloc(num_edges*sizeof(ull));
  ul e = 0;
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = i+1; j < n; ++j) {
      arr[e++] = concat_pairing_fn(i, j);
    }
  }
  std::shuffle(arr,arr+num_edges, std::mt19937(std::random_device()()));
  std::ofstream out(out_f);
  ul m = (ul) (num_edges*p);
  out << n << " " << m << endl;

  while (m--) {
    Edge e = inv_concat_pairing_fn(arr[m]);
    out << e.src << " " << e.dst << endl;
  }
  out.flush();
  out.close();
  free(arr);
}

void insert_delete(double p, int max_appearances, const std::string& in_file,
                   const std::string& out_file) {
  std::ifstream in(in_file);
  std::ofstream out(out_file);
  int n; ul m; in >> n >> m;
  long long full_m = m;
  ull ins_del_arr[(ul)log2(m)+2];
  std::fill(ins_del_arr,ins_del_arr + (ul)log2(m)+2,0);
  ins_del_arr[0] = m;
  if (max_appearances == 0) {
    for (unsigned i = 0; ins_del_arr[i] > 1; ++i) {
      ins_del_arr[i + 1] = (ull) (ins_del_arr[i] * p);
      full_m += ins_del_arr[i + 1];
    }
  } else {
    for (int i = 0; i < max_appearances - 1; ++i) {
      ins_del_arr[i + 1] = (ull) (ins_del_arr[i] * p);
      full_m += ins_del_arr[i + 1];
    }
  }

  out << n << " " << full_m << endl;

  ull* memoized = (ull*) malloc(ins_del_arr[1]*sizeof(ull));
  ul a,b;

  for (unsigned i=0;i<ins_del_arr[1];++i) {
    in >> a >> b;
    out << "0 " << a << " " << b << endl;
    memoized[i] = concat_pairing_fn(a, b);
  }

  for (unsigned i=ins_del_arr[1];i<m;++i) {
    in >> a >> b;
    out << "0 " << a << " " << b << endl;
  }

  in.close();

  unsigned stopping = 1;
  if (max_appearances == 0) {
    for (; ins_del_arr[stopping] >= 1; ++stopping);
  } else {
    stopping = max_appearances;
  }
  for (unsigned i = 1; i < stopping; ++i) {
    int temp = i % 2;
    for (unsigned j = 0; j < ins_del_arr[i]; ++j) {
      out << temp << " ";
      Edge e = inv_concat_pairing_fn(memoized[j]);
      out << e.src << " " << e.dst << endl;
    }
  }
  out.flush();
  out.close();
  free(memoized);
}

void write_cumul(const std::string& stream_f, const std::string& cumul_f) {
  std::ifstream in(stream_f);
  std::ofstream out(cumul_f);
  int n; ull m; in >> n >> m;
  std::vector<std::vector<bool>> adj(n,std::vector<bool>(n,false));
  bool type;
  int a,b;
  for (ull i=1;i<=m;++i) {
    in >> type >> a >> b;
    if ((type == INSERT && adj[a][b] == 1) || (type == DELETE && adj[a][b] == 0)) {
      std::cerr << "Insertion/deletion error at line " << i
                << " in " << stream_f;
      return;
    }
    adj[a][b] = !adj[a][b];
  }

  in.close();

  // write cumul output
  ull m_cumul = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (adj[i][j]) ++m_cumul;
    }
  }
  out << n << " " << m_cumul << endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (adj[i][j]) out << i << " " << j << endl;
    }
  }
  out.flush();
  out.close();
}

void generate_stream(const GraphGenSettings& settings) {
  write_edges(settings.n, settings.p, "./TEMP_F");
  insert_delete(settings.r, settings.max_appearances, "./TEMP_F", settings
  .out_file);
  write_cumul(settings.out_file,settings.cumul_out_file);
}

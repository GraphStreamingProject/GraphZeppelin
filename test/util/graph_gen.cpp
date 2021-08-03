#include <fstream>
#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include "graph_gen.h"
#include "../../include/graph.h"
using uint128_t = boost::multiprecision::uint128_t;

typedef uint32_t ul;
typedef uint64_t ull;

const ull ULLMAX = std::numeric_limits<ul>::max();


std::ofstream& operator<< (std::ofstream &os, const std::pair<ull,ull> p) {
  os << p.first << " " << p.second;
  return os;
}

void write_edges(long n, double p, std::string out_f) {
  ul num_edges = (n*(n-1))/2;
  ul* arr = (ul*) malloc(num_edges*sizeof(ul));
  for (unsigned i=0;i<num_edges;++i) {
    arr[i] = i;
  }
  std::random_shuffle(arr,arr+num_edges);
  std::ofstream out(out_f);
  ul m = (ul) (num_edges*p);
  out << n << " " << m << std::endl;

  while (m--) {
    out << inv_nondir_non_self_edge_pairing_fn(arr[m]) << std::endl;
  }
  out.flush();
  out.close();
  free(arr);
}

void insert_delete(double p, int max_appearances, std::string in_file,
                   std::string out_file) {
  std::ifstream in(in_file);
  std::ofstream out(out_file);
  int n; ul m; in >> n >> m;
  long long full_m = m;
  ul ins_del_arr[(ul)log2(m)+2];
  std::fill(ins_del_arr,ins_del_arr + (ul)log2(m)+2,0);
  ins_del_arr[0] = m;
  if (max_appearances == 0) {
    for (unsigned i = 0; ins_del_arr[i] > 1; ++i) {
      ins_del_arr[i + 1] = (ul) (ins_del_arr[i] * p);
      full_m += ins_del_arr[i + 1];
    }
  } else {
    for (int i = 0; i < max_appearances - 1; ++i) {
      ins_del_arr[i + 1] = (ul) (ins_del_arr[i] * p);
      full_m += ins_del_arr[i + 1];
    }
  }

  out << n << " " << full_m << std::endl;

  ul* memoized = (ul*) malloc(ins_del_arr[1]*sizeof(ul));
  ul a,b;

  for (unsigned i=0;i<ins_del_arr[1];++i) {
    in >> a >> b;
    out << "0 " << a << " " << b << std::endl;
    memoized[i] = nondirectional_non_self_edge_pairing_fn(a, b);
  }

  for (unsigned i=ins_del_arr[1];i<m;++i) {
    in >> a >> b;
    out << "0 " << a << " " << b << std::endl;
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
      out << inv_nondir_non_self_edge_pairing_fn(memoized[j]) << std::endl;
    }
  }
  out.flush();
  out.close();
  free(memoized);
}

void write_cum(std::string stream_f, std::string cum_f) {
  std::ifstream in(stream_f);
  std::ofstream out(cum_f);
  int n; ull m; in >> n >> m;
  std::vector<std::vector<bool>> adj(n,std::vector<bool>(n,false));
  bool type;
  int a,b;
  for (ull i=1;i<=m;++i) {
    in >> type >> a >> b;
    adj[a][b] = !adj[a][b];
  }

  in.close();
  std::cout << "Done reading input" << std::endl;
  // write cum output
  ull m_cum = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (adj[i][j]) ++m_cum;
    }
  }
  out << n << " " << m_cum << endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (adj[i][j]) out << i << " " << j << std::endl;
    }
  }
  out.flush();
  out.close();
  std::cout << "Done writing output" << std::endl;
}

void generate_stream(GraphGenSettings settings) {
  write_edges(settings.n, settings.p, "./TEMP_F");
  insert_delete(settings.r, settings.max_appearances, "./TEMP_F", settings
  .out_file);
  write_cum(settings.out_file,settings.cum_out_file);
}

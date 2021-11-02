#include <fstream>
#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include "../../../include/test/efficient_gen.h"
#include "../../../include/types.h"

#define endl '\n'

using uint128_t = boost::multiprecision::uint128_t;

typedef uint32_t ul;
typedef uint64_t ull;

const ull ULLMAX = std::numeric_limits<ul>::max();

ull nondirectional_non_self_edge_pairing_fn(ull i, ull j) {
  // swap i,j if necessary
  if (i > j) {
    std::swap(i,j);
  }
  ull jm = j-1ull;
  if ((j & 1ull) == 0ull) j>>=1ull;
  else jm>>=1ull;
  if (ULLMAX/j < jm)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  j*=jm;
  if (ULLMAX-j < i)
    throw std::overflow_error("Computation would overflow unsigned long long max");
  return i+j;
}

std::pair<ull, ull> inv_nondir_non_self_edge_pairing_fn(ull idx) {
  uint128_t eidx = 8ull*(uint128_t)idx + 1ull;
  eidx = sqrt(eidx)+1ull;
  eidx/=2ull;
  ull i,j = (ull) eidx;
  if ((j & 1ull) == 0ull) i = idx-(j>>1ull)*(j-1ull);
  else i = idx-j*((j-1ull)>>1ull);
  return {i, j};
}

std::ofstream& operator<< (std::ofstream &os, const std::pair<ull,ull> p) {
  os << p.first << " " << p.second;
  return os;
}

void write_edges(int n, double p, std::string out_f) {
  ul num_edges = (n*(n-1))/2;
  ul* arr = (ul*) malloc(num_edges*sizeof(ul));
  for (unsigned i=0;i<num_edges;++i) {
    arr[i] = i;
  }
  std::random_shuffle(arr,arr+num_edges);
  std::ofstream out(out_f);
  ul m = (ul) (num_edges*p);
  out << n << " " << m << endl;

  while (m--) {
    out << inv_nondir_non_self_edge_pairing_fn(arr[m]) << endl;
  }

  out.close();
  free(arr);
}

void insert_delete(double p, std::string in_file, std::string out_file) {
  std::ifstream in(in_file);
  std::ofstream out(out_file);
  int n; ul m; in >> n >> m;
  long long full_m = m;
  ul ins_del_arr[(ul)log2(m)+2];
  std::fill(ins_del_arr,ins_del_arr + (ul)log2(m)+2,0);
  ins_del_arr[0] = m;
  for (unsigned i = 0; ins_del_arr[i] > 1; ++i) {
    ins_del_arr[i+1] = (ul)(ins_del_arr[i]*p);
    full_m += ins_del_arr[i+1];
  }
  
  out << n << " " << full_m << endl;
  
  ul* memoized = (ul*) malloc(ins_del_arr[1]*sizeof(ul));
  ul a,b;
  
  for (unsigned i=0;i<ins_del_arr[1];++i) {
    in >> a >> b;
    out << "0 " << a << " " << b << endl;
    memoized[i] = nondirectional_non_self_edge_pairing_fn(a, b);
  }

  for (unsigned i=ins_del_arr[1];i<m;++i) {
    in >> a >> b;
    out << "0 " << a << " " << b << endl;
  }

  for (unsigned i = 1; ins_del_arr[i] >= 1; ++i) {
    int temp = i%2;
    for (unsigned j=0;j<ins_del_arr[i];++j) {
      out << temp << " ";
      out << inv_nondir_non_self_edge_pairing_fn(memoized[j]) << endl;
    }
  }
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
}

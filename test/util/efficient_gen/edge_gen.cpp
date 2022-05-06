#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include "../../../include/test/efficient_gen.h"
#include "../../../include/types.h"

typedef uint32_t ul;
typedef uint64_t ull;

const ull ULLMAX = std::numeric_limits<ull>::max();
const uint8_t num_bits = sizeof(node_id_t) * 8;

ull concat_pairing_fn(ul i, ul j) {
  // swap i,j if necessary
  if (i > j) {
    std::swap(i,j);
  }
  return ((ull)i << num_bits) | j;
}

std::pair<ul, ul> inv_concat_pairing_fn(ull idx) {
  ul j = idx & 0xFFFFFFFF;
  ul i = idx >> num_bits;
  return {i, j};
}

std::ofstream& operator<< (std::ofstream &os, const std::pair<ull,ull> p) {
  os << p.first << " " << p.second;
  return os;
}

void write_edges(ul n, double p, std::string out_f) {
  ull num_edges = ((ull)n*(n-1))/2;
  ull* arr = (ull*) malloc(num_edges*sizeof(ull));
  ul idx = 0;

  std::cout << "Generating possible edges" << std::endl;
  for (unsigned i=0; i < n; ++i) {
    for (unsigned j=i+1;j < n; ++j) {
      arr[idx++] = concat_pairing_fn(i, j);
    }
  }

  std::cout << "Permuting edges" << std::endl;  
  std::random_shuffle(arr,arr+num_edges);
  std::ofstream out(out_f);
  ull m = (ull) (num_edges*p);
  out << n << " " << m << std::endl;

  std::cout << "Writing edges to file" << std::endl;
  while (m--) {
    out << inv_concat_pairing_fn(arr[m]) << std::endl;
  }

  out.close();
  free(arr);
}

void insert_delete(double p, std::string in_file, std::string out_file) {
  std::cout << "Deleting and reinserting some edges" << std::endl;
  std::ifstream in(in_file);
  std::ofstream out(out_file);
  int n; ull m; in >> n >> m;
 
  ull full_m = m;
  ull ins_del_arr[(ull)log2(m)+2];
  std::fill(ins_del_arr,ins_del_arr + (ull)log2(m)+2,0);
  ins_del_arr[0] = m;
  for (unsigned i = 0; ins_del_arr[i] > 1; ++i) {
    ins_del_arr[i+1] = (ul)(ins_del_arr[i]*p);
    full_m += ins_del_arr[i+1];
  }
  
  out << n << " " << full_m << std::endl;
  
  ull* memoized = (ull*) malloc(ins_del_arr[1]*sizeof(ull));
  ul a,b;
  
  for (unsigned i=0;i<ins_del_arr[1];++i) {
    in >> a >> b;
    out << "0 " << a << " " << b << std::endl;
    memoized[i] = concat_pairing_fn(a, b);
  }

  for (unsigned i=ins_del_arr[1];i<m;++i) {
    in >> a >> b;
    out << "0 " << a << " " << b << std::endl;
  }

  for (unsigned i = 1; ins_del_arr[i] >= 1; ++i) {
    int temp = i%2;
    for (unsigned j=0;j<ins_del_arr[i];++j) {
      out << temp << " ";
      out << inv_concat_pairing_fn(memoized[j]) << std::endl;
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
  out << n << " " << m_cumul << std::endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (adj[i][j]) out << i << " " << j << std::endl;
    }
  }
}

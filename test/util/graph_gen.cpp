#include <fstream>
#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include <sstream>
#include <string>
#include "graph_gen.h"
#include "../../include/graph.h"
#include "../../include/types.h"

#define endl '\n'

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
  out << n << " " << m << endl;

  while (m--) {
    out << inv_nondir_non_self_edge_pairing_fn(arr[m]) << endl;
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
      out << inv_nondir_non_self_edge_pairing_fn(memoized[j]) << endl;
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
    if ((type == INSERT && adj[a][b] == 1) || (type == DELETE && adj[a][b] == 0)) {
      std::cerr << "Insertion/deletion error at line " << i
                << " in " << stream_f;
      return;
    }
    adj[a][b] = !adj[a][b];
  }

  in.close();

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
      if (adj[i][j]) out << i << " " << j << endl;
    }
  }
  out.flush();
  out.close();
}

void generate_stream(GraphGenSettings settings) {
  write_edges(settings.n, settings.p, "./TEMP_F");
  insert_delete(settings.r, settings.max_appearances, "./TEMP_F", settings
  .out_file);
  write_cum(settings.out_file,settings.cum_out_file);
}

// Warning: pastes the file names directly into the string given to system(),
// can't contain spaces or any other special shell characters
void write_edges_extmem(node_t n, double p, std::string out_file,
    std::string temp_file) {
  std::ofstream ofs_temp(temp_file);
  for (node_t i = 1; i < n; ++i) {
    for (node_t j = 0; j < i; j++) {
      for (int chars = 0; chars < 8; chars++) {
        ofs_temp << "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[rand() % 64];
      }
      ofs_temp << ' ' << j << ' ' << i << '\n';
    }
  }
  ofs_temp.close();

  vec_t num_edges = n*(n-1)/2*p;
  std::ofstream ofs_out(out_file);
  ofs_out << n << ' ' << num_edges << '\n';
  ofs_out.close();

  std::ostringstream cmd;
  cmd << "sort -k1,1 " << temp_file << "|head -n" << num_edges <<
      "|cut -d' ' -f2->>" << out_file;
  system(cmd.str().c_str());
}

void insert_delete_extmem(double r, int max_appearances, std::string in_file,
    std::string out_file, std::string cum_out_file, std::string temp_file) {
  node_t n;
  vec_t num_edges;
  std::ifstream ifs (in_file);
  ifs >> n >> num_edges;

  // Simulate a pass through insertions/deletions to get total stream length
  vec_t num_ins_del = num_edges, stream_len = 0, cum_stream_len = 0;
  bool deletion = false;
  for (int appearances = 0; num_ins_del > 0 &&
      (max_appearances == 0 || appearances < max_appearances); appearances++) {
    stream_len += num_ins_del;
    if (!deletion) {
      cum_stream_len += num_ins_del - static_cast<vec_t>(num_ins_del * r);
    }
    num_ins_del *= r;
    deletion = !deletion;
  }

  std::ofstream ofs_out (out_file), ofs_cum_out(cum_out_file),
      ofs_temp(temp_file);
  ofs_out << n << ' ' << stream_len << '\n';
  ofs_cum_out << n << ' ' << cum_stream_len << '\n';

  std::streampos sp_edge_beg = ifs.tellg();
  num_ins_del = num_edges;
  deletion = false;
  for (int appearances = 0; num_ins_del > 0 &&
      (max_appearances == 0 || appearances < max_appearances); appearances++) {
    node_t a, b;
    ifs.seekg(sp_edge_beg);
    for (vec_t i = 0; i < num_ins_del; i++) {
      ifs >> a >> b;
      ofs_out << deletion << ' ' << a << ' ' << b << '\n';
      if (!deletion && i >= static_cast<vec_t>(num_ins_del * r)) {
        ofs_temp << a << ' ' << b << '\n';
      }
    }
    stream_len += num_ins_del;
    num_ins_del *= r;
    deletion = !deletion;
  }
  ifs.close();
  ofs_out.close();
  ofs_cum_out.close();
  ofs_temp.close();
  system(("sort -k1,1n -k2,2n " + temp_file + ">>" + cum_out_file).c_str());
}

void generate_stream_extmem(GraphGenSettings settings) {
  write_edges_extmem(settings.n, settings.p, "./TEMP_F", "./TEMP_EDGE_PERM");
  insert_delete_extmem(settings.r, settings.max_appearances, "./TEMP_F",
      settings.out_file, settings.cum_out_file, "./TEMP_UNSORTED_CUM");
}

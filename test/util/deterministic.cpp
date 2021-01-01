#include <fstream>
#include <map>
#include "deterministic.h"

//Node* parent;
//Node* size;

Node dsu_find(Node i, Node* parent) {
  if (parent[i] == i) return i;
  return parent[i] = dsu_find(parent[i], parent);
}

void dsu_union(Node i, Node j, Node* parent, Node* size) {
  i = dsu_find(i, parent); j = dsu_find(j, parent);
  if (size[i] < size[j]) std::swap(i,j);
  parent[j] = i;
  size[i] += size[j];
}

GraphVerifier::GraphVerifier(const string &input_file) {
  kruskal_ref = kruskal(input_file);
  ifstream in(input_file);
  Node n,m; in >> n >> m;
  Node a,b;
  parent = (Node*) malloc(n*sizeof(Node));
  size = (Node*) malloc(n*sizeof(Node));
  for (unsigned i = 0; i < n; ++i) {
    boruvka_cc.push_back({i});
    det_graph.emplace_back();
    parent[i] = i;
    size[i] = 1;
  }
  while (m--) {
    in >> a >> b;
    det_graph[a].insert(b);
    det_graph[b].insert(a);
  }
  in.close();
}

GraphVerifier::~GraphVerifier() {
  free(parent);
  free(size);
}

std::vector<std::set<Node>> kruskal(const string& input_file) {
  ifstream in(input_file);
  Node n, m; in >> n >> m;
  Node* parent = (Node*) malloc(n*sizeof(Node));
  Node* size = (Node*) malloc(n*sizeof(Node));

  for (unsigned i = 0; i < n; ++i) {
    parent[i] = i;
    size[i] = 1;
  }
  int a,b;
  while (m--) {
    in >> a >> b;
    dsu_union(a,b, parent, size);
  }
  in.close();

  std::map<Node, std::set<Node>> temp;
  for (unsigned i = 0; i < n; ++i) {
    temp[dsu_find(i, parent)].insert(i);
  }
  free(parent);
  free(size);

  std::vector<std::set<Node>> retval;
  retval.reserve(temp.size());
  for (const auto& entry : temp) {
    retval.push_back(entry.second);
  }
  return retval;
}

void GraphVerifier::verify_edge(Edge edge) {
  Node f = dsu_find(edge.first,parent);
  Node s = dsu_find(edge.second,parent);
  if (boruvka_cc[f].find(edge.second) != boruvka_cc[f].end()
  || boruvka_cc[s].find(edge.first) != boruvka_cc[s].end())
    throw BadEdgeException();
  if (det_graph[edge.first].find(edge.second) == det_graph[edge.first].end())
    throw BadEdgeException();

  // if all checks pass, merge supernodes
  if (size[f] < size[s])
    std::swap(f,s);
  for (auto& i : boruvka_cc[s]) boruvka_cc[f].insert(i);
  dsu_union(f, s, parent, size);
}

void GraphVerifier::verify_cc(Node node) {
  node = dsu_find(node,parent);
  for (const auto& cc : kruskal_ref) {
    if (boruvka_cc[node] == cc) return;
  }
  throw NotCCException();
}

void GraphVerifier::verify_soln(vector<set<Node>> &retval) {
  vector<set<Node>> temp {retval};
  sort(temp.begin(),temp.end());
  sort(kruskal_ref.begin(),kruskal_ref.end());
  assert(kruskal_ref == temp);
  cout << "Solution ok: " << retval.size() << " CCs found." << endl;
}

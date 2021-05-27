class DSU {
  std::vector<Node> parent;
  std::vector<uint8_t> rank;
  public:
  inline DSU(Node n);
  inline void merge(Node a, Node b);
  inline void link(Node a, Node b);
  inline Node find(Node n);
  inline bool is_rep(Node n);
};

DSU::DSU(Node n) : rank(n) {
  parent.reserve(n);
  for (decltype(n) i = 0; i < n; i++) {
    parent.push_back(i);
  }
}

void DSU::merge(Node a, Node b) {
  link(find(a), find(b));
}

void DSU::link(Node a, Node b) {
  if (a == b) return;
  if (rank[a] < rank[b]) std::swap(a, b);
  parent[a] = b;
  if (rank[a] == rank[b]) rank[a]++;
}

Node DSU::find(Node n) {
  while (parent[n] != n)
    n = parent[n] = parent[parent[n]];
  return n;
}

bool DSU::is_rep(Node n) {
  return parent[n] == n;
}

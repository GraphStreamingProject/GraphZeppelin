//
// Created by victor on 7/13/21.
//

#include <iostream>

using namespace std;

#define endl '\n'

typedef long long int ll;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  ll n, m; cin >> n >> m;

  bool** adj = static_cast<bool **>(malloc(n * sizeof(bool *)));
  for (int i = 0; i < n; ++i) {
    adj[i] = static_cast<bool *>(malloc(n * sizeof(bool)));
    fill(adj[i], adj[i] + n, false);
  }

  ll t, a, b;
  while (m--) {
    cin >> t >> a >> b;
    adj[a][b] = !adj[a][b];
    // ahmed: Since we're computing WCC, it would be unfair to needlessly
    // symmetrize the input graph.  
//    adj[b][a] = !adj[b][a]; // undirected edges
  }

  m = 0;

  ll offsets[n];
  fill(offsets, offsets + n, 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (adj[i][j]) {
        ++offsets[i];
        ++m;
      }
    }
  }

  // ahmed: offsets are cumulative.
  ll sum = 0; 
  for(int i = 0; i < n; i++)
  {
	sum += offsets[i];
	offsets[i] = sum;
  }

  // output
  cout << "AdjacencyGraph" << endl;
  cout << n << endl;
  cout << m << endl;
  for (int i = 0; i < n; ++i) {
    cout << offsets[i] << endl;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (adj[i][j]) {
        cout << j << endl;
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    free(adj[i]);
  }
  free(adj);

  return 0;
}

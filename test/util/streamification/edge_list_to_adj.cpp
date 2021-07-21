#include <iostream>
#include <vector>

using namespace std;

#define endl '\n'

int main() 
{
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  long unsigned int n, m; cin >> n >> m;

  std::vector<std::vector<long unsigned int>> adj_list (n);

  long unsigned int a, b;
  for (int i = 0; i < m; i++)
  {
    cin >> a >> b;
    adj_list[a].push_back(b);
  }

  // output
  cout << "AdjacencyGraph" << endl;
  cout << n << endl;
  cout << m << endl;
  long unsigned int sum = 0;
  for (int i = 0; i < n; ++i) 
  {
    sum += adj_list[i].size();
    cout << sum << endl;
  }

  for (auto neighbors : adj_list) 
    for (auto neighbor : neighbors) 
        cout << neighbor << endl; 

  return 0;
}
